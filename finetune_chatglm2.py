import os
import torch
import argparse

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.model.official import ChatGLM2Model
from sat.model.finetune import PTuningV2Mixin
from sat.model.finetune.lora2 import LoraMixin

class FineTuneModel(ChatGLM2Model):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kw_args)
        if args.use_ptuning:
            self.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_multi_query_heads, args.pre_seq_len))
        if args.use_lora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank), reinit=True)
        self.args = args

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM2-finetune', 'ChatGLM2 finetune Configurations')
        group.add_argument('--pre_seq_len', type=int, default=8)
        group.add_argument('--lora_rank', type=int, default=32)
        group.add_argument('--use_ptuning', action="store_true")
        group.add_argument('--use_lora', action="store_true")
        return super().add_model_specific_args(parser)

    # def disable_untrainable_params(self):
    #     enable = []
    #     if self.args.use_ptuning:
    #         enable.extend(['ptuning'])
    #     if self.args.use_lora:
    #         enable.extend(['matrix_A', 'matrix_B'])
    #     for n, p in self.named_parameters():
    #         flag = False
    #         for e in enable:
    #             if e.lower() in n.lower():
    #                 flag = True
    #                 break
    #         if not flag:
    #             p.requires_grad_(False)


from transformers import DataCollatorForSeq2Seq
def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    
    return tokens, labels


from torch.nn import CrossEntropyLoss

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits = model(input_ids=tokens)[0]
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {'loss': loss}


def create_dataset_function(path, args):
    tokenizer = get_tokenizer()

    def preprocess_function_train(examples):
        max_seq_length = args.max_source_length + args.max_target_length + 2
        prefix = args.source_prefix if args.source_prefix is not None else ""

        model_inputs =[]
        for i in range(len(examples)):
            query, answer = examples[i]['prompt'], examples[i]['response']
            prompt = tokenizer.build_prompt(query, None) # [Round 1]\n\n问：{query}\n\n答：
            prompt = prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                        max_length=args.max_source_length)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                        max_length=args.max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
            
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

            ret = {}
            ret["input_ids"] = input_ids
            ret["labels"] = labels
            model_inputs.append(ret)

        return model_inputs
    
    # load jsonl from path and process
    import json
    with open(path, 'r', encoding='utf-8') as f:
        dataset = []
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    dataset = preprocess_function_train(dataset)
    
    return dataset


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_source_length', type=int)
    py_parser.add_argument('--max_target_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
    py_parser.add_argument('--source_prefix', type=str, default="")
    py_parser = FineTuneModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    model_type = '/gpfs/gpfs1/zphz/official_pretrains/sat/glm2-6b'
    # model_type = 'chatglm2-6b'
    model, args = FineTuneModel.from_pretrained(model_type, args)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/gpfs/gpfs1/zphz/official_pretrains/hugging_face/chatglm2-6b', trust_remote_code=True)
    get_tokenizer(outer_tokenizer=tokenizer)
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, collate_fn=data_collator)