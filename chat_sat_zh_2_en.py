import json

import torch
from sat import AutoModel
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.model.official import ChatGLM2Model
from sat.model.finetune.lora2 import LoraMixin
from tqdm import tqdm


class FineTuneModel(ChatGLM2Model):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kw_args)
        self.add_mixin("lora", LoraMixin(args.num_layers, 10), reinit=True)

def chat_response(query, model, tokenizer,
        max_length: int = 256, num_beams=1, top_p=0.7, top_k=0, temperature=0.95):
    prompt = f"[Round 0]\n\n问：{query}\n\n答："
    inputs = tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id])
    strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id], num_beams=num_beams, consider_end=True)
    
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy
    )[0] # drop memory
    
    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq
    output_list = list(output)

    response = tokenizer.decode(output_list[0])
    return response


def split_translate(query, model, tokenizer, max_length: int = 256, num_beams=1, top_p=0.7, top_k=0, temperature=0.95):
    if len(query) < 15:
        return chat_response(query, model, tokenizer, max_length, num_beams, top_p, top_k, temperature).split(sep="答：")[-1]
    res_query = ""
    for sent in query.split('。'):
        res_sent = ""
        cur_sent = ""
        for sub_sent in sent.split('，'):
            cur_sent += sub_sent+"，"
            if len(cur_sent) >= 50:
                cur_trans = chat_response(cur_sent, model, tokenizer, max_length, num_beams, top_p, top_k, temperature).split(sep="答：")[-1]
                res_sent += cur_trans.rstrip(",.?!")+","
                cur_sent = ""
        if cur_sent != "":
            cur_trans = chat_response(cur_sent, model, tokenizer, max_length, num_beams, top_p, top_k, temperature).split(sep="答：")[-1]
            res_sent += cur_trans
        res_sent = res_sent.rstrip(",.?!")+"."
        res_query += res_sent+" "
    res_query = res_query.strip()
    return res_query


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.01)
    args = parser.parse_args()

    # load model
    # model, model_args = FineTuneModel.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690270701725/checkpoints/finetune-chatglm-6b-lora-07-25-15-39', 
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690281036622/checkpoints/finetune-chatglm2-6b-07-25-18-31',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690278447134/checkpoints/finetune-chatglm-6b-lora-07-25-17-48',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690281747909/checkpoints/finetune-chatglm2-6b-07-25-18-53',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690296760278/checkpoints/finetune-chatglm2-6b-07-25-22-53',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-07-27-10-19',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-07-28-10-26',
    model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/sat-finetune-sample/checkpoints/finetune-chatglm2-6b-08-08-10-44',
    args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
    ))
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    input_path = "/nxchinamobile2/shared/img_datasets/cleaned_instructions/image_caption/diy_caption_v1/meta.jsonl"
    output_path = "/nxchinamobile2/shared/jjh/projects/sat-finetune-sample/sat_zh_2_cn_100.jsonl"
    with open(input_path, 'r', encoding='utf-8') as f1, open(output_path, 'w', encoding='utf-8') as f2:
        for i, line in enumerate(tqdm(f1)):
            data = json.loads(line)
            data['prompt'] = chat_response(data['prompt'], model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k).split(sep="答：")[-1]
            # print(data['prompt'])
            # data['txt'] = chat_response(data['txt'], model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k).split(sep="答：")[-1]
            data['txt'] = split_translate(data['txt'], model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)
            # data['details'] = chat_response(data['details'], model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k).split(sep="答：")[-1]
            data['details'] = split_translate(data['details'], model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)

            f2.write(json.dumps(data, ensure_ascii=False) + '\n')
            if i == 50:
                break
    f1.close()
    f2.close()
