import torch
import os
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.model.official import ChatGLM2Model
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import json


class CoyoDataset(Dataset):
    def __init__(self, path, tokenizer, device, max_length: int = 256):
        self.path = path
        self.tokenizer = tokenizer
        self.original_datas = []
        self.caption_prompts = []
        self.max_length = max_length
        self.device = device

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.original_datas.append(data)
                self.caption_prompts.append(f"[Round 0]\n\n问：{data['caption']}\n\n答：")

    def __len__(self):
        return len(self.original_datas)

    def __getitem__(self, idx):
        datas = self.original_datas[idx]
        caption_prompts = self.caption_prompts[idx]
        caption_tensors = self.tokenizer(caption_prompts, return_tensors="pt").to(self.device)['input_ids']
        caption_tensors = torch.cat(
            [caption_tensors, torch.tensor([-1] * (self.max_length - caption_tensors.shape[-1]), device=caption_tensors.device)], dim=0
        )
        return datas, caption_tensors


def chat(query, model, tokenizer, 
        max_length: int = 256, num_beams=1, top_p=0.7, temperature=0.95):
    prompt = f"[Round 0]\n\n问：{query}\n\n答："
    inputs = tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    # strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id])
    strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id], num_beams=num_beams, consider_end=True)
    
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy
    )[0]

    output_list = list(output)

    response = tokenizer.decode(output_list[0])
    print(response)


def infer(inputs, model, tokenizer, num_beams=1, top_p=0.7, temperature=0.95, batch_size=16):
    strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id],
                                  num_beams=num_beams, consider_end=True)

    output = filling_sequence(
        model, inputs,
        batch_size=batch_size,
        strategy=strategy
    )
    output_list = list(output)
    response = tokenizer.decode(output_list)
    return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = parser.parse_args()

    # load model
    model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/sat-finetune-sample/checkpoints/en_to_zh',
    args=argparse.Namespace(
        mode='inference',
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
        device=f"cuda:{args.local_rank}"
    ))
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    meta_path = "/nxchinamobile2/shared/img_datasets/cleaned_imgs_data/coyo_700m_merged/part-00000/000000.meta.jsonl"
    dataset = CoyoDataset(meta_path, tokenizer, max_length=args.max_length, device=model.parameters().__next__().device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    for batch in tqdm(dataloader):
        datas = batch[0]
        inputs = batch[1]
        outputs = infer(inputs, model, tokenizer, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature)
        print(outputs)
        break
