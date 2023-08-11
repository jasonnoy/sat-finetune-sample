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
import math


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
                data = dict(json.loads(line))
                self.original_datas.append(data)
                self.caption_prompts.append(f"[Round 0]\n\n问：{data['caption']}\n\n答：")

    def __len__(self):
        return len(self.original_datas)

    def __getitem__(self, idx):
        caption_prompt = self.caption_prompts[idx]
        caption_tensor = self.tokenizer(caption_prompt, return_tensors="pt").to(self.device)['input_ids'][0]
        caption_tensor = torch.cat([caption_tensor, torch.tensor([-1] * (self.max_length - len(caption_tensor)), device=caption_tensor.device)], dim=0)
        return caption_tensor

    def get_origincal_datas(self):
        return self.original_datas


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


def infer(seqs, model, tokenizer, num_beams=1, top_p=0.7, temperature=0.95):
    strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id],
                                  num_beams=num_beams, consider_end=True)
    outputs = []
    for seq in seqs:
        output = filling_sequence(
            model, seq,
            batch_size=1,
            strategy=strategy
        )[0]

        response = tokenizer.decode(output[0].cpu())
        response = response.split("\n\n答：")[1]
        outputs.append(response)

    return outputs


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def get_all_ids_under_dir(path):
    dirs = os.listdir(path)
    ids = []
    for d in dirs:
        dir_path = os.path.join(path, d)
        all_files = os.listdir(dir_path)
        all_files = [f.split(sep='.')[0] for f in all_files if f.endswith(".jsonl")]
        ids.extend(all_files)
    ids.sort()
    return ids


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
        device=f"cuda:{args.local_rank // 2}"
    ))
    model = model.eval()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    print("rank {}, device: {}".format(args.rank, model.parameters().__next__().device))

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

    input_path = "/nxchinamobile2/shared/img_datasets/cleaned_imgs_data/coyo_700m_merged"
    output_path = "/nxchinamobile2/shared/jjh/coyo_700m_merged_translate"
    all_ids = get_all_ids_under_dir(input_path)
    divided_ids = split_list_by_n(all_ids, args.world_size)
    select_ids = divided_ids[args.rank]

    for idx in select_ids:
        print("rank {} processing {}...".format(args.rank, idx))
        dir_id = idx[:-5]
        dir_name = "part-00000"[:-len(dir_id)]
        dir_name += dir_id
        input_dir_path = os.path.join(input_path, dir_name)
        output_dir_path = os.path.join(output_path, dir_name)
        os.makedirs(output_dir_path, exist_ok=True)
        meta_filename = f"{idx}.meta.jsonl"
        input_meta_path = os.path.join(input_dir_path, meta_filename)
        output_meta_path = os.path.join(output_dir_path, meta_filename)

        dataset = CoyoDataset(input_meta_path, tokenizer, max_length=args.max_length, device=model.parameters().__next__().device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        total_outputs = []
        for batch in tqdm(dataloader):
            outputs = infer(batch, model, tokenizer, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature)
            total_outputs.extend(outputs)

        with open(output_meta_path, "w", encoding='utf-8') as f:
            for data, output in zip(dataset.get_origincal_datas(), total_outputs):
                data['caption_zh'] = output
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.close()
