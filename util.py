import json
import os


if __name__ == "__main__":
    input_path = "/gpfs/gpfs1/zphz/img_datasets/en2cn_wmt17_dataset/train/0000.json"
    output_path = "/gpfs/gpfs1/zphz/jjh/projects/sat-finetune-sample/train.jsonl"
    with open(input_path, "r", encoding='utf-8') as f1, open(output_path, "w", encoding='utf-8') as f2:
        datas = json.load(f1)
        for data in datas:
            prompt = data['translation.en']
            response = data['translation.cn']
            f2.write(json.dumps({'prompt': prompt, 'response': response}, ensure_ascii=False) + '\n')
