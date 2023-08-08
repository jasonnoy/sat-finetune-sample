import json
import random
import pandas as pd


# if __name__ == "__main__":
#     # 读取jsonl文件
#     prompts = []
#     texts = []
#     details = []
#     keys = []
#     with open("/nxchinamobile2/shared/img_datasets/cleaned_instructions/image_caption/diy_caption_v1/meta.jsonl", "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             data = json.loads(line)
#             prompts.append(data["prompt"])
#             texts.append(data["txt"])
#             details.append(data["details"])
#             keys.append(data["__key__"])
#             if i == 50:
#                 break
#     f.close()
#     df = {"key": keys, "prompt": prompts, "txt": texts, "details": details}
#     df = pd.DataFrame(df)
#     df.to_csv("./sat_zh_50.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    df = pd.read_csv("sat_zh_50.csv")
    test_ids = df['key'].to_list()
    p = 0.6
    with open("train_zh_100k.jsonl", 'r', encoding='utf-8') as f1, open("mix_combine_zh.jsonl", 'w', encoding='utf-8') as f2:
        prompt = ""
        response = ""
        for line in f1:
            data = json.loads(line)
            if random.random() < p:
                prompt += " "+data["prompt"]
                response += " "+data["response"]
                continue
            f2.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            prompt = ""
            response = ""
    f2.close()
    f1.close()

    with open("/nxchinamobile2/shared/wy/data/input/rankv3_short_new.jsonl", 'r', encoding='utf-8') as f1, open("mix_combine_zh.jsonl", 'a', encoding='utf-8') as f2:
        for line in f1:
            data = json.loads(line)
            if data['__key__'] in test_ids or data['status'] != 'success':
                continue
            f2.write(json.dumps({'prompt': data['prompt'], 'response': data['prompt_en']}))
            f2.write(json.dumps({'prompt': data['txt'], 'response': data['txt_en']}))
    f1.close()
    f2.close()
