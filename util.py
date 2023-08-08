import json
import random
import pandas as pd


if __name__ == "__main__":
    # 读取jsonl文件
    prompts = []
    texts = []
    details = []
    keys = []
    df = pd.read_csv("sat_zh_50.csv", dtype={"key": str})
    test_ids = df['key'].tolist()
    with open("/nxchinamobile2/shared/wy/data/input/rankv3_short_new.jsonl", "r", encoding="utf-8") as f:
        count = 0
        for i, line in enumerate(f):
            data = json.loads(line)
            if data["__key__"] not in test_ids:
                continue
            prompts.append(data["prompt_en"])
            texts.append(data["txt_en"])
            # details.append(data["details_en"])
            keys.append(data["__key__"])
            count += 1
            if count == 50:
                break
    f.close()
    df = {"key": keys, "prompt": prompts, "txt": texts}
    df = pd.DataFrame(df, dtype={"key": str})
    df = df.sort_values(by='key')
    df.to_csv("./chat_sat_zh_50.csv", index=False, encoding="utf-8")

# if __name__ == "__main__":
#     df = pd.read_csv("sat_zh_50.csv")
#     test_ids = df['key'].to_list()
#     p = 0.75
#     with open("train_zh_100k.jsonl", 'r', encoding='utf-8') as f1, open("mix_combine_zh.jsonl", 'w', encoding='utf-8') as f2:
#         prompt = ""
#         response = ""
#         for line in f1:
#             data = json.loads(line)
#             prompt += " " + data["prompt"]
#             response += " " + data["response"]
#             if random.random() < p:  # concat the next sentence
#                 continue
#             f2.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
#             prompt = ""
#             response = ""
#     f2.close()
#     f1.close()
#
#     with open("/nxchinamobile2/shared/wy/data/input/rankv3_short_new.jsonl", 'r', encoding='utf-8') as f1, open("mix_combine_zh.jsonl", 'a', encoding='utf-8') as f2:
#         for line in f1:
#             data = json.loads(line)
#             if data['__key__'] in test_ids or data['status'] != 'success':
#                 continue
#             f2.write(json.dumps({'prompt': data['prompt'], 'response': data['prompt_en']}, ensure_ascii=False) + "\n")
#             f2.write(json.dumps({'prompt': data['txt'], 'response': data['txt_en']}, ensure_ascii=False) + "\n")
#     f1.close()
#     f2.close()
#
#     with open("mix_combine_zh.jsonl", 'r', encoding='utf-8') as f1, open("shuffle_combine_zh.jsonl", 'a', encoding='utf-8') as f2:
#         datas = [json.loads(line) for line in f1]
#         random.shuffle(datas)
#         for data in datas:
#             f2.write(json.dumps(data, ensure_ascii=False) + "\n")
#     f1.close()
#     f2.close()
