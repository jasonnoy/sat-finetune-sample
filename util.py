import json
import pandas as pd


if __name__ == "__main__":
    # 读取jsonl文件
    prompts = []
    texts = []
    details = []
    keys = []
    with open("/nxchinamobile2/shared/img_datasets/cleaned_instructions/image_caption/diy_caption_v1/meta.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            prompts.append(data["prompt"])
            texts.append(data["txt"])
            details.append(data["details"])
            keys.append(data["__key__"])
            if i == 50:
                break
    f.close()
    df = {"key": keys, "prompt": prompts, "txt": texts, "details": details}
    df = pd.DataFrame(df)
    df.to_csv("./sat_zh_50.csv", index=False, encoding="utf-8")
