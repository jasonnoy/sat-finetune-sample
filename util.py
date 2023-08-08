import json
import pandas as pd


if __name__ == "__main__":
    # 读取jsonl文件
    prompts = []
    texts = []
    details = []
    with open("./sat_zh_2_cn_50.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            texts.append(data["txt"])
            details.append(data["details"])
    f.close()
    df = {"prompt": prompts, "txt": texts, "details": details}
    df = pd.DataFrame(df)
    df.to_csv("./sat_zh_2_cn_50.csv", index=False, encoding="utf-8")
