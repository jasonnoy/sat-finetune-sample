import json


with open("train_100k.jsonl", 'r', encoding='utf-8') as f, open("train_cn_100k.jsonl", 'w', encoding='utf-8') as f2:
    for line in f:
        data = json.loads(line)
        zh = data['response']
        en = data['prompt']
        f2.write(json.dumps({'prompt': zh, 'response': en}, ensure_ascii=False)+'\n')
f.close()
f2.close()

