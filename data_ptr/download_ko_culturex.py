from huggingface_hub import login
from datasets import load_dataset
import json
import copy
from tqdm import tqdm

#login()

lang = "ko"

SCHEMA = {
    "text": None,
    "date": "2023/01",
    "language": lang,
    "uri": None,
    "domain": "general",
    "source": None,
    "corpus": "uonlp/CulturaX",
}


dataset = load_dataset("uonlp/CulturaX", lang, use_auth_token=True, streaming=True)
print(dataset["train"].take(1))


output_path = "./CultureX-{}.json".format(lang)

count=0
with open(output_path, 'w', encoding='utf-8') as fo:
    for line in tqdm(dataset["train"]):
        content = line["text"]
        #if "title" in line.keys():
        #    content = line["title"] + "\n\n" + content
        p = copy.deepcopy(SCHEMA)
        p["text"] = content
        p["date"] = line["timestamp"]
        p["uri"] = line["url"]
        p["source"] = line["source"]
        jsoned = json.dumps(p, ensure_ascii=False) 
        fo.write(jsoned)
        fo.write('\n')
        count += 1
        if count % 100 == 0:
            fo.flush()
        if count >= 20000000:
            break

print("Num of docs: {}".format(count))




