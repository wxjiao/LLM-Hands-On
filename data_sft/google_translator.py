#coding:utf-8
# Google Translate
# 2024-06-13

import os
import argparse
import time
import random
import json
import requests
from tqdm import tqdm


# Google Translate
def google_translator(src_text, src, tgt, api_key):
    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    payload = {
        "q": src_text,
        "target": tgt,
        "source": src,
        "format": "text"
    }
    for idx in range(5):
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            translated_text = response.json()["data"]["translations"][0]["translatedText"]
            return translated_text
    return None


def generate(input_path, output_path, src, tgt, api_key, st_idx, ed_idx):
    count = 0
    index = -1
    with open(input_path, 'r', encoding='utf-8') as fi,open(output_path, 'w', encoding='utf-8') as fo:
        for l in tqdm(fi):
            index += 1
            if index < st_idx:
                continue
            if index >= ed_idx:
                break
            line = json.loads(l)
            src_text = line[src]
            ##
            msg_out = google_translator(src_text, src, tgt, api_key)
            if msg_out is None:
                continue
            line[tgt] = msg_out
            line["model_name"] = "Google Translate"
            jsoned = json.dumps(line, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            #fo.flush()
            count += 1
            if count % 10 == 0:
                fo.flush()
            #if count >= 100:
            #    break


if __name__ == "__main__":
    """
    python3 google_translator.py -l zh-ko -i culturex-zh.2k.lines.dedup.json -o culturex-zh.2k.lines.dedup.out.json -sp 0:5000 -ak xxx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--langpair','-l', type=str, required=True, help='zh-ko')
    parser.add_argument('--in-file','-i', type=str, required=True, help='input json')
    parser.add_argument('--out-file','-o', type=str, required=True, help='output json line file')
    parser.add_argument('--api-key','-ak', type=str, required=True, help='api key')
    parser.add_argument('--line-span','-sp', type=str, required=True, help='0:1000 include left ignore right')
    args = parser.parse_args()
    print(args)
    in_file = args.in_file
    out_file = args.out_file
    src, tgt = args.langpair.split("-")
    print("Google Translate for: {}".format(args.langpair))
    
    st_idx, ed_idx = args.line_span.split(":")
    st_idx, ed_idx = int(st_idx), int(ed_idx)
    print("Processing lines: {}".format(args.line_span))
    
    out_file = out_file.replace(".json", ".{}-{}.json".format(st_idx, ed_idx))
    
    generate(in_file, out_file, src, tgt, args.api_key, st_idx, ed_idx)
    
    
