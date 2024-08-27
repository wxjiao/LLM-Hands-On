#coding:utf-8
#
# 2024-05-29
# openai==1.30.4

import os
import argparse
import time
import random
import json
import requests
from tqdm import tqdm

import openai
import tiktoken
import langid
import logging
from datetime import datetime
from openai import OpenAI


'''
GPT models:
    gpt-35-turbo-16k-0613，gpt-4-32k-0613，gpt-35-turbo-0301
    gpt-35-turbo-1106,gpt-4-1106-preview
    gpt-4-vision-preview
    gpt-4-0125-preview
'''


LANGS={
    "en": "英文",
    "zh": "中文",
    "ru": "俄语",
    "ja": "日语",
    "fr": "法语",
    "pt": "葡萄牙语",
    "es": "西班牙语",
    "tr": "土耳其语",
    "ar": "阿拉伯语",
    "ko": "韩语",
    "th": "泰语",
    "it": "意大利语",
    "de": "德语",
    "vi": "越南语",
    "ms": "马来语",
    "id": "印尼语"
}


client = OpenAI(base_url="", api_key="")
print("Client connected...")


# GPT generator
def gpt_generator(prompt, gpt_model, temperature):
    msg_prefix=[
        {"role": "user", "content": prompt}
        ]
    response = client.chat.completions.create(
            model=gpt_model,
            messages=msg_prefix,
            temperature=temperature,
        )
    gen = response.choices[0].message.content
    msg_out = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": gen}
        ]
    return msg_out


def generate(input_path, output_path, prompt, src, tgt, gpt_model, temperature):
    with open(input_path, 'r', encoding='utf-8') as fi,open(output_path, 'w', encoding='utf-8') as fo:
        for l in tqdm(fi):
            line = json.loads(l)
            question = prompt + line[src]
            # Handle errors with atmost 5 try
            try_success = False
            for _ in range(5):
                try:
                    msg_out = gpt_generator(question, gpt_model, temperature)
                    try_success = True
                    break
                except:
                    pass
            # Failed to access api
            if not try_success:
                continue
            line[tgt] = msg_out[-1]["content"]
            line["model_name"] = gpt_model
            jsoned = json.dumps(line, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            fo.flush()


if __name__ == "__main__":
    """
    python3 gpt_generate_mt.py -l ko-zh -gm gpt-35-turbo-1106 -i CultureX-korean-sample.lines.dedup.json -o CultureX-korean-sample.lines.dedup.ko-zh.json
    gpt-35-turbo-1106
    gpt-4-1106-preview
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt-model','-gm', type=str, default="gpt-4-1106-preview", help='gpt model name')
    parser.add_argument('--temperature','-tau', type=float, default=0., help='temerature')
    parser.add_argument('--langpair','-l', type=str, required=True, help='zh-ko')
    parser.add_argument('--in-file','-i', type=str, required=True, help='input json')
    parser.add_argument('--out-file','-o', type=str, required=True, help='output json line file')
    args = parser.parse_args()
    print(args)
    in_file = args.in_file
    out_file = args.out_file
    gpt_model = args.gpt_model
    temperature = args.temperature
    
    src, tgt = args.langpair.split("-")
    prompt = "将以下文本翻译为{}：\n\n".format(LANGS[tgt])
    print(prompt)
    
    generate(in_file, out_file, prompt, src, tgt, gpt_model, temperature)
    
    
