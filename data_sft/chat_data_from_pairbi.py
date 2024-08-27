##############################
# Function: convert mt pair data (json for long-text) to chat data format
# Author: Wenxiang Jiao
# Last modified: 2024/04/25
##############################

import os
import time
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
#random.seed(1)


# Instruction language, default: 'en'
lang_instruction = {
    'de': {'de': ["Deutsch"], 'en': ["Englisch"], 'ja': ["Japanisch"], 'zh': ["Chinesisch"]},
    'en': {'de': ["German"], 'en': ["English"], 'ja': ["Japanese"], 'zh': ["Chinese"], 'ko': ["Korean"]},
    'ja': {'de': ["ドイツ語"], 'en': ["英語"], 'ja': ["日本語"], 'zh': ["中国語"]},
    'zh': {'de': ["德语", "德文"], 'en': ["英语", "英文"], 'ja': ["日语", "日文"], 'zh': ["中文", "汉语"], 'ko': ["韩语", "韩文"]},
}


# Lang script
def lang_script(script_lang, lang):
    selected = random.choice(lang_instruction[script_lang][lang])
    return selected
    

# Read task instruction
def read_prompt(path):
    prompt_list = []
    with open(path, 'r', encoding='utf-8') as fi:
        for line in fi:
            j = json.loads(line)
            prompt_list.append(j)
    return prompt_list
        

# Create data
def create_prompt(path_in, path_out, prompt_list, src, tgt, bidirection=False):
    with open(path_in, 'r', encoding='utf-8') as fi,open(path_out, 'w', encoding='utf-8') as fo:
        for line in tqdm(fi):
            j = json.loads(line)
            _src, _tgt = src, tgt
            if bidirection and random.choice([False, True]):
                _src, _tgt = tgt, src
            src_text, tgt_text = j[_src], j[_tgt]
            src_tok, tgt_tok = lang_script('zh', _src), lang_script('zh', _tgt)
            
            index = random.randint(0, len(prompt_list) - 1)
            prompt = prompt_list[index]
            input = prompt["instruction"].replace("{{{text}}}", src_text).replace("{{{target_lang}}}", tgt_tok).replace("{{{origin_lang}}}", src_tok)
            output = prompt["output"].replace("{{{target}}}", tgt_text)
            
            p = dict()
            msg = []
            msg.append({"role": "user", "content": input})
            msg.append({"role": "assistant", "content": output})
            p["conversations"] = msg
            jsoned = json.dumps(p, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            fo.flush()


if __name__ == "__main__":
    """
    python3 chat_data_from_pairbi.py -l zh-en -i newstest17-20.zh-en.jsonl -o chat_mt.zh-en.bi.jsonl -p instruct_mt.jsonl -bi
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--langpair', '-l', type=str, required=True, help='zh-en')
    parser.add_argument('--input-path','-i', type=str, required=True, help='input file')
    parser.add_argument('--output-path','-o', type=str, required=True, help='output jsonl')
    parser.add_argument('--prompt-path', '-p', type=str, required=True, help='prompt jsonl')
    parser.add_argument('--seed', '-s', type=int, default=1, help='random seed')
    parser.add_argument('--bidirection','-bi', action='store_true')
    args = parser.parse_args()
    print(args)
    src, tgt = args.langpair.split('-')
    print("Making data for: {}".format(args.langpair))
    
    prompt_list = read_prompt(args.prompt_path)
    print("Number of prompts: {}".format(len(prompt_list)))

    random.seed(args.seed)
    create_prompt(args.input_path, args.output_path, prompt_list, src, tgt, args.bidirection)
