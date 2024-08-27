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
random.seed(1)
        

# Create data
def create_prompt(path_si, path_ti, path_out, src, tgt):
    with open(path_si, 'r', encoding='utf-8') as fs,open(path_ti, 'r', encoding='utf-8') as ft,open(path_out, 'w', encoding='utf-8') as fo:
        for ls,lt in tqdm(zip(fs, ft)):
            p = dict()
            p[src] = ls.strip()
            p[tgt] = lt.strip()
            jsoned = json.dumps(p, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            fo.flush()


if __name__ == "__main__":
    """
    python3 text2json.py -l zh-en -si newstest17-20.en-zh.zh -ti newstest17-20.en-zh.en -o newstest17-20.zh-en.jsonl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--langpair', '-l', type=str, required=True, help='zh-en')
    parser.add_argument('--src-path','-si', type=str, required=True, help='input file')
    parser.add_argument('--tgt-path','-ti', type=str, required=True, help='input file')
    parser.add_argument('--output-path','-o', type=str, required=True, help='output jsonl')
    args = parser.parse_args()
    print(args)
    src, tgt = args.langpair.split('-')
    print("Convert {} text to json".format(args.langpair))

    create_prompt(args.src_path, args.tgt_path, args.output_path, src, tgt)
