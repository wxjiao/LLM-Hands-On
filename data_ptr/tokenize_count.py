##############################
# Function: count tokens
# Author: Wenxiang Jiao
# Last modified: 2024/06/26
##############################

import os
import hashlib
import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np
from transformers import AutoTokenizer


# Read json lines
def read_text(path):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(line.strip())
    return data_list


def read_json2text(path):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data["text"])
    return data_list


if __name__ == "__main__":
    """
    python3 xxx.py -i text.json -m Qwen1.5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--model-name-or-path','-m', type=str, required=True)
    parser.add_argument('--max-length','-maxlen', type=int, default=2048)
    parser.add_argument('--num-line','-n', type=int, default=1000)
    parser.add_argument('--use-fast','-fast', action='store_true')
    args = parser.parse_args()
    print(args)
    in_file = args.in_file
    model_name_or_path = args.model_name_or_path
    num_line = args.num_line
    #max_length = args.max_length
    use_fast = True  #args.use_fast
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
    )
    # Tokenize
    count_token = 0
    count_line = 0
    with open(in_file, 'r', encoding='utf-8') as fi:
        for line in tqdm(fi):
            data = json.loads(line)
            token = tokenizer.tokenize(data["text"])
            count_token += len(token)
            count_line += 1
            if count_line >= num_line:
                break
    count_token_avg = count_token / (count_line + 1e-6)
    print("Total lines: {}\nTokens per line: {}".format(count_line, count_token_avg))
