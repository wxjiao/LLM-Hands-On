##############################
# Function: convert alpaca data format to chat data format
# Author: Wenxiang Jiao
# Last modified: 2024/04/25
##############################

import os
import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_prompt(path):
    list_data_dict = read_json(path)
    prompts = []
    for i in tqdm(range(len(list_data_dict))):
        example = list_data_dict[i]
        p = dict()
        msg = []
        instruct, response = example["instruction"], example["output"]
        input = example["input"] if example.get("input", "") != "" else None
        instruct = instruct + "\n" + input if input != None else instruct
        msg.append({"role": "user", "content": instruct})
        msg.append({"role": "assistant", "content": response})
        p["conversations"] = msg
        prompts.append(p)
    return prompts


# Save conv into json data
def write_json(prompts, out_file):
    with open(out_file, 'w', encoding='utf-8') as fo:
        for p in prompts:
            jsoned = json.dumps(p, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            fo.flush()


if __name__ == "__main__":
    """
    python3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--out-file','-o', type=str, required=True, help='output file')
    args = parser.parse_args()
    in_file = args.in_file
    out_file = args.out_file
    
    # Read alpaca data and convert
    prompts = create_prompt(in_file)
    write_json(prompts, out_file)
