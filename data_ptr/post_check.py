##############################
# Function: split and save conv data into hf format
# Author: Wenxiang Jiao
# Last modified: 2023/07/06
##############################

import os
import hashlib
import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np
#from transformers import AutoTokenizer
import copy


# Read json lines
def post_check(in_file, out_file):
    count = 0
    with open(in_file, 'r', encoding='utf-8') as fi,open(out_file, 'w', encoding='utf-8') as fo:
        for line in tqdm(fi):
            data = json.loads(line)
            # check field type
            for k,v in data.items():
                if type(v) is str:
                    data[k] = v
                elif type(v) is list:
                    data[k] = v[0]
                else:
                    data[k] = None
            # add sample seperator: \n\n\n
            data["text"] = data["text"].strip() + "\n\n\n"
            jsoned = json.dumps(data, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            count += 1
            if count % 10000 == 0: 
                fo.flush()


if __name__ == "__main__":
    """
    python3 post_check.py -i input.json -o output.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--out-file','-o', type=str, required=True, help='output file')
    args = parser.parse_args()
    print(args)
    in_file = args.in_file
    out_file = args.out_file
    print("Checking fields...")
    post_check(in_file, out_file)


