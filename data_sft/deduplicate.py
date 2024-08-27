##############################
# Function: deduplicate parallel or monolingual data saved in json
# Author: Wenxiang Jiao
# Last modified: 2024/06/25
##############################

import os
import time
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import hashlib
random.seed(1)


def get_str_md5(string):
    m = hashlib.md5()
    m.update(string.encode("utf-8"))
    return m.hexdigest()


def get_part_md5(string, ratio, part):
    """
    part: full, head, tail, middle, double
    """
    if part == "full":
        return get_str_md5(string)
    str_part = ""
    num_char = int(len(string) * ratio)
    num_char_left = len(string) - num_char
    if part == "head":
        str_part = string[:num_char]
    elif part == "tail":
        str_part = string[-num_char:]
    elif part == "middle":
        str_part = string[num_char_left//2: num_char_left//2 + num_char]
    else:
        str_part = string[:num_char//2] + string[-(num_char - num_char//2):]
    return get_str_md5(str_part)


def efficient_dedup(samples, src, tgt, ratio, part="full"):
    dedup_md5_samp = dict()
    dup = []
    for idx in tqdm(range(len(samples))):
        samp = samples[idx]
        #samp_txt = samp[src] + samp[tgt]
        samp_txt = samp[src]    # only source
        samp_md5 = get_part_md5(samp_txt, ratio, part)
        #samp_md5 = get_str_md5(samp_txt)
        if samp_md5 in dedup_md5_samp.keys():
            dup.append(dedup_md5_samp[samp_md5])
        dedup_md5_samp[samp_md5] = samp
    dedup_samp = list(dedup_md5_samp.values())
    return dedup_samp, dup
    

def round_dedup(samples, src, tgt, ratio):
    dedup_full, dup_full = efficient_dedup(samples, src, tgt, ratio, "full")
    dedup_head, dup_head = efficient_dedup(dedup_full, src, tgt, ratio, "head")
    dedup_tail, dup_tail = efficient_dedup(dedup_head, src, tgt, ratio, "tail")
    dedup_midd, dup_midd = efficient_dedup(dedup_tail, src, tgt, ratio, "middle")
    dedup_doub, dup_doub = efficient_dedup(dedup_midd, src, tgt, ratio, "double")
    dup_all = dup_full + dup_head + dup_tail + dup_midd + dup_doub
    return dedup_doub, dup_all
    

def read_pairs(path):
    samples = list()
    with open(path, 'r', encoding='utf-8') as fi:
        for line in tqdm(fi):
            j = json.loads(line)
            samples.append(j)
    return samples
            

def write_pairs(samples, path):
    with open(path, 'w', encoding='utf-8') as fo:
        for samp in tqdm(samples):
            jsoned = json.dumps(samp, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            fo.flush()
            

if __name__ == "__main__":
    """
    python3 deduplicate.py -l zh-ko -i DNF_pair.240603.v2.json -o DNF_pair.240603.v2.dedup.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--langpair', '-l', type=str, required=True, help='zh-ko')
    parser.add_argument('--input-path','-i', type=str, required=True, help='input file')
    parser.add_argument('--output-path','-o', type=str, required=True, help='output jsonl')
    parser.add_argument('--part-ratio','-r', type=float, default=0.6, help='part ratio')
    args = parser.parse_args()
    print(args)
    src, tgt = args.langpair.split('-')
    print("Dedup data for: {}".format(args.langpair))
    part_ratio = args.part_ratio
    
    samples = read_pairs(args.input_path)
    dedup_all, dup_all = round_dedup(samples, src, tgt, part_ratio)
    print("Deduped: {} / {}".format(len(dedup_all), len(samples)))
    
    # write valid
    write_pairs(dedup_all, args.output_path)
    
    # write dup
    #write_pairs(dup_all, args.output_path.replace(".json", ".dup.json"))
