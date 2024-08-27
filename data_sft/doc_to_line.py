##############################
# Function: convert doc-level data to lines
# Author: Wenxiang Jiao
# Last modified: 2024/06/12
##############################

import os
import time
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import langid


# language correctness
def is_lang_correct(sent, target_lang):
    detected = langid.classify(sent)[0]
    if detected == target_lang:
        return True
    return False


# split doc (or not)
def split_doc(doc, target_lang, to_split, delimiter):
    sents = []
    if to_split:
        sents = doc.strip().split("\n")
    else:
        sents = sent.append(doc.strip())
    sents_left = []
    for s in sents:
        s = s.strip()
        s_tok = s.split(delimiter) if delimiter != "" else s
        if len(s_tok) < 10:
            continue
        if not is_lang_correct(s, target_lang):
            continue
        sents_left.append(s)
    return sents_left


def process(path_in, path_out, target_lang, to_split, delimiter):
    num_doc = 0
    num_line = 0
    with open(path_in, 'r', encoding='utf-8') as fi, open(path_out, 'w', encoding='utf-8') as fo:
        for line in tqdm(fi):
            j = json.loads(line)
            doc = j["text"]
            num_doc += 1
            ###
            sents_left = split_doc(doc, target_lang, to_split, delimiter)
            for s in sents_left:
                p = dict()
                p[target_lang] = s
                jsoned = json.dumps(p, ensure_ascii=False)
                fo.write(jsoned)
                fo.write('\n')
                #fo.flush()
                num_line += 1
                if num_line % 1000 == 0:
                    fo.flush()
    return num_doc, num_line


if __name__ == "__main__":
    """
    python3 doc_to_line.py -l ko -i CultureX-korean-sample.json -o CultureX-korean-sample.lines.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', type=str, required=True, help='ko')
    parser.add_argument('--input-path','-i', type=str, required=True, help='input file')
    parser.add_argument('--output-path','-o', type=str, required=True, help='output json')
    args = parser.parse_args()
    print(args)
    
    # set lang set
    lang_set = ['ar', 'de', 'en', 'es', 'fr', 'id', 'it', 'ja', 'ko', 'ms', 'pt', 'ru', 'th', 'tr', 'vi', 'zh']
    lang_set.append(args.lang)
    lang_set = set(lang_set)
    langid.set_languages(lang_set)
    
    # process
    lang_nosplit = ['zh', 'ja', 'ko']
    delimiter = "" if args.lang in lang_nosplit else " "
    num_doc, num_line = process(args.input_path, args.output_path, args.lang, True, delimiter)
    print("Obtained lines / docs: {} / {}".format(num_line, num_doc))
