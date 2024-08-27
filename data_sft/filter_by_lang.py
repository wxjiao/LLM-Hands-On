##############################
# Function: filter lines by language detection
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
from langid.langid import LanguageIdentifier, model


# language correctness
def is_lang_correct(sent, target_lang, langid_norm):
    #detected = langid.classify(sent)[0]
    detected = langid_norm.classify(sent)
    #print(detected)
    if detected[0] == target_lang and detected[1] > 0.99999:
        return True
    return False


# detect and remove
def remove_mislang(path_in, path_out, target_lang, langid_norm):
    count = 0
    with open(path_in, 'r', encoding='utf-8') as fi, open(path_out, 'w', encoding='utf-8') as fo:
        for line in tqdm(fi):
            j = json.loads(line)
            text = j[target_lang]
            if not is_lang_correct(text, target_lang, langid_norm):
                continue
            jsoned = json.dumps(j, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            count += 1
            if count % 1000 == 0:
                fo.flush()
    return count


if __name__ == "__main__":
    """
    python3 filter_by_lang.py -l ko -i sample.json -o sample.lang.json
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
    
    langid_norm = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    langid_norm.set_languages(lang_set)    
 
    # process
    count = remove_mislang(args.input_path, args.output_path, args.lang, langid_norm)
    print("Valid lines after language detection: ", count)
