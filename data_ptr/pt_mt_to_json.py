import argparse
from tqdm import tqdm
#import jsonlines
import json
import copy
import random
import numpy as np


# Instruction language, default: 'en'
lang_instruction = {
    'de': {'de': ["Deutsch"], 'en': ["Englisch"], 'ja': ["Japanisch"], 'zh': ["Chinesisch"]},
    'en': {'de': ["German"], 'en': ["English"], 'ja': ["Japanese"], 'zh': ["Chinese"], 'ko': ["Korean"]},
    'ja': {'de': ["ドイツ語"], 'en': ["英語"], 'ja': ["日本語"], 'zh': ["中国語"]},
    'zh': {'de': ["德语", "德文"], 'en': ["英语", "英文"], 'ja': ["日语", "日文"], 'zh': ["中文", "汉语"], 'ko': ["韩语", "韩文"]},
}


SCHEMA = {
    "text": None,
    "date": "2022",    # 2022, 2022, 2021, 2021, 2021
    "language": None,
    "uri": None,
    "domain": "general",    # news, literary, government, mixed, oral
    "source": None,
    "corpus": "TranSmart",
}


# Lang script
def lang_script(script_lang, lang):
    selected = random.choice(lang_instruction[script_lang][lang])
    return selected
    

# Read task instruction, fill in languages
def read_instruct(path, langpair, lang_ins="en"):
    src, tgt = langpair.split('-')
    ins_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            ins_list.append(l.strip())
    return ins_list


def get_line_text(line_list, src, tgt, order, ins_list, lang_ins):
    source, target = lang_script(lang_ins, src), lang_script(lang_ins, tgt)
    # pair
    line_text = ""
    for line in line_list:
        src_line = f'{source}: ' + line[0]
        tgt_line = f'{target}: ' + line[1]
        if order == 1:
            src_line, tgt_line = tgt_line, src_line
        line_text += "{}\n{}\n\n".format(src_line, tgt_line)
    
    # add prompt
    if ins_list is None:
        return line_text
    
    prompt = random.choice(ins_list)
    prompt = prompt.replace("[SRC]", source).replace("[TGT]", target) if order == 0 else prompt.replace("[SRC]", target).replace("[TGT]", source)
    line_text = prompt + "\n\n" + line_text
    return line_text


def pair_to_json(src_file, tgt_file, out_file, langpair, max_win, max_num, ins_list, lang_ins, bidirection):
    #random.seed(0)
    count = 0
    count_doc = 0
    line_list = []
    window = random.randint(2, max_win)
    src, tgt = langpair.split('-')
    order = 0
    with open(src_file, 'r', encoding='utf-8') as fs,open(tgt_file, 'r', encoding='utf-8') as ft,open(out_file, 'w', encoding='utf-8') as fo:
        for sl, tl in tqdm(zip(fs, ft), ascii='#'):
            count += 1
            sline, tline = sl.strip(), tl.strip()
            if len(sline) < 1 or len(tline) < 1:
                continue
            line_list.append([sl.strip(), tl.strip()])
            if len(line_list) == window:
                count_doc += 1
                line_text = get_line_text(line_list, src, tgt, order, ins_list, lang_ins)
                lang = "{}-{}".format(src, tgt) if order == 0 else "{}-{}".format(tgt, src)
                p = copy.deepcopy(SCHEMA)
                p["language"] = lang
                p["text"] = line_text
                #print(p)
                jsoned = json.dumps(p, ensure_ascii=False)
                fo.write(jsoned)
                fo.write('\n')
                fo.flush()
                line_list = []
                if bidirection:
                    order = random.choice([0, 1])
                
            # early stop
            if count_doc > max_num:
                break
    return count_doc, count


if __name__ == "__main__":
    """
    python3 ../../pt_mt_to_json.py -sf ../filtered/new_train2.065.zh -tf ../filtered/new_train2.065.ko -if ../../pt_mt_instruct.zh.txt -l zh-ko -il zh -o test_n.json -n 100 -bi
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file','-sf', type=str, required=True, help='input src file')
    parser.add_argument('--tgt-file','-tf', type=str, required=True, help='input tgt file')
    parser.add_argument('--ins-file','-if', type=str, default=None, help='instruction file')
    parser.add_argument('--out-file','-o', type=str, required=True, help='output file')
    parser.add_argument('--langpair','-l', type=str, required=True, help='language pair, zh-en')
    parser.add_argument('--ins-lang','-il', type=str, default="zh", help='instruct language, zh')
    parser.add_argument('--max-win','-mw', type=int, default=10, help='max windows for pairs')
    parser.add_argument('--num-doc','-n', type=int, default=1000000000, help='max documents to extract')
    parser.add_argument('--bidirection','-bi', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=0, help='random seed')
    args = parser.parse_args()
    print(args)
    src_file = args.src_file
    tgt_file = args.tgt_file
    ins_file = args.ins_file
    out_file = args.out_file
    langpair = args.langpair
    ins_lang = args.ins_lang
    max_win = args.max_win
    num_doc = args.num_doc
    bidirection = args.bidirection
    
    # read insruct
    ins_list = read_instruct(ins_file, langpair) if ins_file is not None else None
    print("Number of instructions: {}".format(len(ins_list) if ins_list is not None else 0))
    
    # convert pairs
    random.seed(args.seed)
    count_doc, count = pair_to_json(src_file, tgt_file, out_file, langpair, max_win, num_doc, ins_list, ins_lang, bidirection)
    print("Num of docs/lines: {} / {}".format(count_doc, count))

