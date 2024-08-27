from dataclasses import dataclass, field
import multiprocessing
import sys
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
#xxx
import json
from tqdm import tqdm

"""
# save to json
python3 anthropic_hh_rlhf.py  --dataset_name Anthropic/hh-rlhf --output_prefix anthropic_hh_rlhf
"""


api = HfApi()
@dataclass
class ScriptArguments:
    debug: Optional[bool] = field(default=False, metadata={"help": "Enable debug mode"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "The Hugging Face dataset repo"})
    output_prefix: Optional[str] = field(default="anthropic_hh_rlhf", metadata={"help": "Prefix for output files"})
    

# GPT-4 generated 😄 Define a function to process the input and extract the dialogue into structured format
def extract_dialogue(input_text):
    # Split the input by lines and initialize variables
    lines = input_text.strip().split('\n\n')
    dialogue_list = []
    
    # Iterate through each line and extract the dialogue
    for line in lines:
        # Check if the line starts with "Human" or "Assistant" and split accordingly
        if line.startswith("Human:"):
            role = "user"
            content = line.replace("Human: ", "").strip()
        elif line.startswith("Assistant:"):
            role = "assistant"
            content = line.replace("Assistant: ", "").strip()
        else:
            # If the line doesn't start with "Human" or "Assistant", it's part of the previous message's content
            # Append it to the last message's content
            dialogue_list[-1]['content'] += "\n\n" + line.strip()
            continue

        # Append the extracted dialogue piece to the list
        dialogue_list.append({"role": role, "content": content})
    
    return dialogue_list


if __name__ == "__main__":
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    ds = load_dataset(args.dataset_name)
    #ds = ds.select_columns(["prompt", "chosen", "rejected"])
    #ds.add_column(name="prompt", column=ds["chosen"])
    if args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        msg_chosen = extract_dialogue(row["chosen"])
        prompt = msg_chosen[:-1]
        chosen = msg_chosen[-1:]
        rejected = extract_dialogue(row["rejected"])[-1:]
        row["prompt"], row["chosen"], row["rejected"] = prompt, chosen, rejected
        return row

    ds = ds.map(
        process,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=True,
    )
    # xxx: 2024-04-29, save into json
    print(ds['train'][0])
    for k in ds.keys():
        fname = '.'.join([args.output_prefix, k, 'json'])
        with open(fname, 'w', encoding='utf-8') as fo:
            for line in tqdm(ds[k]):
                jsoned = json.dumps(line, ensure_ascii=False)
                fo.write(jsoned)
                fo.write('\n')
                fo.flush()
    # xxx: save chosen part for sft
    fname = '.'.join([args.output_prefix, 'train.chosen', 'json'])
    with open(fname, 'w', encoding='utf-8') as fo:
        for line in tqdm(ds['train']):
            jsoned = json.dumps({"conversations": line["prompt"]+line["chosen"] }, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')
            fo.flush()
