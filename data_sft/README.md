# SFT Data

## Machine Translation Data

**Parallel Data From WMT**

Transform parallel data in TEXT into json:
```
python3 pair_to_json.py -l zh-en -si newstest17-20.en-zh.zh -ti newstest17-20.en-zh.en -o newstest17-20.zh-en.json
```

**Parallel Data From Monolingual**

Preprocess monolingual data from ``data_ptr`` and translate:
```
#Split doc to lines
python3 doc_to_line.py -l ko -i culturex-ko.2k.json -o culturex-ko.2k.lines.json

#Deduplicate
python3 deduplicate.py -l ko-ko -i culturex-ko.2k.lines.json -o culturex-ko.2k.lines.dedup.json

#Translate by Google
python3 google_translator.py -l ko-zh -i culturex-ko.2k.lines.dedup.json -o culturex-ko.2k.lines.dedup.out.json -sp 0:5000 -ak xxx

#Or by GPT
python3 gpt_translator.py -l ko-zh -gm gpt-35-turbo-1106 -i culturex-ko.2k.lines.dedup.json -o culturex-ko.2k.lines.dedup.out.json
```

The obtained parallel data are saved line by line as below:
```
{"zh": "28岁厨师被发现死于旧金山一家商场", "en": "28-Year-Old Chef Found Dead at San Francisco Mall"}
```

**Chat Data From Parallel Data**

Transform the parallel data into chat format data:
```
python3 chat_data_from_pairbi.py -l zh-en -i newstest17-20.zh-en.json -o chat_mt.zh-en.bi.json -p instruct_mt.json -bi
```

```ruby
{"conversations": [
    {"role": "user", "content": "请将以下中文语言的文本翻译成英文：28岁厨师被发现死于旧金山一家商场"},
    {"role": "assistant", "content": "28-Year-Old Chef Found Dead at San Francisco Mall"}
]}
```

## General Task Data

**Chat Data From Alpaca Data**

Transform the Alpaca data into chat format data:
```
python3 chat_data_from_alpaca.py -i alpaca_gpt4_data_zh.json -o chat_alpaca.json
```

```ruby
{"conversations": [
    {"role": "user", "content": "保持健康的三个提示。"},
    {"role": "assistant", "content": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"}
]}
```


