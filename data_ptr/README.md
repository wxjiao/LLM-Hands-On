# Pretrain Data

**Monolingual Data Collection**

Obtain data resources from HuggingFace:
```
#CultureX
python3 download_ko_culturex.py

#Post-check
python3 post_check.py -i CultureX-ko.json -o CultureX-ko.pc.json
```

**Integrating Parallel Data**

Arrange parallel data into few-shot format:
```
python3 pt_mt_to_json.py -sf train.zh -tf train.ko -if pt_mt_instruct.zh.txt -l zh-ko -il zh -o train.zh-ko.json -n 100 -bi
```

```ruby
{"text": "将以下句子翻译成韩语。\n\n汉语: 在国际水平上，主要活动是：\n韩语: 국제 수준에서 주요 활동은 다음과 같습니다.\n\n汉语: 没有这些权利带来的安全和自由，保持高水平的精神健康状况是很困难的。\n韩语: 이런 권리가 가져온 안전과 자유가 없다면 높은 수준의 정신건강상태를 유지하는 것은 매우 어렵다.\n\n汉语: 深圳晚报讯（记者蔡志军）今天是香港回归15周年庆典日。\n韩语: 선전석간소식(기자 채지군): 오늘은 홍콩 반환 15주년 축제의 날입니다.\n\n汉语: 在高法判决前，台北地方法院判处在调查中泄露机密情报的黄姓前校长有期徒刑14个月。\n韩语: 고법 판결에 앞서 타이베이 지방법원은 수사중 기밀 정보를 누설한 황 전 총장에게 징역 14개월을 선고했다.\n\n", "date": "2022", "language": "zh-ko", "uri": null, "domain": "general", "source": null, "corpus": "TranSmart"}
```

Estimate the number of tokens in pretraining data:
```
python3 tokenize_count.py -i CultureX-ko.pc.json -m Qwen1.5
```
