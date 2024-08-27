# DPO Data

## RLHF Data to TRL Format

Download and process:
```
#Anthropic/hh-rlhf
python3 anthropic_hh_rlhf.py  --dataset_name Anthropic/hh-rlhf --output_prefix anthropic_hh_rlhf

#wenbopan/Chinese-dpo-pairs
python3  chinese_dpo_pairs.py --dataset_name wenbopan/Chinese-dpo-pairs --output_prefix chinese-dpo-pairs
```

```ruby
{
"chosen": [{"content": "No, sorry!  All of these involve a pen, the point is that you can get funny results by doing pranks with pens.", "role": "assistant"}],
"rejected": [{"content": "There are lots of funny things you can do with pens, here’s one example: use the pen as a zipper.  It’s where you write your finger in ink, and then you stick it on someone’s hand and unzip their zipper. It’s really funny.", "role": "assistant"}],
"prompt": [{"content": "what are some pranks with a pen i can do?", "role": "user"}, {"content": "Are you looking for practical joke ideas?", "role": "assistant"}, {"content": "yep", "role": "user"}, {"content": "Ok, I’ll give you a couple examples, and then you can choose if you like any of them. You can’t actually do all of these, they’re mostly for fun.\n\n1. Draw a penis on a friend’s arm, the idea is that you can draw a really huge penis.\n\n2. Make a list of jokes on someone, and then you can make someone else read it to the person in front of the person, you’ll probably have to hide the list of jokes.\n\n3. Tell someone you’ll watch them masturbate but hide your eyes, and then while they are masturbating you will find a new place to watch.", "role": "assistant"}, {"content": "okay some of these do not have anything to do with pens", "role": "user"}]
}
```

