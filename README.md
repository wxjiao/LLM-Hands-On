# LLM-Hands-On: Pipeline to Develop Multiturn Chatbots Based on LLMs

- Continual Pre-Training
- Supervised Fine-Tuning
- Human Alignment (DPO)
- Web Chatbot

## Environment

We develop the chatbots based on open-sourced LLMs (e.g., Llama-2/3, Qwen1.5/2) with HuggingFace's transformers library.

Framework Versions:

- Python 3.8.12
- Pytorch 1.13.1+cu117
- Transformers >= 4.39.0.dev0
- Peft 0.9.0
- Flash-attn 2.5.6
- Other requirements

```
pip install -r requirements.txt
```


## Chatbot Format

**Chat Data Structure**

We use multi-turn conversations as the input with two fields, namely, ``role`` and ``content``, where ``role`` takes "user" or "assistant". You can also add a "system" role for system prompts.

```ruby
chat = [
        {"role": "system", "content": "你是一个智能助手。"},
        {"role": "user", "content": "如果你能提供以下这些句子的汉语翻译，我将不胜感激。"},
        {"role": "assistant", "content": "当然！我很高兴帮助您进行汉语翻译。请提供您希望翻译的句子。"},
        {"role": "user", "content": "28-Year-Old Chef Found Dead at San Francisco Mall"},
        {"role": "assistant", "content": "翻译：\n28岁厨师被发现死于旧金山一家商场"}
    ]
```


**Chat to Text**

We adopt the tokenizer to tranform the chat data into text strings for training and inference.
The chat templates are written in Jinja languages, and are different among different LLMs.
We define a default chat template as below:  

```
{% for message in messages %}{{'###' + message['role'] + ':\n' + message['content'] + '</s>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '###assistant:\n' }}{% endif %}
```

The chat data can be transformed into the following text string using   ``tokenizer.apply_chat_template(chat, tokenize=False)``:

```
###system:
你是一个智能助手。</s>
###user:
如果你能提供以下这些句子的汉语翻译，我将不胜感激。</s>
###assistant:
当然！我很高兴帮助您进行汉语翻译。请提供您希望翻译的句子。</s>
###user:
28-Year-Old Chef Found Dead at San Francisco Mall</s>
###assistant:
翻译：
28岁厨师被发现死于旧金山一家商场</s>
```


