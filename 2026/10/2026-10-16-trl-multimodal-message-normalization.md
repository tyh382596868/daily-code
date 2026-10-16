---
date: 2026-10-16
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/data_utils.py
permalink: https://github.com/huggingface/trl/blob/447be34c59a4ad7fb6212917382c62053bab147b/trl/data_utils.py#L32-L124
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, multimodal-messages]
---

# TRL 多模态消息：先归一化结构，再填图像 / TRL Multimodal Messages: Normalize Structure Before Filling Images

> **一句话 / In one line**: TRL 把字符串消息标准化为 block 列表，再严格按占位符数量注入图像。 / TRL normalizes string messages into block lists, then injects images only when placeholder accounting agrees.

## 为什么重要 / Why this matters

同一训练集经常混着纯文本、已结构化的消息和外部图像列表。先统一 schema，才能让 chat template 和 processor 不必猜输入形态。

A training set often mixes plain text, already-structured messages, and separate image lists. Normalizing the schema first means chat templates and processors never need to guess the input shape.

## 代码 / The code

`huggingface/trl` — [`data_utils.py`](https://github.com/huggingface/trl/blob/447be34c59a4ad7fb6212917382c62053bab147b/trl/data_utils.py#L32-L124)

```python
images = images or []
new_messages = []
images_included = False
for message in messages:
    if message["role"] == "user" and isinstance(message["content"], str) and not images_included:
        image_entries = [{"type": "image"} for _ in range(len(images))]
        message = {**message, "content": [*image_entries, {"type": "text", "text": message["content"]}]}
        images_included = True
    new_messages.append(message)
```

## 逐行讲解 / What's happening

1. 中文: `images or []` 让后续计数逻辑只面对一种容器。
   English: `images or []` gives downstream accounting one container shape.
2. 中文: 首个用户 turn 接收自动插入的图像占位符，避免把同一批图片重复塞给每轮对话。
   English: The first user turn receives automatic placeholders, avoiding repeated insertion into every turn.
3. 中文: 后续代码统计未填充 placeholder；数量不等就报错，而不是静默错配图文。
   English: Later code counts unfilled placeholders and raises on mismatch instead of silently pairing the wrong image with text.

## 类比 / The analogy

像餐厅先把订单都转成同一张点单纸，再按编号上菜；少一格或多一道菜都必须停下来核对。

It is a restaurant standardizing every order ticket before plating by slot number; a missing or extra dish must stop the line.

## 自己跑一遍 / Try it yourself

```python
def normalize(text, images):
    blocks = [{"type": "image", "image": x} for x in images]
    return blocks + [{"type": "text", "text": text}]

print(normalize("describe this", ["a.png", "b.png"]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[{'type': 'image', 'image': 'a.png'}, {'type': 'image', 'image': 'b.png'}, {'type': 'text', 'text': 'describe this'}]
```

中文: 模板只需要消费 blocks，不需要分支处理原始字符串。  
English: A template now consumes blocks and avoids branching on raw strings.

## 注意事项 / Caveats / when it breaks

- 中文: 已含 image payload 的 block 不应再次填充。 / English: Blocks that already contain an image payload must not be filled again.
- 中文: placeholder 顺序就是图像语义顺序。 / English: Placeholder order defines image semantics.

## 延伸阅读 / Further reading

- [TRL message preparation](https://github.com/huggingface/trl/blob/447be34c59a4ad7fb6212917382c62053bab147b/trl/data_utils.py#L32-L124)
