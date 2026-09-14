---
date: 2026-09-14
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: train.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/train.py#L114-L175
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, nanoVLM, streaming-dataset, ddp, validation-split]
---

# nanoVLM 流式数据加载：没有长度也能切验证集 / nanoVLM Streaming Data: Split Validation Without Random Access

> **一句话 / In one line**: nanoVLM 让同一个 loader 同时支持本地随机访问数据和 Hub iterable stream，并用 `take`/`skip` 做流式验证切分。 / nanoVLM lets one loader support local random-access data and Hub iterable streams, using `take`/`skip` for streaming validation.

## 为什么重要 / Why this matters

中文：视觉语言训练数据很容易大到不适合完整下载。一个只会 `len()`、`shuffle()`、`select()` 的 loader，在 Hub streaming 模式下会立刻失效。nanoVLM 的 `get_dataloaders` 先把多个 dataset config 合并，再根据 `stream_dataset` 选择两套不同的切分语义：普通数据用索引，流式数据用一次性迭代器的前缀和剩余部分。

English: Vision-language datasets quickly become too large to download in full. A loader built only around `len()`, `shuffle()`, and `select()` breaks as soon as Hub streaming is enabled. nanoVLM first combines dataset configs, then chooses between index-based splitting for local data and prefix/remainder splitting for one-pass streams.

## 代码 / The code

`huggingface/nanoVLM` — [`train.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/train.py#L114-L175)

```python
def get_dataloaders(train_cfg, vlm_cfg):
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    dataset_names_to_load = train_cfg.train_dataset_name
    if "shards" in train_cfg.train_dataset_name:
        print("Loading shards")
        total_shards = 56
        dataset_names_to_load = [train_cfg.train_dataset_path + f"/shard_{i}" for i in range(total_shards)]

    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    # Load and combine all training datasets
    combined_train_data = []

    for dataset_name in dataset_names_to_load:
        print(f"Loading dataset: {dataset_name}")
        if "shard_" in dataset_name:
            try:
                train_ds = load_from_disk(dataset_name)
                combined_train_data.append(train_ds)
                continue
            except Exception as e:
                print(f"Warning: Failed to load dataset shard '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
                continue
        try:
            train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, streaming=train_cfg.stream_dataset, on_bad_files='warn')['train']
            if train_cfg.stream_dataset:
                next(iter(train_ds)) # Check if the dataset is loaded correctly
            else:
                train_ds[0] # Check if the dataset is loaded correctly
            combined_train_data.append(train_ds)
        except Exception as e:
            if is_master():
                print(f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
            continue

    if not combined_train_data:
        raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")
    
    train_ds = concatenate_datasets(combined_train_data)

    if not train_cfg.stream_dataset:
        train_ds = train_ds.shuffle(seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  


    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # train_ds = train_ds.shuffle(buffer_size=10000, seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  

    val_size = int(train_cfg.val_size/get_world_size())
    print(f"Val size per GPU: {val_size}")

    if train_cfg.stream_dataset:
        val_ds = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)
    else:
        val_ds = train_ds.select(range(val_size))
        train_ds = train_ds.select(range(val_size, len(train_ds)))
```

## 逐行讲解 / What's happening

1. **第 114-118 行 / Lines 114-118 (processor and tokenizer setup)**:
   - 中文：数据集构造前先初始化图像 processor 和 tokenizer，后面无论数据来自磁盘还是 Hub，都走同一套样本包装逻辑。
   - English: The image processor and tokenizer are created before dataset construction, so disk-backed and Hub-backed samples share one downstream path.
2. **第 120-128 行 / Lines 120-128 (dataset names)**:
   - 中文：`shards` 和 `all` 是两个配置级快捷方式：前者展开成固定 shard 路径，后者向 Hub 查询所有 config。
   - English: `shards` and `all` are configuration shortcuts: one expands to local shard paths, while the other asks the Hub for every config.
3. **第 132-151 行 / Lines 132-151 (failure-tolerant loading)**:
   - 中文：每个 config 独立尝试。流式数据用 `next(iter(train_ds))` 做最小可用性检查，避免把它当成可索引数组。
   - English: Each config is attempted independently. Streaming data is probed with `next(iter(train_ds))`, avoiding the assumption that it supports indexing.
4. **第 154-157 行 / Lines 154-157 (combine)**:
   - 中文：如果所有 config 都失败，立即报错；否则把多个来源拼成一个训练流。
   - English: If every config fails, the loader stops early; otherwise, it concatenates the surviving sources into one training dataset.
5. **第 159-165 行 / Lines 159-165 (shuffle and DDP shard)**:
   - 中文：只有非流式数据做随机打乱，因为 iterable stream 没有普通的全局 shuffle 语义。分布式时在 dataset 层切 shard，替代 `DistributedSampler`。
   - English: Only non-streaming data gets ordinary shuffling. In distributed mode, sharding happens at the dataset level because an iterable stream does not fit the usual sampler model.
6. **第 168-175 行 / Lines 168-175 (validation split)**:
   - 中文：流式数据先 `take(val_size)`，再从同一条流 `skip(val_size)` 得到训练部分；随机访问数据则用 `select` 做精确索引切分。
   - English: A stream uses `take(val_size)` for validation and `skip(val_size)` for training, while a random-access dataset uses exact `select` ranges.

## 类比 / The analogy

中文：像一条没有页码的传送带。你不能先跳到第 10 万件货再拿 100 件做验证，只能先取前 100 件贴上 validation 标签，再让传送带继续往前走，把剩余货物交给训练工位。

English: Think of a conveyor belt with no page numbers. You cannot jump to item 100,000 and select a validation slice. You label the first 100 items as validation, then let the belt continue and send the remainder to training.

## 自己跑一遍 / Try it yourself

```python
from itertools import islice


def split_stream(stream, val_size):
    iterator = iter(stream)
    validation = list(islice(iterator, val_size))
    training = list(iterator)
    return validation, training


val, train = split_stream(["a", "b", "c", "d", "e"], 2)
print(val)
print(train)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
['a', 'b']
['c', 'd', 'e']
```

中文：`islice` 只消费前两项，后面的迭代器状态自然落在训练起点；这就是 `take`/`skip` 的核心语义。

English: `islice` consumes only the first two items, leaving the iterator positioned at the start of training. That is the essential meaning of `take` and `skip`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Hugging Face Datasets streaming** / **Hugging Face Datasets streaming**: iterable datasets trade random access for bounded download and memory. / Iterable datasets trade random access for bounded download and memory.
- **WebDataset shards** / **WebDataset shards**: workers consume assigned tar shards instead of a global sampler. / Workers consume assigned tar shards instead of a global sampler.
- **PyTorch IterableDataset** / **PyTorch IterableDataset**: worker-aware iteration must be designed explicitly. / Worker-aware iteration must be designed explicitly.

## 注意事项 / Caveats / when it breaks

- **stream 只能消费一次** / **Streams are single-pass**: 取 validation 后不能复用同一个 iterator 重新从头训练。 / After taking validation, the same iterator cannot be reused from the beginning.
- **多卡切分顺序很重要** / **Shard before splitting carefully**: 每个 rank 的 `val_size` 和数据顺序必须一致，否则验证集会重叠或缺样本。 / Each rank needs consistent ordering and `val_size`, or validation may overlap or lose examples.
- **流式 shuffle 不是免费操作** / **Streaming shuffle is not free**: 需要 buffer-based shuffle，并接受它只是近似随机。 / Buffer-based shuffle is approximate and has a memory tradeoff.

## 延伸阅读 / Further reading

- [nanoVLM train.py](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/train.py)
- [Hugging Face Datasets streaming guide](https://huggingface.co/docs/datasets/stream)
