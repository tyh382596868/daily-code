---
date: 2026-09-13
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: bindings/python/examples/train_parity_bpe.py
permalink: https://github.com/huggingface/tokenizers/blob/6cfd9d385ca0ed91c10b49f0ce97d02cfde1b607/bindings/python/examples/train_parity_bpe.py#L35-L84
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers, bpe, multilingual, compression-parity]
---

# tokenizers Parity BPE：让低资源语言也拿到 merge 预算 / tokenizers Parity BPE: Give Low-Resource Languages a Share of the Merge Budget

> **一句话 / In one line**: `ParityBpeTrainer` 不让最大语料独占 BPE merge，而是用平行 dev 集或目标压缩率给每种语言一个反馈信号。 / `ParityBpeTrainer` stops the largest corpus from monopolizing BPE merges by using a parallel dev set or target compression ratios as feedback for each language.

## 为什么重要 / Why this matters

中文：多语言 tokenizer 的共享词表是一个竞争市场。英语语料更多、字节模式更密集时，普通 BPE 会把 merge 预算主要花在英语上，低资源语言只能用更长的 token 序列表达相同句子。这个示例展示了两种约束：用对齐的 FLORES+ dev 集直接观察各语言 token 数，或者用按字节归一化的 `ratio` 指定目标。

English: A multilingual tokenizer's shared vocabulary is a competitive market. With more English data and denser byte patterns, ordinary BPE spends most merges on English, leaving low-resource languages with longer sequences. This example shows two controls: measure token counts on aligned FLORES+ development text, or provide byte-normalized target `ratio` values.

## 代码 / The code

`huggingface/tokenizers` — [`bindings/python/examples/train_parity_bpe.py`](https://github.com/huggingface/tokenizers/blob/6cfd9d385ca0ed91c10b49f0ce97d02cfde1b607/bindings/python/examples/train_parity_bpe.py#L35-L84)

```python
# Target compression rates, one per language, in the order of LANGUAGES.
RATIOS = [1.00, 1.19, 1.23, 2.57]

ARTICLES_PER_LANGUAGE = 200
NUM_MERGES = 8000


def wikipedia_iterator(config, limit=ARTICLES_PER_LANGUAGE):
    """Stream up to `limit` articles for one language."""
    dataset = datasets.load_dataset(WIKIPEDIA, config, split="train", streaming=True)
    for count, article in enumerate(dataset):
        if count >= limit:
            return
        yield article["text"]


def flores_iterator(config):
    """The FLORES+ dev split: 997 aligned sentences."""
    dataset = datasets.load_dataset(FLORES_PLUS, config, split="dev")
    yield from dataset["text"]


def new_tokenizer():
    """Byte-level BPE, so every language is representable."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def new_trainer():
    return ParityBpeTrainer(
        num_merges=NUM_MERGES,
        variant="window",
        window_size=100,
        alpha=2.0,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
```

## 逐行讲解 / What's happening

1. **第 35-45 行 / Lines 35-45 (`RATIOS`)**:
   - 中文：`ratio` 的绝对值不如相对值重要。Hindi 使用更大的目标，是因为 ByteLevel 按 UTF-8 字节计数，Devanagari 每个字符通常占更多字节。
   - English: Relative ratios matter more than their absolute scale. Hindi gets a larger target because ByteLevel counts UTF-8 bytes and Devanagari characters usually occupy more bytes.
2. **第 47-60 行 / Lines 47-60 (streaming iterator)**:
   - 中文：每种语言只流式读取有限文章，避免为了训练 tokenizer 把整份 Wikipedia dump 一次性装进内存。
   - English: Each language is streamed with a cap, avoiding a full Wikipedia dump in memory just to train a tokenizer.
3. **第 63-66 行 / Lines 63-66 (parallel dev text)**:
   - 中文：FLORES+ 的同一句子有多个语言版本，所以 token count 的比较不受句子内容差异干扰。
   - English: FLORES+ provides translations of the same sentences, so token-count comparisons are less confounded by different content.
4. **第 69-74 行 / Lines 69-74 (ByteLevel BPE)**:
   - 中文：ByteLevel 先保证任意 Unicode 文本都有表示路径，再让 Parity trainer 决定哪些 byte sequence 值得 merge。
   - English: ByteLevel first guarantees a path for arbitrary Unicode text, then ParityBpeTrainer decides which byte sequences deserve merges.
5. **第 77-84 行 / Lines 77-84 (window variant)**:
   - 中文：`window_size` 和 `alpha` 限制某种语言在一个窗口内连续获得 merge 的比例，避免“追赶语言”短暂垄断下一批 merge。
   - English: `window_size` and `alpha` limit how much one language can dominate a merge window, preventing a catching-up language from monopolizing the next batch.
6. **训练调用的关键差异 / The key training difference**:
   - 中文：`dev_iterators=[...]` 是以 held-out 平行文本为反馈；`ratio=RATIOS` 不需要平行数据，而是直接指定训练过程中的压缩目标。两种模式都共享一个 vocabulary。
   - English: `dev_iterators=[...]` uses held-out parallel text as feedback; `ratio=RATIOS` needs no parallel data and directly specifies compression targets. Both modes still share one vocabulary.

## 类比 / The analogy

中文：像给四个国家共用的一间厨房分配砧板。普通 BPE 让声音最大、订单最多的厨房一直拿新砧板；Parity BPE 每隔一段时间看各国菜品切得是否一样细，再把下一批砧板发给最需要的一方。

English: Imagine one kitchen with cutting boards shared by four countries. Ordinary BPE keeps giving boards to the loudest, busiest kitchen; Parity BPE checks how finely each cuisine is being cut and allocates the next batch where it is most needed.

## 自己跑一遍 / Try it yourself

```python
def allocate_merges(needs, budget):
    counts = [0] * len(needs)
    for _ in range(budget):
        winner = min(range(len(needs)), key=lambda i: counts[i] / needs[i])
        counts[winner] += 1
    return counts


print(allocate_merges([1.0, 1.2, 2.6], 12))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[3, 3, 6]
```

中文：这里把“需要更多 merge 的语言”简化成更大的 `needs`。真实 trainer 不是静态按比例发放，而是根据窗口内 compression rate、dev iterator 或 ratio 动态选下一次 merge。

English: The toy treats a larger `needs` value as a language needing more merges. The real trainer does not allocate statically; it chooses merges from windowed compression feedback, a dev iterator, or target ratios.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SentencePiece sampling** / **SentencePiece sampling**: 训练数据混合比例也会改变词表覆盖。 / The mixture ratio of training data also changes vocabulary coverage.
- **multilingual data loaders** / **multilingual data loaders**: temperature sampling gives low-resource languages more batches. / Temperature sampling gives low-resource languages more batches.
- **token budget routing** / **token budget routing**: 先测量序列长度，再把有限计算预算分配到最拥挤的路径。 / Measure sequence length first, then spend limited compute on the most crowded paths.

## 注意事项 / Caveats / when it breaks

- **Byte ratio 不是字符 ratio** / **Byte ratio is not character ratio**: 不同脚本的 UTF-8 字节宽度会改变结果。
- **dev set 和训练语料要分开** / **Keep dev data separate from training data**: 只在训练语料上平衡，可能把 tokenizer 调到训练集最优却牺牲泛化。
- **共享词表仍然存在竞争** / **The vocabulary is still shared**: Parity 会缩小差距，不会让所有语言获得完全相同的 token 数。
- **FLORES+ 是 gated 数据集** / **FLORES+ is gated**: 实际运行 dev-set demo 需要接受数据集条款并完成 Hub 登录；ratio demo 不依赖它。
- **generator 只能消费一次** / **Generators are single-use**: 示例在 dev 和 ratio 两种模式中分别重建 iterator。

## 延伸阅读 / Further reading

- [Parity BPE example](https://github.com/huggingface/tokenizers/blob/6cfd9d385ca0ed91c10b49f0ce97d02cfde1b607/bindings/python/examples/train_parity_bpe.py)
- [Hugging Face tokenizers](https://github.com/huggingface/tokenizers)
