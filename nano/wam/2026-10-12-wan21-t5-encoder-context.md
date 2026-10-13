---
date: 2026-10-12
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/t5.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/t5.py#L472-L513
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, text-conditioning, t5]
build_role: text-conditioning advanced variant, trimmed prompt context tokens
---

# Wan2.1 T5 encoder：文本先变 context，再剪回有效长度 / Wan2.1 T5 Encoder: Turn Text into Context, Then Trim to Valid Length

> **一句话 / In one line**: Wan2.1 用冻结的 UMT5 encoder 把 prompt 编成 context token，并根据 mask 去掉 padding 部分。 / Wan2.1 uses a frozen UMT5 encoder to turn prompts into context tokens, then removes padded positions according to the mask.

## 为什么重要 / Why this matters

WAM 不是只看视频 latent。文本 prompt 需要变成一串上下文 token，后面的 DiT block 才能用 cross-attention 或条件注入理解“要生成什么”。这段包装器把三件事放在一起：初始化冻结文本 encoder，初始化 tokenizer，调用时把 padding 后的 batch 输出剪回每条 prompt 的真实长度。

A WAM does not look only at video latents. A text prompt must become context tokens so later DiT blocks can use cross-attention or conditioning to know what to generate. This wrapper groups three jobs: initialize a frozen text encoder, initialize the tokenizer, and trim padded batch outputs back to each prompt's real length.

## 代码 / The code

`Wan-Video/Wan2.1` - [`wan/modules/t5.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/t5.py#L472-L513)

```python
class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device).eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
```

## 逐行讲解 / What's happening

1. **第 474-487 行 / Lines 474-487 (constructor arguments)**:
   - 中文: `text_len` 是最大 token 长度，`dtype/device` 控制文本 encoder 的运行位置，`shard_fn` 允许分布式环境把 T5 分片。
   - English: `text_len` is the maximum token length, `dtype/device` choose where the text encoder runs, and `shard_fn` lets distributed setups shard T5.

2. **第 490-497 行 / Lines 490-497 (frozen encoder)**:
   - 中文: `encoder_only=True` 表示这里只要文本编码，不要 decoder；`.eval().requires_grad_(False)` 把它固定成条件提取器。
   - English: `encoder_only=True` keeps only text encoding, not decoding. `.eval().requires_grad_(False)` turns it into a fixed conditioning extractor.

3. **第 498-504 行 / Lines 498-504 (placement and tokenizer)**:
   - 中文: 如果有 `shard_fn`，模型交给外部分片逻辑；否则搬到当前 device。tokenizer 则用 whitespace 清理并固定 `seq_len`。
   - English: With `shard_fn`, placement is delegated to external sharding; otherwise the model moves to the chosen device. The tokenizer cleans whitespace and uses a fixed `seq_len`.

4. **第 506-513 行 / Lines 506-513 (`__call__`)**:
   - 中文: tokenizer 返回 padded `ids` 和 `mask`。`seq_lens = mask.gt(0).sum(...)` 找到每条文本的真实长度，最后把 encoder 输出逐条切掉 padding。
   - English: The tokenizer returns padded `ids` and `mask`. `seq_lens = mask.gt(0).sum(...)` finds each prompt's real length, and the final list comprehension trims padding from each encoded context.

## 类比 / The analogy

这像把每张纸都塞进同样大小的透明文件袋，方便整批搬运；到了桌上再按每张纸真实长度裁掉空白边。batch padding 是文件袋，mask 是裁切线。

It is like putting pages into same-sized transparent sleeves for batch transport, then cutting away blank margins at the desk. Batch padding is the sleeve, and the mask is the cut line.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，这是 `text-conditioning` 组件。上游输入是 prompt 字符串列表，下游输出是每条 prompt 的 context token 序列，通常喂给 DiT 的 cross-attention 或条件投影层。如果省掉这个模块，WAM 就只能做无文本或固定条件生成。生产级实现还要处理 prompt cache、negative prompt、CPU/GPU offload、分布式 shard、以及 tokenizer 和 checkpoint 的版本绑定。

In a nanoWAM, this is the `text-conditioning` component. Upstream input is a list of prompt strings. Downstream output is one context-token sequence per prompt, usually consumed by DiT cross-attention or conditioning projection layers. Without it, the WAM can only do unconditional or fixed-condition generation. A production implementation should add prompt caching, negative prompts, CPU/GPU offload, distributed sharding, and tokenizer/checkpoint version binding.

## 自己跑一遍 / Try it yourself

```python
def tokenize(texts, max_len):
    ids, masks = [], []
    for text in texts:
        toks = text.split()[:max_len]
        pad = max_len - len(toks)
        ids.append(toks + ["<pad>"] * pad)
        masks.append([1] * len(toks) + [0] * pad)
    return ids, masks

def encode(ids):
    return [[token.upper() for token in row] for row in ids]

ids, mask = tokenize(["red cube", "move left slowly"], 4)
context = encode(ids)
seq_lens = [sum(row) for row in mask]
print([row[:n] for row, n in zip(context, seq_lens)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[['RED', 'CUBE'], ['MOVE', 'LEFT', 'SLOWLY']]
```

示例里没有真正的 T5，但保留了核心数据流：padding batch 方便并行，最后根据 mask 还原有效 context。

The example does not run a real T5, but it preserves the core data flow: pad for batch processing, then use the mask to recover valid context.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers text encoders** / **Diffusers text encoders**: Stable Diffusion 类 pipeline 也先把 prompt 编成 hidden states，再给 UNet 或 Transformer 使用。 / Stable Diffusion-style pipelines also encode prompts into hidden states before passing them into a UNet or Transformer.
- **Open-Sora text conditioning** / **Open-Sora text conditioning**: Open-Sora 也有文本 embedder，把 tokenizer、encoder 和 mask 处理封装成条件入口。 / Open-Sora also wraps tokenizer, encoder, and mask handling as a conditioning entry point.

## 注意事项 / Caveats / when it breaks

- **默认 device 表达式会触发 CUDA 查询** / **The default device expression queries CUDA**: `device=torch.cuda.current_device()` 在无 CUDA 环境里不适合直接构造。生产代码常把 device 设成显式参数。 / `device=torch.cuda.current_device()` is not friendly to CPU-only construction. Production code often makes device explicit.
- **输出是 list，不是 padded tensor** / **The output is a list, not a padded tensor**: 下游模块必须能处理变长 context，或者自己重新 padding。 / Downstream modules must handle variable-length context or pad it again.

## 延伸阅读 / Further reading

- [Source permalink](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/t5.py#L472-L513)
- [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1)

