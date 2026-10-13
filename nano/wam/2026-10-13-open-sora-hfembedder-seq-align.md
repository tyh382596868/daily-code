---
date: 2026-10-13
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/text/conditioner.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/text/conditioner.py#L9-L74
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, text-conditioning, t5]
build_role: text-conditioning advanced variant, sequence-aligned frozen text embedder
---

# Open-Sora HFEmbedder：文本 token 要补到并行友好的长度 / Open-Sora HFEmbedder: Pad Text Tokens to a Parallel-Friendly Length

> **一句话 / In one line**: Open-Sora 把 CLIP/T5 文本 encoder 冻结，并按 `seq_align` 补齐 token 长度。 / Open-Sora freezes CLIP/T5 text encoders and pads token length according to `seq_align`.

## 为什么重要 / Why this matters

WAM 的文本条件会进入大视频 DiT，序列长度不仅要有语义，还要适合 tensor/sequence parallel 切分。

Text conditioning in a WAM feeds a large video DiT, so sequence length must carry semantics and also divide cleanly for tensor or sequence parallelism.

## 代码 / The code

`hpcaitech/Open-Sora` - [`opensora/models/text/conditioner.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/text/conditioner.py#L9-L74)

```python
@MODELS.register_module("text_embedder")
class HFEmbedder(nn.Module):
    def __init__(self, from_pretrained: str, max_length: int, shardformer: bool = False, **hf_kwargs):
        super().__init__()
        self.is_clip = "openai" in from_pretrained
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(from_pretrained, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(from_pretrained, **hf_kwargs)
            assert not shardformer, "Shardformer is not supported for CLIP"
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                from_pretrained, max_length=max_length, legacy=True
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(from_pretrained, **hf_kwargs)
            if shardformer:
                self.hf_module = shardformer_t5(self.hf_module)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str], added_tokens: int = 0, seq_align: int = 1) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        seq_len = batch_encoding["input_ids"].shape[1]
        if (added_tokens + seq_len) % seq_align != 0:
            num_pad_tokens = seq_align - (added_tokens + seq_len) % seq_align
            batch_encoding["input_ids"] = nn.functional.pad(
                batch_encoding["input_ids"], (0, num_pad_tokens), value=self.tokenizer.pad_token_id
            )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


def shardformer_t5(t5: T5EncoderModel) -> T5EncoderModel:
    """
    Shardformer for T5 model

    Args:
        t5: T5 model to be optimized

    Returns:
        optimized T5 model
    """
    dtype = t5.shared.weight.dtype
    shard_config = ShardConfig(
        enable_tensor_parallelism=False,
        enable_jit_fused=True,
    )
    shard_former = ShardFormer(shard_config=shard_config)
    optim_model, _ = shard_former.optimize(t5, policy=T5EncoderPolicy())
    optim_model = optim_model.to(dtype).eval().requires_grad_(False)
    return optim_model
```

## 逐行讲解 / What's happening

1. **第 13-29 行 / Lines 13-29 (CLIP vs T5)**:
   - 中文: 根据 checkpoint 选择 encoder，并冻结成条件提取器。
   - English: It chooses the encoder from the checkpoint and freezes it as a condition extractor.
2. **第 31-47 行 / Lines 31-47 (alignment padding)**:
   - 中文: `added_tokens + seq_len` 不能整除 `seq_align` 时右侧补 pad token。
   - English: If `added_tokens + seq_len` is not divisible by `seq_align`, it pads on the right.
3. **第 48-53 行 / Lines 48-53 (output key)**:
   - 中文: CLIP 和 T5 通过 `output_key` 统一接口。
   - English: CLIP and T5 share one interface through `output_key`.
4. **第 56-74 行 / Lines 56-74 (ShardFormer)**:
   - 中文: T5 可走 fused/JIT 优化后继续保持冻结。
   - English: T5 can be optimized with fused/JIT paths and remain frozen.

## 类比 / The analogy

这像电影院排座：最后一排也要补空座，放映时每排才能整齐推进。

It is like seating a cinema: empty seats are added in the last row so projection advances cleanly row by row.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `text-conditioning` 的高级版本，位于 prompt tokenizer 和 DiT 条件注入之间；小模型可先不做并行对齐，生产级长序列模型最好从入口就处理。

English: This is an advanced `text-conditioning` component between prompt tokenization and DiT conditioning; tiny models may skip alignment, but production long-sequence models should handle it at the boundary.

## 自己跑一遍 / Try it yourself

```python
def aligned_length(seq_len, added_tokens, seq_align):
    r = (added_tokens + seq_len) % seq_align
    return seq_len if r == 0 else seq_len + (seq_align - r)
print(aligned_length(77, 3, 8))
print(aligned_length(78, 3, 8))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
77
85
```

中文: 对齐看的是文本 token 加额外条件 token 后的总长度。

English: Alignment is computed after adding extra condition tokens to text tokens.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 T5 encoder** / **Wan2.1 T5 encoder**: Wan2.1 T5 encoder。 / Wan2.1 T5 encoder.
- **Transformer packed batches** / **Transformer packed batches**: Transformer packed batches。 / Transformer packed batches.

## 注意事项 / Caveats / when it breaks

- **attention mask 被省略** / **Swapping encoders requires checking pad-token behavior**: attention mask 被省略。 / Swapping encoders requires checking pad-token behavior.
- **CLIP 不支持 shardformer** / **The code explicitly blocks ShardFormer for CLIP**: CLIP 不支持 shardformer。 / The code explicitly blocks ShardFormer for CLIP.

## 延伸阅读 / Further reading
- [Open-Sora conditioner.py](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/text/conditioner.py#L9-L74)
