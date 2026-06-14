---
date: 2026-06-14
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/text/conditioner.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/text/conditioner.py#L1-L74
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, text-conditioning, t5, clip, seq-align, tensor-parallel]
build_role: text-conditioning (cross-repo variant — production-grade companion to Wan2.1's cross-attention text injection)
---

# Open-Sora 把"T5 文本编码"做成 46 行,顺手解决了 TP 的对齐难题 / Open-Sora's 46-line "T5 text encoder" sneaks in a tensor-parallel-friendly alignment trick

> **一句话 / In one line**: 一个 `HFEmbedder` 类同时承担 T5(序列输出)和 CLIP(pooled 输出)两种 text encoder,并通过把 input_ids pad 到 `seq_align` 的整数倍,让下游 DiT 的 sequence parallel 永远能整除。 / A single `HFEmbedder` class serves as both a T5 (sequence) and a CLIP (pooled) text encoder, and quietly pads input_ids until total length is a multiple of `seq_align` so the downstream DiT's sequence-parallel sharding always divides evenly.

## 为什么重要 / Why this matters

视频扩散模型几乎都用 T5-XXL(4096 维 sequence)做文本条件——CLIP 的 77 token pooled vector 信息量不够。但 T5 输出的长度是 `tokenizer(text, max_length=N)` 决定的——而 DiT 那边用 SP / TP 跑长 context,要求总 token 数能被 world_size 整除。如果你不管这事,运行时会偶发 "shape mismatch in all_to_all" 之类的 NCCL 错误,而且只在某些 batch 触发,极难复现。Open-Sora 的 `HFEmbedder.forward` 第 41-46 行用 5 行 `nn.functional.pad` 优雅地解决了这个问题。整个 46 行的 forward 是文生视频 / 文生图模型里 **text-conditioning** 的标准模板。

Video diffusion models almost universally use T5-XXL (4096-dim sequence) for text conditioning — CLIP's 77-token pooled vector simply isn't expressive enough. But the T5 output length is whatever the tokenizer returns for `max_length=N` — and meanwhile the DiT runs SP / TP across long context, requiring total token count divisible by world_size. Ignore this, and you'll hit sporadic "shape mismatch in all_to_all" NCCL errors that fire on some batches and not others — a nightmare to reproduce. Open-Sora's `HFEmbedder.forward` solves it cleanly in five lines of `nn.functional.pad` (lines 41-46). The entire 46-line forward is the standard **text-conditioning** template for any text-to-video / text-to-image system.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/text/conditioner.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/text/conditioner.py#L1-L74)

```python
from colossalai.shardformer import ShardConfig, ShardFormer
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from opensora.acceleration.shardformer.policy.t5_encoder import T5EncoderPolicy
from opensora.registry import MODELS


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
```

## 逐行讲解 / What's happening

1. **第 12-15 行 / Lines 12-15 (双模式分支)**:
   - 中文: 一个布尔 `is_clip = "openai" in from_pretrained`——靠 model name 中是否带 "openai" 来推断是 CLIP 还是 T5。`output_key = "pooler_output" if is_clip else "last_hidden_state"`:CLIP 输出一个 `(B, D)` 的池化向量,T5 输出 `(B, L, D)` 的序列。两种模型公用一个类,靠这个 key 切换。
   - English: One bool: `is_clip = "openai" in from_pretrained` — infers CLIP-vs-T5 from the model name. `output_key = "pooler_output" if is_clip else "last_hidden_state"`. CLIP returns a `(B, D)` pooled vector; T5 returns a `(B, L, D)` sequence. One class, two backends, switched by this key.

2. **第 27 行 / Line 27 (shardformer for T5)**:
   - 中文: `shardformer_t5` 用 ColossalAI 的 ShardFormer 自动把 T5 的 attention 拆 TP,顺便开 JIT-fused 优化。这是给 T5-XXL 准备的——一张 80GB H100 跑不下纯 fp32 的 T5-XXL,必须 shard。
   - English: `shardformer_t5` uses ColossalAI's ShardFormer to TP-split T5's attention and enable JIT fusion. This is for T5-XXL — a single 80GB H100 can't hold fp32 T5-XXL on its own, so TP is mandatory.

3. **第 29 行 / Line 29 (`eval().requires_grad_(False)`)**:
   - 中文: 文本编码器永远是 frozen 的,扩散模型从不更新它。这一行用 `requires_grad_(False)` 同时让 PyTorch 跳过这部分的梯度计算图——既省显存又省算力,而且 `eval()` 关掉了 dropout。
   - English: Text encoders are always frozen — diffusion training never updates them. `requires_grad_(False)` skips graph construction for these params, saving memory and compute, while `eval()` disables dropout.

4. **第 32-40 行 / Lines 32-40 (the tokenizer call)**:
   - 中文: `padding="max_length"` 强制把每条文本都 pad 到 `max_length`(通常 256 或 512),`truncation=True` 砍掉超出的。结果 `input_ids` 形状一定是 `(B, max_length)`——一个稳定的 shape 对扩散模型的 batch dim 很重要,否则每个 batch 形状都不同会让 dynamic_shape compile 不稳。
   - English: `padding="max_length"` pads every example to exactly `max_length` (typically 256 or 512); `truncation=True` chops off the rest. `input_ids` is guaranteed shape `(B, max_length)` — a stable shape that matters for diffusion training where varying batch shapes would destabilise dynamic-shape compile.

5. **第 41-46 行 / Lines 41-46 (the seq_align trick — the gem)**:
   - 中文: 这是整个文件的灵魂。下游 DiT 用 SP/TP,要求 `(added_tokens + text_seq_len)` 能被 `seq_align` 整除——`added_tokens` 是 DiT 那边视频 latent 加 timestep 嵌入的总长度。算出还差几个 token 才能整除,然后 `F.pad(input_ids, (0, num_pad_tokens), value=pad_token_id)` 在右侧补 pad token。注意 pad value 用 `tokenizer.pad_token_id` 而不是 0——T5 的 attention 把 pad token 自然忽略掉了,所以不会影响语义。
   - English: The heart of the file. The downstream DiT does SP/TP and requires `(added_tokens + text_seq_len) % seq_align == 0`, where `added_tokens` is the DiT-side video-latent + timestep tokens. We compute how many extras are needed, then `F.pad(input_ids, (0, num_pad_tokens), value=pad_token_id)` appends pad tokens on the right. Crucially, the pad value is `tokenizer.pad_token_id`, not 0 — T5's attention mask naturally ignores pad tokens, so the semantics are unaffected.

6. **第 48-52 行 / Lines 48-52 (encoder forward)**:
   - 中文: `attention_mask=None` 这个细节有意思——这里**不传 mask**是因为下游 DiT 自己会处理"哪些 token 是 padding",这里只要文本特征序列。`output_hidden_states=False` 节约一份显存。
   - English: Setting `attention_mask=None` is intentional — the downstream DiT computes its own mask. Here we only want the feature sequence. `output_hidden_states=False` saves one tensor's worth of memory.

## 类比 / The analogy

想象你在做一个流水线工厂,每条生产线必须每 8 个零件一组才能装入下一道工序的传送带。来料的纸箱里通常有 5 个零件,有时 7 个。你不能把 5 个直接送上,会卡住。`seq_align=8` 这个机制就是工厂里那个补料机:看到 5 个就自动添 3 个空白件凑成 8(空白件下游会被识别为"不要处理")。这样下游的传送带永远在整齐节拍上运转,不会因为某天来料数量奇怪就卡线。pad_token_id 就是那种带"忽略我"标签的空白件。

Picture an assembly line where the next conveyor only accepts widgets in groups of 8. Your supplier ships boxes of 5 widgets (sometimes 7). Putting 5 directly onto the conveyor jams it. The `seq_align=8` mechanism is the dummy-filler station: see 5, add 3 blank widgets labelled "ignore me" to make 8. The line runs in clean rhythm regardless of supplier irregularities. `pad_token_id` is the "ignore me" tag.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 **`text-conditioning`** curriculum slot 的**编码器侧**——我们在 2026-05-29 看的 Wan2.1 cross-attention 是**消费侧**(DiT 怎么吃 T5 特征);今天这是**生产侧**(T5 特征怎么算出来)。两者构成一对完整的 text-conditioning 流水线。在 nanoWAM 里,你的 `__init__` 长这样:`self.text_encoder = HFEmbedder("t5-v1_1-xxl", max_length=256, seq_align=world_size)`,然后 `forward(text)` 一次就拿到 `(B, max_length_padded, D_text)` 的特征,直接喂给 DiT 的 cross-attention KV。依赖关系上,text-conditioning 是 `dit-block` 的依赖项(在 curriculum 里就这么写的)——没它 DiT 没语义信号。生产实现还要加:**negative prompt 的并行编码**(CFG 用,所以一次得编码 2B 条文本)、**T5 离线 cache**(同一个 prompt 反复出现时,事先编码好存盘,推理时直接 load)、**multi-stage classifier-free guidance schedule**(不同 timestep 用不同 CFG 强度)。

This is the **encoder side** of the `text-conditioning` slot — Wan2.1's cross-attention (covered 2026-05-29) was the **consumer side** (how the DiT eats T5 features); today is the **producer side** (how T5 features are computed). Together they form one complete text-conditioning pipeline. In your nanoWAM, the `__init__` looks like: `self.text_encoder = HFEmbedder("t5-v1_1-xxl", max_length=256, seq_align=world_size)`, then `forward(text)` hands you `(B, max_length_padded, D_text)` to plug into the DiT's cross-attention KV. Dependency-wise, text-conditioning is a `dit-block` dependency in the curriculum — without it the DiT has no semantic signal. To turn this into production, add: **negative-prompt parallel encoding** (CFG needs both pos and neg, so encode 2B sentences in one batch), **T5 offline caching** (re-encoding the same prompt is wasteful; pre-compute and disk-cache), and a **multi-stage CFG schedule** (different CFG strength at different timesteps).

## 自己跑一遍 / Try it yourself

```python
import torch
from torch import nn
from transformers import T5Tokenizer, T5EncoderModel


class HFEmbedder(nn.Module):
    def __init__(self, name="google/t5-v1_1-small", max_length=64):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(name, legacy=True)
        self.encoder = T5EncoderModel.from_pretrained(name).eval().requires_grad_(False)

    def forward(self, text, added_tokens=0, seq_align=1):
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        seq_len = enc["input_ids"].shape[1]
        if (added_tokens + seq_len) % seq_align != 0:
            n = seq_align - (added_tokens + seq_len) % seq_align
            enc["input_ids"] = nn.functional.pad(enc["input_ids"], (0, n), value=self.tokenizer.pad_token_id)
        out = self.encoder(input_ids=enc["input_ids"], output_hidden_states=False)
        return out.last_hidden_state


emb = HFEmbedder(max_length=12)
feat = emb(["a cat surfing"], added_tokens=4, seq_align=8)
print("padded text feat shape:", feat.shape)   # (1, 16, 512) — 12 + 4 padded to nearest multiple of 8
```

运行 / Run with:
```bash
pip install "transformers>=4.45" "torch>=2.4" sentencepiece
python try.py
```

预期输出 / Expected output:
```
padded text feat shape: torch.Size([1, 16, 512])
```

中文一句:文本长度本是 12,加上 `added_tokens=4` 等于 16,正好是 8 的倍数——所以**不会** pad。把 `seq_align` 改成 7,你会看到 shape 变成 `(1, 19, 512)`(16 不是 7 的倍数,要补 3 个 token 凑齐 21,减掉 added_tokens 还剩 19)。

English: text length is 12, plus `added_tokens=4` is 16 — already a multiple of 8, so **no padding** happens. Set `seq_align=7` and you'll see the shape change to `(1, 19, 512)` (16 isn't divisible by 7; we pad three to reach 21, minus added_tokens leaves 19 on the text side).

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 text cross-attention**(2026-05-29 那篇)/ **Wan2.1 text cross-attention** (covered 2026-05-29): 消费 T5 特征的那一端;今天这是生产端,两者一对。 / The consumer of T5 features; today is the producer — a paired duo.
- **Diffusers `T5EncoderModel` wrapper** / **Diffusers `T5EncoderModel` wrapper**: 也用 max_length padding,但没有 `seq_align`——Diffusers 假设你单卡推理,不做 SP。 / Same `max_length` padding but no `seq_align` — Diffusers assumes single-card inference, no SP.
- **CogVideoX `FrozenT5Embedder`** / **CogVideoX `FrozenT5Embedder`**: 几乎一模一样的最小版本,76 token max_length;没做 alignment 处理。 / Nearly identical minimal version, `max_length=77`; no alignment logic.
- **Flux text encoder** / **Flux text encoder**: 用同样的双 backbone(CLIP 给 pooled vector、T5 给 sequence),代码骨架也借鉴自 Open-Sora 的 `HFEmbedder`。 / Uses the same dual-backbone setup (CLIP for pooled vec, T5 for sequence); its code skeleton borrows from Open-Sora's `HFEmbedder`.

## 注意事项 / Caveats / when it breaks

- **pad_token_id 不能用 0** / **Don't hardcode 0 as the pad token**: T5 的 `pad_token_id` 是 0 没错,但 CLIP 的是 49407;靠 tokenizer 自己提供这个值才安全。 / T5's `pad_token_id` is 0, but CLIP's is 49407; rely on `tokenizer.pad_token_id`, never hardcode.
- **shardformer 用 CLIP 会 assert** / **`shardformer` + CLIP asserts**: 这是有意的——CLIP 太小,shard TP 反而慢。 / Intentional — CLIP is too small for TP to pay off.
- **`attention_mask=None` 不是 bug** / **`attention_mask=None` is intentional**: 但这意味着 T5 的 attention 内部默认所有 token 都参与(包括 pad)。下游 DiT 一定要自己用 `(input_ids != pad_id)` mask 掉那些位置,否则模型会被 pad token 的 hidden state 污染。 / This means T5's attention internally treats every token (including pads) as valid. The downstream DiT *must* construct its own `(input_ids != pad_id)` mask, or pad-token hidden states pollute training.
- **max_length 改了要重新 fine-tune** / **Change `max_length` and you must re-finetune**: 训练时用 256,推理用 512,DiT 看到的 text 序列长度对不上,cross-attention 会失配。 / If you train at 256 and infer at 512, the DiT's cross-attention sees a mismatched text-sequence length and gets confused.

## 延伸阅读 / Further reading

- [T5 paper, "Exploring the Limits of Transfer Learning"](https://arxiv.org/abs/1910.10683)
- [Imagen paper, § text encoder](https://arxiv.org/abs/2205.11487)
- [ColossalAI ShardFormer docs](https://colossalai.org/docs/features/shardformer)
- Wan2.1 text cross-attention note (daily-code 2026-05-29) — the consumer end of this pipeline
