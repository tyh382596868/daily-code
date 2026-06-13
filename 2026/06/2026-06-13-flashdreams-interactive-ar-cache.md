---
date: 2026-06-13
topic: diffusion
source: trending
repo: NVIDIA/flashdreams
file: flashdreams/recipes/wan/pipeline.py
permalink: https://github.com/NVIDIA/flashdreams/blob/83733233701ce37320c45ef06dd1423c42f0c6e6/flashdreams/recipes/wan/pipeline.py#L170-L289
difficulty: advanced
read_time: ~11 min
tags: [code-of-the-day, diffusion, world-model, autoregressive, kv-cache, inference]
---

# NVIDIA flashdreams 的 `initialize_cache`:一次性 encode + 流式 VAE,这是交互式 AR 视频生成的设计核心 / NVIDIA flashdreams' `initialize_cache`: one-shot encode + streaming VAE — the design core of interactive AR video generation

> **一句话 / In one line**: text encoder 跑一次,negative prompt 也只在 CFG > 1 时跑一次,CLIP 图像编码也只跑一次 —— 全部"one-shot encode"完后释放;**唯独 VAE encoder 不在这里跑,因为它必须随 AR 步骤流式推进自己的时间缓存**,这一条设计决定就是交互式世界模型推理服务的灵魂。/ Run the text encoder once, the negative prompt only when CFG > 1, CLIP image encoder once — every "one-shot" encoder releases its weights afterward. **But the VAE encoder is deliberately *not* called here, because it must stream alongside each AR step to advance its temporal cache** — that one design decision is the soul of interactive world-model inference.

## 为什么重要 / Why this matters

NVIDIA 刚开源了 FlashDreams,一个专门为"交互式自回归视频 / 世界模型"做推理与 serving 的库 —— 把 Wan2.x、Cosmos 等大模型在 5090 / H100 上跑成"你说一句、它生成 2 秒视频、接着你再说"这种交互。这种场景跟离线 batch 生成完全不一样:每次 AR step 必须只算"增量",text/image 不能重复 encode,VAE 编码器要保持 temporal cache。这份 120 行的 `initialize_cache` 是整条 pipeline 设计的入口 —— 它把"哪些东西 encode 一次就行"和"哪些必须每步 encode" 划得清清楚楚,值得把每一个 if 都讲一遍。

NVIDIA just open-sourced FlashDreams, a library dedicated to inference + serving for **interactive autoregressive video / world models** — running Wan2.x, Cosmos etc. on 5090 / H100s in a "you talk, it generates 2 s of video, you talk again" interactive loop. This is radically different from offline batch generation: each AR step must be incremental, text/image cannot be re-encoded, and the VAE encoder has to maintain a temporal cache. This 120-line `initialize_cache` is the pipeline's entry point — it draws the hard line between "encode once" and "encode every step", and every `if` in it is worth walking through.

## 代码 / The code

`NVIDIA/flashdreams` — [`flashdreams/recipes/wan/pipeline.py`](https://github.com/NVIDIA/flashdreams/blob/83733233701ce37320c45ef06dd1423c42f0c6e6/flashdreams/recipes/wan/pipeline.py#L170-L289)

```python
def initialize_cache(
    self,
    text: list[str],
    image: Tensor | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
    release_oneshot_encoders: bool = True,
) -> WanInferencePipelineCache:
    """Initialize the per-rollout cache for a batch of prompts."""
    assert len(text) > 0, "text must be non-empty"
    n = len(text)

    self._ensure_oneshot_encoders_loaded()
    assert self.text_encoder is not None, "text_encoder is not set"
    text_embeddings = self.text_encoder(text)                        # [B, L, D]

    guidance_scale = self._transformer_config.guidance_scale
    if guidance_scale > 1.0:
        negative_text_embeddings = self.text_encoder([NEGATIVE_PROMPT] * n)
    else:
        negative_text_embeddings = None

    # Encoder presence and image presence must agree. The image is *not*
    # VAE-encoded here: that happens per AR step inside the encoder so
    # the streaming Wan VAE's temporal cache advances correctly.
    if image is not None:
        assert self.encoder is not None, (
            "Image was provided but the pipeline has no I2V input "
            "encoder; configure encoder to a WanI2VCtrlEncoderConfig."
        )
        assert image.shape[-4] == 1, (
            f"image must have a single time step (T=1), got shape {tuple(image.shape)}"
        )
    else:
        assert self.encoder is None, (
            "Image was not provided but the pipeline has an I2V input encoder."
        )

    # Derive (or cross-check) latent (height, width) from the image when
    # it is provided. The decoder owns the pixel<->latent ratio.
    if image is not None:
        assert isinstance(self.decoder, StreamingVideoDecoder), (
            f"I2V requires a StreamingVideoDecoder; got {type(self.decoder).__name__}."
        )
        sp = self.decoder.spatial_compression_ratio
        pixel_h, pixel_w = image.shape[-2], image.shape[-1]
        assert pixel_h % sp == 0 and pixel_w % sp == 0, (
            f"image pixel size ({pixel_h}, {pixel_w}) must be divisible by "
            f"decoder.spatial_compression_ratio={sp}."
        )
        derived_h, derived_w = pixel_h // sp, pixel_w // sp
        if height is None:
            height = derived_h
        else:
            assert height == derived_h
        if width is None:
            width = derived_w
        else:
            assert width == derived_w
    assert height is not None and width is not None, (
        "T2V (image=None) requires explicit `height` and `width` latent dims."
    )

    image_embeddings: Tensor | None = None
    if self.image_encoder is not None:
        assert image is not None
        # CLIP wants [..., C, H, W]; drop the T=1 axis.
        image_embeddings = self.image_encoder(image.squeeze(-4))

    parent = super().initialize_cache(
        transformer_context={
            "height": height,
            "width": width,
            "text_embeddings": text_embeddings,
            "negative_text_embeddings": negative_text_embeddings,
            "image_embeddings": image_embeddings,
        },
    )

    if release_oneshot_encoders:
        self.release_oneshot_encoders()

    return WanInferencePipelineCache(
        transformer_cache=parent.transformer_cache,
        encoder_cache=parent.encoder_cache,
        decoder_cache=parent.decoder_cache,
        image=image,
    )
```

## 逐行讲解 / What's happening

1. **第 170-198 行 / Lines 170-198 (函数签名 + docstring)**:
   - 中文: API 是"per-rollout cache 初始化" —— 一个 rollout = 一段连续的 AR 生成过程 (例如"生成一段 6 秒的视频")。注意 `release_oneshot_encoders` 默认 True —— 这是这套设计最重要的一个 flag:**encode 一次后立刻释放 encoder 占的 GPU 显存**,让 transformer 有更多内存做 AR step。
   - English: API is "per-rollout cache initialisation" — one rollout = a contiguous AR generation episode (e.g. "render a 6 s video"). Note `release_oneshot_encoders=True` by default — this is the most important flag in the design: **release the encoders' GPU memory the moment encoding is done**, freeing space for the transformer's AR steps.

2. **第 203-205 行 / Lines 203-205 (text encoder one-shot)**:
   - 中文: T5/UMT5 文本编码器在整个 rollout 里**只跑一次**,产出 `(B, L, D)` 的 text embeddings 缓存住。之后每一个 AR step 都直接复用这份 embedding —— 不会重复 token化、不会重复跑 encoder。
   - English: The T5 / UMT5 text encoder runs **exactly once** over the rollout, producing a `(B, L, D)` text-embedding cache. Every subsequent AR step reuses this — no re-tokenisation, no re-encoding.

3. **第 207-211 行 / Lines 207-211 (CFG 条件 negative prompt)**:
   - 中文: classifier-free guidance 需要正负两条 prompt 同时跑。**关键优化**:如果 `guidance_scale <= 1.0` 就压根不算 negative —— 省一次 text encoder 推理。这是接收端控制 inference cost 的一个小杠杆。
   - English: Classifier-free guidance needs both positive and negative prompt branches. **Key optimisation**: if `guidance_scale <= 1.0`, skip the negative entirely — saves one text-encoder pass. A small lever the server gives the caller for cost control.

4. **第 213-228 行 / Lines 213-228 (encoder/image 一致性检查 + 整段最关键的注释)**:
   - 中文: 这一段唯一做的"事情"是断言 —— `encoder` 存在 ⇔ image 存在;image 必须 T=1。但**最关键的是注释**:"The image is *not* VAE-encoded here: that happens per AR step inside the encoder so the streaming Wan VAE's temporal cache advances correctly."。这一句话浓缩了整个流式视频推理的设计哲学:**VAE 是有状态的,它的 temporal cache 必须随 AR step 一步一步推进**,放在 `initialize_cache` 里编码会让 cache 一开始就错位。
   - English: This block is just assertions — `encoder` exists ⇔ `image` exists; image must be T=1. **But the critical part is the comment**: "The image is *not* VAE-encoded here: that happens per AR step inside the encoder so the streaming Wan VAE's temporal cache advances correctly." That single sentence distils the entire streaming-video inference philosophy: **the VAE is stateful, its temporal cache must advance one AR step at a time**; encoding it here would misalign the cache from step 0.

5. **第 230-261 行 / Lines 230-261 (从 image 反推 latent 高宽)**:
   - 中文: 当用户传了 image 但没传 `height`/`width`,从 image 像素尺寸和 `decoder.spatial_compression_ratio` 反推 latent 大小。**这里的设计纪律**:decoder 拥有 pixel↔latent 的转换比,encoder 不掌权 —— 这样以后换 VAE 也只用改 decoder。如果用户同时传了 image 和 height/width,就交叉校验,不一致直接 assert error 而不是悄悄改写参数。
   - English: When the user passes `image` but not `height`/`width`, derive the latent size from pixel dims and `decoder.spatial_compression_ratio`. **Design discipline**: the decoder owns the pixel↔latent ratio, the encoder doesn't — so swapping VAEs later means changing only the decoder. If the user supplies both, cross-check; mismatch is an assert error, not a silent override.

6. **第 263-269 行 / Lines 263-269 (CLIP image embeddings)**:
   - 中文: 如果配置了 `image_encoder` (一般是 CLIP),就把 image 的 T=1 维度 squeeze 掉,跑一次 CLIP,产出全局图像 embedding。这一份会作为额外条件喂给 DiT —— I2V 模型靠它对齐"风格 / 主体"。
   - English: If an `image_encoder` (typically CLIP) is configured, squeeze the T=1 axis, run CLIP once, and stash a global image embedding. This is fed as an extra DiT conditioning signal — how I2V models lock onto "style / subject".

7. **第 271-279 行 / Lines 271-279 (super().initialize_cache)**:
   - 中文: 把所有 one-shot 算好的 conditioning 打包成 `transformer_context` dict,**交给父类**初始化更底层的 transformer / encoder / decoder cache。这是 FlashDreams 整个对象层级的接缝 —— 父类负责 streaming KV cache、Triton kernel cache 之类的硬件级缓存。
   - English: Bundle all one-shot conditioning into a `transformer_context` dict, **delegate to the parent class** to initialise lower-level transformer / encoder / decoder caches. This is the seam between FlashDreams' object layers — the parent owns streaming KV cache, Triton kernel caches, and similar hardware-level state.

8. **第 281-282 行 / Lines 281-282 (release_oneshot_encoders)**:
   - 中文: 默认 True 时,立即调用 `self.release_oneshot_encoders()` 把 text encoder、image encoder 的权重从显存释放 —— 一次 5B 的 T5 + 1B 的 CLIP 加起来约 12 GB 显存,释放后立刻能腾给 DiT 做更长的 AR rollout。
   - English: If True (default), immediately call `self.release_oneshot_encoders()` to free the text encoder + CLIP weights from VRAM — combined ~12 GB for a 5B T5 + 1B CLIP. That memory immediately becomes available for longer AR rollouts on the DiT.

9. **第 284-289 行 / Lines 284-289 (返回完整 cache)**:
   - 中文: 把所有子 cache 打包成一个 `WanInferencePipelineCache` 数据类返回。注意把 raw `image` 也存进去 —— 后面每个 AR step 都需要它去过流式 VAE encoder。**所以"image 没在 `initialize_cache` 里 encode"不是省略,是延迟到 `_preprocess_i2v_input` 里去做**。
   - English: Bundle all sub-caches into a `WanInferencePipelineCache` dataclass and return. **Note the raw `image` is stored on the cache** — every subsequent AR step needs it to push through the streaming VAE encoder. **So "image isn't VAE-encoded in `initialize_cache`" isn't an omission — it's deferred to `_preprocess_i2v_input` at each step.**

## 类比 / The analogy

中文: 把这套设计想成**演播室的录制前准备**。导演 (你的 prompt) 在录制前一次性把所有"静态资产"准备好 —— 剧本背完 (text encoder)、副剧本 (negative prompt) 备好、定妆照拍完 (CLIP image embedding) —— **然后剧本和定妆照印发完就把那些临时工 (encoder 们) 都打发回家了** (release_oneshot_encoders)。**但摄像机 (VAE encoder) 必须一直留在现场**,因为它要拍一段、转一段、再拍下一段地连续录制,中间不能断片 —— 它的"胶片传动机构" (temporal cache) 必须和实际拍摄节奏同步。`initialize_cache` 把这一切准备好之后,接下来 `generate` 每一步都只让摄像机走一格,效率才上得去。

English: Think of this as **pre-production for a studio shoot**. Before rolling, the director (your prompt) gets every "static asset" ready once — script memorised (text encoder), b-roll script prepped (negative prompt), headshots taken (CLIP image embedding) — **and then once those are printed and distributed, the day-hires (the encoders) are sent home** (release_oneshot_encoders). **But the camera (VAE encoder) stays on set the whole time**, because it has to roll continuously — record a clip, advance, record another — with no gap; its "film-transport mechanism" (temporal cache) must stay in sync with the live shoot. Once `initialize_cache` has set all this up, each `generate` step only advances the camera one frame — that's how throughput stays high.

## 自己跑一遍 / Try it yourself

```python
# minimal_oneshot_cache.py — model the "one-shot then release" pattern
import torch, torch.nn as nn, gc

class StubEncoder(nn.Module):
    """Pretend big encoder — eats memory until released."""
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.payload = nn.Parameter(torch.randn(1024, 1024))  # ~4 MB
    def forward(self, x): return x @ self.payload

class StreamingCache:
    def __init__(self, text_enc, image_enc):
        self.text_enc = text_enc
        self.image_enc = image_enc
        self.cache = {}
    def initialize(self, text_x, image_x, release=True):
        self.cache["text_emb"] = self.text_enc(text_x).detach()
        if image_x is not None:
            self.cache["img_emb"] = self.image_enc(image_x).detach()
        if release:
            self.text_enc = None
            self.image_enc = None
            gc.collect()
            print("[release] one-shot encoders freed")
        return self.cache

text_enc = StubEncoder("text")
img_enc = StubEncoder("image")
sc = StreamingCache(text_enc, img_enc)
cache = sc.initialize(torch.randn(2, 1024), torch.randn(2, 1024), release=True)
print("cache keys:", list(cache.keys()))
print("text_enc is None ->", sc.text_enc is None)
```

运行 / Run with:
```bash
pip install torch
python minimal_oneshot_cache.py
```

预期输出 / Expected output:
```
[release] one-shot encoders freed
cache keys: ['text_emb', 'img_emb']
text_enc is None -> True
```

中文: 一旦把 encoders 置 None 并 `gc.collect()`,它们占的显存就立刻还给 PyTorch CUDA caching allocator。在真实 FlashDreams 里 release 一次 T5 (5B) + CLIP (1B) 可以腾出 ~12 GB 显存,这让 transformer cache 能多撑两秒视频的 rollout。

English: Once you set the encoders to `None` and call `gc.collect()`, their VRAM is returned to PyTorch's CUDA caching allocator. In real FlashDreams, releasing T5 (5B) + CLIP (1B) frees ~12 GB, letting the transformer cache support roughly two extra seconds of video rollout.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/diffusers` pipelines 的 `enable_model_cpu_offload` flag** / **`huggingface/diffusers` `enable_model_cpu_offload`**: 同样思路 —— 用完的子模块 offload 到 CPU。FlashDreams 更激进,直接 release。/ Same idea — offload unused submodules to CPU. FlashDreams is more aggressive: outright release.
- **vLLM 的 prefix cache / prefill** / **vLLM prefix cache / prefill**: 跟"text encoder one-shot then never re-encode"是同一类思想,只是在 LLM 上。/ Same family as "text encoder one-shot then never re-encode", just on LLMs.
- **Wan2.x 自家的 `t2v` pipeline** / **Wan2.x's own `t2v` pipeline**: 是这份的"非流式"对照,可以看到没有 release / 没有 streaming VAE 时整体显存占用差多少。/ The "non-streaming" reference; lets you measure how much VRAM the no-release / no-streaming version eats by comparison.

## 注意事项 / Caveats / when it breaks

- **release 之后再调用 `initialize_cache` 会重新 load encoder / Re-calling `initialize_cache` after release re-loads encoders**:
  - 中文: `_ensure_oneshot_encoders_loaded` 会从 `self.config` 重新加载 encoder 权重 —— 这意味着**第二次 rollout 的第一次 `initialize_cache` 会有一次显著的 cold start 延迟** (几秒)。如果是 batch serving 场景应该考虑共享 cache 或不 release。
  - English: `_ensure_oneshot_encoders_loaded` reloads encoder weights from `self.config` — meaning **the first `initialize_cache` of every second rollout takes a noticeable cold-start hit** (seconds). For batched serving, share the cache or skip the release.
- **T=1 强制约束 / The T=1 image constraint**:
  - 中文: `assert image.shape[-4] == 1` 说明这套 I2V 只接受"单帧锚定" —— 多帧条件 (video-to-video) 需要换一条 pipeline。
  - English: `assert image.shape[-4] == 1` means this I2V path only accepts a single anchor frame — multi-frame conditioning (video-to-video) requires a different pipeline.
- **`spatial_compression_ratio` 取自 decoder / `spatial_compression_ratio` is read from the decoder**:
  - 中文: 注释里特别提到"encoder 假设与 decoder 共享比例 (Wan VAE 是这样)"。**如果你换非对称 VAE,这一行 derived_h / derived_w 就错了**。
  - English: The comment explicitly says "encoder is assumed to share the ratio (Wan VAE does)". **If you swap in an asymmetric VAE, this `derived_h / derived_w` calculation breaks.**

## 延伸阅读 / Further reading

- [FlashDreams README — NVIDIA's interactive AR video / world model serving library](https://github.com/NVIDIA/flashdreams)
- [Wan2.1 paper — "Wan: Open and Advanced Large-Scale Video Generation Models"](https://arxiv.org/abs/2503.20314)
- [HF diffusers CPU offload guide](https://huggingface.co/docs/diffusers/optimization/memory)
- [Classifier-free guidance original paper — Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)
