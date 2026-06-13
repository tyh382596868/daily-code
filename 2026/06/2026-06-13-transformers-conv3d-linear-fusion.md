---
date: 2026-06-13
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/fusion_mapping.py
permalink: https://github.com/huggingface/transformers/blob/08a7ef05bcf9723cb2e58855afb8dc2c799323ff/src/transformers/fusion_mapping.py#L71-L187
difficulty: advanced
read_time: ~11 min
tags: [code-of-the-day, huggingface, transformers, fusion, patch-embedding, weight-conversion]
---

# 一句数学恒等式 = 加载期外科手术:Transformers 把 Conv3d patch-embed 在 load 时换成 Linear / A math identity becomes a load-time surgery: Transformers swaps Conv3d patch-embed for Linear at checkpoint load

> **一句话 / In one line**: 当 Conv3d 满足 `stride == kernel`、`padding == 0`、`dilation == 1`、`groups == 1` 时,它就和"把 patch 摊平再过一个 Linear"完全等价 —— 新的 `fusion_mapping.py` 把这条数学事实做成了一个 117 行的 spec:发现匹配的模块、调包新的 Mixin、再注册一个 `Conv3dToLinear` 的 WeightConverter,让原始 checkpoint 还能直接 load 进 fused 后的网络。/ When a `Conv3d` is `stride == kernel`, `padding == 0`, `dilation == 1`, `groups == 1`, it is *mathematically* identical to "flatten the patch and run a Linear". The new `fusion_mapping.py` codifies that fact in 117 lines: discover matching modules, swap the runtime class with a mixin, and register a `Conv3dToLinear` WeightConverter so the original checkpoint loads cleanly into the fused network.

## 为什么重要 / Why this matters

视频 transformer 里第一步几乎都是一个 Conv3d patch-embedding —— 但运行时这层 Conv3d 在 GPU 上其实是个小 kernel,跑得慢、launch 开销大,而且大多数视频模型 patch 又是不重叠的 (stride == kernel)。这种情况下其实写成一个 `Linear(patch_volume → embed_dim)` 性能更好、Triton/Inductor 更容易融合。HF transformers 这次的做法不是手改每个模型,而是**做了一个统一的 fusion 注册机制**:把"什么样的模块可以 fuse、fuse 之后的 forward 是什么、原 checkpoint 怎么映射到新权重"三件事打包成一个 `ModuleFusionSpec`。这是模型生产线里"load-time 重写"的范式级例子。

The first layer of nearly every video transformer is a `Conv3d` patch embedding — but at runtime that `Conv3d` is a tiny kernel on the GPU: low compute density, expensive launch, and almost always run with stride equal to kernel (non-overlapping patches). In that regime, rewriting it as a `Linear(patch_volume → embed_dim)` is faster and easier for Triton / Inductor to fuse. HF transformers' move here isn't to hand-edit every model — it's to **build a unified fusion registry**: "what modules are fusable, what the fused forward looks like, how the original checkpoint maps onto the new weight layout" all bundled into one `ModuleFusionSpec`. A textbook example of load-time architectural rewrites in production.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/fusion_mapping.py`](https://github.com/huggingface/transformers/blob/08a7ef05bcf9723cb2e58855afb8dc2c799323ff/src/transformers/fusion_mapping.py#L71-L187)

```python
class _FusedPatchEmbeddingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # run original_cls.__init__
        self.patch_volume = self.proj.in_channels * math.prod(self.proj.kernel_size)

        self.linear_proj = nn.Linear(
            self.patch_volume,
            self.proj.out_channels,
            bias=self.proj.bias is not None,
            device=self.proj.weight.device,
            dtype=self.proj.weight.dtype,
        )
        del self.proj                                # the Conv3d is gone

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.linear_proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.patch_volume)
        hidden_states = self.linear_proj(hidden_states.to(dtype=target_dtype))
        return hidden_states.view(-1, self.embed_dim)


class PatchEmbeddingsFusionSpec(ModuleFusionSpec):
    """Fuse compatible Conv3d patch embeddings into flattened Linear projections."""

    target_modules_patterns = (r"(^|\.)patch_embed$",)

    def is_fusable(self, module: nn.Module) -> bool:
        if not isinstance(proj := getattr(module, "proj", None), nn.Conv3d):
            return False
        # no overlap between the patches
        return (
            proj.stride == proj.kernel_size
            and proj.padding == (0, 0, 0)
            and proj.dilation == (1, 1, 1)
            and proj.groups == 1
        )

    def make_fused_class(self, original_cls: type[nn.Module]) -> type[nn.Module]:
        fused_cls = type(
            f"Fused{original_cls.__name__}",
            (_FusedPatchEmbeddingMixin, original_cls),
            {},
        )
        fused_cls.__qualname__ = f"Fused{original_cls.__qualname__}"
        return fused_cls

    def make_transforms(self, config: "PretrainedConfig") -> list[WeightTransform]:
        vision_config = getattr(config, "vision_config", config)
        patch_size = vision_config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        kernel_size = (vision_config.temporal_patch_size, *tuple(patch_size))
        in_channels = vision_config.in_channels

        return [
            WeightConverter(
                source_patterns=r"patch_embed\.proj\.weight$",
                target_patterns=r"patch_embed\.linear_proj\.weight$",
                operations=[
                    Conv3dToLinear(in_channels=in_channels, kernel_size=kernel_size)
                ],
            ),
            WeightRenaming(
                source_patterns=r"patch_embed\.proj\.bias$",
                target_patterns=r"patch_embed\.linear_proj\.bias$",
            ),
        ]


def _discover_fusable_modules(cls, config, fusion_name, spec):
    """Discover compatible module classes for one fusion family on a meta-init model."""
    cache = _FUSION_DISCOVERY_CACHE.setdefault(fusion_name, {})
    if cls in cache:
        return cache[cls]

    with torch.device("meta"):
        model = cls(config)                          # zero-mem instantiation

    seen_classes = set()
    patch_mapping = {}
    target_module_pattern = (
        re.compile("|".join(spec.target_modules_patterns))
        if spec.target_modules_patterns else None
    )
    for module_name, module in model.named_modules():
        module_cls = type(module)
        if module_cls in seen_classes:
            continue
        if target_module_pattern is not None and target_module_pattern.search(module_name) is None:
            continue
        if not spec.is_fusable(module):
            continue
        seen_classes.add(module_cls)
        patch_mapping[module_cls.__name__] = spec.make_fused_class(module_cls)

    cache[cls] = patch_mapping
    return patch_mapping
```

## 逐行讲解 / What's happening

1. **第 71-91 行 / Lines 71-91 (`_FusedPatchEmbeddingMixin`)**:
   - 中文: 这是个 mixin —— 通过 `super().__init__(*args, **kwargs)` 先把原始类的 `__init__` 跑一遍,让 `self.proj` (那个 Conv3d) 被建出来,然后**算出 `patch_volume = in_channels × ∏ kernel_size`**,新建一个**形状对齐**的 `nn.Linear(patch_volume, out_channels)`,最后 `del self.proj` —— Conv3d 在内存中正式下线。`forward` 干的事就是把输入 view 成 `(N, patch_volume)`、过 Linear、再 view 回 `(N, embed_dim)`。**这一段就是数学恒等式的实现**。
   - English: This is a mixin — `super().__init__(*args, **kwargs)` first runs the original class's `__init__` so `self.proj` (the `Conv3d`) gets built, then it computes `patch_volume = in_channels × ∏ kernel_size`, allocates a **shape-matched** `nn.Linear(patch_volume, out_channels)`, and `del self.proj` retires the Conv3d. `forward` views the input to `(N, patch_volume)`, runs the Linear, then views back to `(N, embed_dim)`. **This block is the math identity, made concrete.**

2. **第 94-97 行 / Lines 94-97 (`target_modules_patterns`)**:
   - 中文: 一个正则元组,只匹配模块名以 `.patch_embed` 结尾的 `nn.Module`。这是**预筛选** —— 在跑昂贵的 `is_fusable` 之前先剔除明显不相关的模块,在大型多模态模型 (几百个子模块) 上能省一两个数量级时间。
   - English: A tuple of regexes restricting matches to modules whose path ends in `.patch_embed`. This is a **prefilter** — runs before the expensive `is_fusable` check, saving an order of magnitude or two on large multimodal models with hundreds of submodules.

3. **第 99-109 行 / Lines 99-109 (`is_fusable`)**:
   - 中文: 这就是开篇说的那条数学条件 —— `stride == kernel`、`padding == 0`、`dilation == 1`、`groups == 1`。**满足这四条,Conv3d 的滑窗就不重叠,等价于"取每个 (kt, kh, kw) 块、摊平、过 Linear"**。注意用了 walrus operator (`proj := getattr(...)`) 一行内完成"取出 + 类型检查"。
   - English: This is the math condition from the intro — `stride == kernel`, `padding == 0`, `dilation == 1`, `groups == 1`. **Satisfy these four and the Conv3d's sliding window is non-overlapping; equivalent to "take each `(kt, kh, kw)` block, flatten, run a Linear".** Note the walrus operator (`proj := getattr(...)`) packs extraction + isinstance into one line.

4. **第 111-118 行 / Lines 111-118 (`make_fused_class`)**:
   - 中文: **动态创建一个新类**,继承自 `_FusedPatchEmbeddingMixin` 和原始类 —— MRO 让 Mixin 的 `__init__` 和 `forward` 优先生效,但其他方法 (比如 norm) 还是从原始类继承。这就是为什么 mixin 而不是直接替换 —— 保留了原始类里其它行为。
   - English: **Dynamically synthesises a new class** inheriting from `_FusedPatchEmbeddingMixin` *and* the original class. The MRO makes the mixin's `__init__` / `forward` win, while everything else (norms, helpers) is still inherited from the original. That's the reason for a mixin instead of a plain replacement — keep the rest of the original class's behaviour intact.

5. **第 116-139 行 / Lines 116-139 (`make_transforms`)**:
   - 中文: 这里返回**两个 WeightTransform**,负责把原 checkpoint 里的 `patch_embed.proj.weight` (Conv3d 形状 `(C_out, C_in, kt, kh, kw)`) 通过 `Conv3dToLinear` 操作变形成 `(C_out, patch_volume)` 的 Linear 权重,bias 直接改名。这是整个 fusion 不破坏 checkpoint 的关键 —— Hugging Face 上已经训好的视频模型 (Wan、Cosmos 等) 不用重训就能直接 load 进 fused 网络。
   - English: Returns **two WeightTransforms** that reshape the original checkpoint's `patch_embed.proj.weight` (Conv3d shape `(C_out, C_in, kt, kh, kw)`) into `(C_out, patch_volume)` Linear weight via `Conv3dToLinear`, with the bias just renamed. This is what makes the whole fusion non-breaking — every pretrained video model on the Hub (Wan, Cosmos, …) loads cleanly into the fused network without retraining.

6. **第 142-187 行 / Lines 142-187 (`_discover_fusable_modules`)**:
   - 中文: 整个 fusion 的"自动发现"逻辑。三个关键招式:(a) **`with torch.device("meta")`** —— 在 meta 设备上实例化模型,**不分配任何真实显存**;(b) 跑一遍 `model.named_modules()` 收集所有可 fuse 的模块**类** (不是实例);(c) 每个类只处理一次 (`seen_classes`),并把"原类 → fused 类"的映射缓存到 `_FUSION_DISCOVERY_CACHE`,后续相同 `(fusion_name, cls)` 直接复用。
   - English: The auto-discovery logic. Three key tricks: (a) **`with torch.device("meta")`** instantiates the model on the meta device — **zero real memory allocated**; (b) walks `model.named_modules()` to collect every fusable module **class** (not instance); (c) dedup per-class via `seen_classes` and cache the `{original → fused}` mapping in `_FUSION_DISCOVERY_CACHE`, so repeated `(fusion_name, cls)` lookups skip the work.

## 类比 / The analogy

中文: 想象你买了一台**老式胶片相机**和一个全自动数码后背。Conv3d 就像那台胶片相机:能拍,但 launch 一次曝光的延迟很大,且大多数你的拍摄需求 (stride == kernel) 其实根本用不到它的 overlap 能力。`PatchEmbeddingsFusionSpec` 就是那个**数码后背改装套件**:把后背装上去 (`make_fused_class`),交付前先确认你的镜头座规格匹配 (`is_fusable` 检查那四个条件),然后**把过去拍好的所有胶片底片也"洗成"数码格式** (`Conv3dToLinear` 的 WeightConverter)。改装完你拍照的工作流不变,但快门延迟少了一大截,而且过去 10 年的照片库一张没丢。

English: Picture you bought an **old film camera body** and a fully-automatic digital back. The `Conv3d` is the film body — it works, but each "exposure" launches with painful latency, and most of your shoots (stride == kernel) don't actually need its overlap capability. `PatchEmbeddingsFusionSpec` is the **digital-back retrofit kit**: bolt the back on (`make_fused_class`), but first confirm your lens mount matches (`is_fusable` checks the four conditions), and *also* **convert every roll of film you've ever shot into the new digital format** (the `Conv3dToLinear` WeightConverter). Workflow is unchanged after the retrofit, shutter latency drops dramatically, and ten years of archived shots come along for free.

## 自己跑一遍 / Try it yourself

```python
# fuse_conv3d_to_linear.py — prove the math identity
import torch
import torch.nn as nn

C_in, C_out = 3, 16
kt, kh, kw = 2, 4, 4
B, T, H, W = 2, 4, 8, 8

conv = nn.Conv3d(C_in, C_out, kernel_size=(kt, kh, kw), stride=(kt, kh, kw), bias=False)

patch_volume = C_in * kt * kh * kw
linear = nn.Linear(patch_volume, C_out, bias=False)
linear.weight.data = conv.weight.data.reshape(C_out, patch_volume).clone()

x = torch.randn(B, C_in, T, H, W)

# Conv3d path: stride==kernel, so output is (B, C_out, T/kt, H/kh, W/kw)
y_conv = conv(x).permute(0, 2, 3, 4, 1).reshape(-1, C_out)

# Linear path: reshape patches by hand
x_patches = x.unfold(2, kt, kt).unfold(3, kh, kh).unfold(4, kw, kw)
x_patches = x_patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, patch_volume)
y_lin = linear(x_patches)

print("max abs diff:", (y_conv - y_lin).abs().max().item())  # ~0
```

运行 / Run with:
```bash
pip install torch
python fuse_conv3d_to_linear.py
```

预期输出 / Expected output:
```
max abs diff: 0.0
```

中文: 注意 `linear.weight = conv.weight.reshape(C_out, patch_volume).clone()` 这一步 —— **没有任何数值改动**,只是 reshape。`Conv3dToLinear` 在 HF 源码里做的也是这件事。如果你打破任何一个 fuse 条件 (比如把 stride 改成 `(1, kh, kw)`),这个等价就立刻破掉,你会看到 `max abs diff` 变成 ~10^0 数量级。

English: Notice the line `linear.weight = conv.weight.reshape(C_out, patch_volume).clone()` — **no numerical change**, just a reshape. `Conv3dToLinear` does the same thing in the real HF code. Break any of the fuse conditions (e.g. drop stride to `(1, kh, kw)`) and the equivalence is gone — `max abs diff` will jump to order 10⁰.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenAI Triton tutorials** / **OpenAI Triton tutorials**: 经常出现的"Conv with stride == kernel == im2col + GEMM"等价,Triton block-pointer 算子里直接是 Linear-shaped 访存。/ Triton repeatedly shows the same "Conv with stride == kernel is im2col + GEMM" equivalence — block-pointer kernels write the access pattern as a linear straight-through.
- **PyTorch `torch._inductor` patch-embed fusion pass** / **PyTorch `torch._inductor` patch-embed fusion pass**: 早期就有类似的 conv → mm 重写,但需要 trace-time 触发,而 HF 的版本是在 load 时静态完成。/ Earlier conv → matmul rewrites exist but only at trace time; HF's version does it statically at load.
- **diffusers VAE optimization** / **diffusers VAE optimization**: Stable Diffusion 的 VAE 里有个类似的 `image_to_patches` 替换,把 conv 换成 view+linear 走得更快。/ Stable Diffusion's VAE has a sibling `image_to_patches` swap that replaces a conv with view + linear for speed.

## 注意事项 / Caveats / when it breaks

- **数值差不是零,精度上有一点点漂移 / Numerics aren't strictly zero**:
  - 中文: 数学等价是浮点等价,但 Conv3d 和 Linear 走的可能是不同的 cuBLAS / cuDNN kernel,**累加顺序不同**会产生 ULP 级别的差异。生产 fine-tune 时这种差异可能反映在 loss 第 5、6 位小数上,通常不要紧但要心里有数。
  - English: Mathematical equivalence is floating-point equivalence — Conv3d and Linear may take different cuBLAS / cuDNN kernels with **different accumulation orders**, producing ULP-level drift. In production fine-tuning that drift may show up in the 5th-6th decimal of the loss; usually harmless but worth knowing.
- **`make_transforms` 假设 config 命名 / `make_transforms` hardcodes config field names**:
  - 中文: `vision_config.patch_size` / `temporal_patch_size` / `in_channels` 是名字写死的。如果某个模型用了 `time_patch_size` 之类的别名,需要扩展这个 spec 或写一个子类。
  - English: `vision_config.patch_size` / `temporal_patch_size` / `in_channels` are hard-coded names. A model that uses `time_patch_size` or similar aliases needs a subclass spec.
- **第一次 load 比平时慢一点 / First-time load is slightly slower**:
  - 中文: `_discover_fusable_modules` 要在 meta 设备上完整跑一遍 `cls(config)`,大模型上几百毫秒。但有 `_FUSION_DISCOVERY_CACHE`,同一个 cls 后续 load 都是 cache hit。
  - English: `_discover_fusable_modules` runs a full `cls(config)` on the meta device — hundreds of ms for big models. The `_FUSION_DISCOVERY_CACHE` makes subsequent loads of the same class free.

## 延伸阅读 / Further reading

- [HF PR introducing fusion mapping — #45041](https://github.com/huggingface/transformers/pull/45041)
- [PyTorch meta device docs — "Lazy initialization"](https://pytorch.org/docs/stable/meta.html)
- [WeightConverter overview in HF transformers `core_model_loading.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/core_model_loading.py)
- [Vision Transformer paper — "An Image is Worth 16×16 Words"](https://arxiv.org/abs/2010.11929)
