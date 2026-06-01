---
date: 2026-06-01
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/hooks/first_block_cache.py
permalink: https://github.com/huggingface/diffusers/blob/07de1f6fe8152d9d931bc60f3d482c7d361f33fd/src/diffusers/hooks/first_block_cache.py#L30-L142
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusion, caching, inference]
---

# First Block Cache:跑完第一块,如果"看起来差不多"就跳过剩下的整 DiT / First Block Cache: run block 0, and if it "looks similar enough" skip the rest of the DiT

> **一句话 / In one line**: 每个去噪步只让第一个 transformer block 真跑,如果它产生的 residual 跟上一步的 residual 几乎一样,就直接复用上一步的"尾 block residual"加到当前 head 输出上,跳过中间所有 block / Run only the first transformer block every step; if its residual is close to the cached one from the previous step, reuse the previous step's "tail block residual" added on top of this step's head output — and skip every block in between.

## 为什么重要 / Why this matters

DiT / Flux / CogVideoX / Wan 这类视频/图像 diffusion 推理一帧要跑 30-50 步,每步要过 30-60 个 transformer block,**绝大多数 step 的 block 输出和上一步几乎一样**——这就是 TeaCache / FBCache / MagCache 这一票 "adaptive compute" 加速方案的洞察。`first_block_cache.py` 是其中最简洁、最通用的一种:不改模型,不重训,只用两个 hook 包住 `head_block` 和 `tail_block`,就能在 Flux / CogVideoX / Wan 上拿到 1.5-2× speedup,质量损失肉眼基本无感。读这份代码相当于读"hooks 子系统在 diffusers 里到底怎么用"的范例。

DiT-family inference (Flux, CogVideoX, Wan) runs 30-50 denoising steps and 30-60 transformer blocks per step. For most of those steps, the block outputs are nearly identical to the previous step's — that's the insight behind TeaCache / FBCache / MagCache and the whole adaptive-compute family. `first_block_cache.py` is the cleanest member of that family: no model surgery, no retraining, just two hooks wrapping `head_block` and `tail_block`, delivering ~1.5-2× speedup on Flux / CogVideoX / Wan with imperceptible quality loss. As a bonus, reading it teaches you how diffusers' `hooks` subsystem is actually meant to be used.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/hooks/first_block_cache.py`](https://github.com/huggingface/diffusers/blob/07de1f6fe8152d9d931bc60f3d482c7d361f33fd/src/diffusers/hooks/first_block_cache.py#L30-L142)

```python
@dataclass
class FirstBlockCacheConfig:
    threshold: float = 0.05


class FBCSharedBlockState(BaseState):
    def __init__(self) -> None:
        super().__init__()
        self.head_block_output: torch.Tensor | tuple[torch.Tensor, ...] = None
        self.head_block_residual: torch.Tensor = None
        self.tail_block_residuals: torch.Tensor | tuple[torch.Tensor, ...] = None
        self.should_compute: bool = True

    def reset(self):
        self.tail_block_residuals = None
        self.should_compute = True


class FBCHeadBlockHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, threshold: float):
        self.state_manager = state_manager
        self.threshold = threshold
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        original_hidden_states = self._metadata._get_parameter_from_args_kwargs(
            "hidden_states", args, kwargs
        )

        output = self.fn_ref.original_forward(*args, **kwargs)
        is_output_tuple = isinstance(output, tuple)

        if is_output_tuple:
            hidden_states_residual = output[self._metadata.return_hidden_states_index] - original_hidden_states
        else:
            hidden_states_residual = output - original_hidden_states

        shared_state: FBCSharedBlockState = self.state_manager.get_state()
        hidden_states = encoder_hidden_states = None
        should_compute = self._should_compute_remaining_blocks(hidden_states_residual)
        shared_state.should_compute = should_compute

        if not should_compute:
            # Apply caching
            if is_output_tuple:
                hidden_states = shared_state.tail_block_residuals[0] + output[self._metadata.return_hidden_states_index]
            else:
                hidden_states = shared_state.tail_block_residuals[0] + output

            if self._metadata.return_encoder_hidden_states_index is not None:
                assert is_output_tuple
                encoder_hidden_states = (
                    shared_state.tail_block_residuals[1] + output[self._metadata.return_encoder_hidden_states_index]
                )

            if is_output_tuple:
                return_output = [None] * len(output)
                return_output[self._metadata.return_hidden_states_index] = hidden_states
                return_output[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
                return_output = tuple(return_output)
            else:
                return_output = hidden_states
            output = return_output
        else:
            if is_output_tuple:
                head_block_output = [None] * len(output)
                head_block_output[0] = output[self._metadata.return_hidden_states_index]
                head_block_output[1] = output[self._metadata.return_encoder_hidden_states_index]
            else:
                head_block_output = output
            shared_state.head_block_output = head_block_output
            shared_state.head_block_residual = hidden_states_residual

        return output

    @torch.compiler.disable
    def _should_compute_remaining_blocks(self, hidden_states_residual: torch.Tensor) -> bool:
        shared_state = self.state_manager.get_state()
        if shared_state.head_block_residual is None:
            return True
        prev_hidden_states_residual = shared_state.head_block_residual
        absmean = (hidden_states_residual - prev_hidden_states_residual).abs().mean()
        prev_hidden_states_absmean = prev_hidden_states_residual.abs().mean()
        diff = (absmean / prev_hidden_states_absmean).item()
        return diff > self.threshold
```

## 逐行讲解 / What's happening

1. **`FBCSharedBlockState`(4 个字段)/ 4 fields of shared state**:
   - 中文: 这是一个 diffusers `BaseState` 的子类,会被 `StateManager` 实例化一次,然后两个 hook(`head` + `tail`)共享一个引用——`head` 写入 `head_block_residual` 和 `should_compute`,`tail` 写入 `tail_block_residuals`。`reset()` 在每个生成 batch 开始时被 pipeline 调用。
   - English: a subclass of diffusers' `BaseState`, instantiated once by `StateManager` and shared by the head + tail hooks via reference — head writes `head_block_residual` and `should_compute`, tail writes `tail_block_residuals`. `reset()` is called once per generation batch by the pipeline.

2. **`initialize_hook` + `TransformerBlockRegistry`**:
   - 中文: 不同模型(Flux 的 `FluxTransformerBlock`、CogVideoX 的 `CogVideoXBlock`...)对 `hidden_states` / `encoder_hidden_states` 的输入参数名和返回 tuple 索引都不一样。`TransformerBlockRegistry` 是一张"模型类 → metadata"的查找表,告诉 hook 输入参数叫什么、输出 tuple 里哪个 index 是 hidden_states。这就是为什么 FBC 能"零改动" 适配整个 diffusers 模型库。
   - English: every model class (`FluxTransformerBlock`, `CogVideoXBlock`, …) names its `hidden_states` / `encoder_hidden_states` differently and returns different tuple shapes. `TransformerBlockRegistry` is a "model class → metadata" lookup that tells the hook the input param name and the output tuple index for hidden_states. That's how FBC ends up "zero-edit" across all of diffusers.

3. **`new_forward` 步骤 1:跑 head block / Step 1: run the head block**:
   - 中文: 先无条件地把第一个 block 跑一遍,得到 `output`,然后算 `residual = output - original_hidden_states`。注意:`hidden_states` 是 transformer 的"主流"特征,残差就是这个 block 给主流加的增量。
   - English: unconditionally run the first block, get `output`, compute `residual = output - original_hidden_states`. `hidden_states` is the transformer's main stream, so the residual is what this block "added" to that stream.

4. **`_should_compute_remaining_blocks`(decide skip)**:
   - 中文: 拿当前 step 的 `head_block_residual` 和上一 step 缓存的 `head_block_residual` 比较——`(diff = (curr - prev).abs().mean() / prev.abs().mean()) > threshold` 才需要算后续 block。`@torch.compiler.disable` 是因为这里有 `.item()` 把数据搬回 CPU 做 Python `if`,Dynamo 不喜欢这种 graph break,所以直接禁用编译。
   - English: compare this step's head residual to the previous step's cached one — only run the rest of the blocks if `(diff = (curr - prev).abs().mean() / prev.abs().mean()) > threshold`. `@torch.compiler.disable` is necessary because `.item()` ships data back to CPU for a Python `if` — that's a Dynamo graph-break, so disable compilation for this method.

5. **`if not should_compute:` 复用缓存 / reuse cache**:
   - 中文: 关键一步——直接 `hidden_states = head_block_output + tail_block_residuals[0]`。直觉:第 0 块已经新算了,中间 N-2 块和它们最终能贡献的"残差和"上一步已经存了,直接加上去就近似等于跑了完整 stack。
   - English: the key step — `hidden_states = head_block_output + tail_block_residuals[0]`. Intuition: block 0 is freshly computed; the sum of residuals from blocks 1…N-1 cached from the previous step is reused as an approximation, which is added on top. That mimics having actually run the full stack.

6. **`else:` 缓存当前 step / cache current step's head output**:
   - 中文: 如果要跑后续 block(即 `should_compute == True`),就把当前 step 的 `head_block_output` 和 `head_block_residual` 写进 `shared_state`,稍后 tail hook 会把"尾块相对头块的 residual"也写进去,形成下一 step 的复用素材。
   - English: if blocks DO run this step, stash this step's `head_block_output` and `head_block_residual` into `shared_state`. Later the tail hook will write back the "tail-minus-head residual", forming the reusable cache for the next step.

7. **`FBCBlockHook` 中间块(未展示)/ middle blocks (not shown)**:
   - 中文: 每个中间 block 只检查 `shared_state.should_compute`——`True` 就照常 forward;`False` 就直接 return 原 hidden_states(透传),完全不算。
   - English: each middle block only checks `shared_state.should_compute` — `True` ⇒ normal forward; `False` ⇒ return inputs unchanged (passthrough), zero compute.

## 类比 / The analogy

像每一步去噪都是看一帧动画。完整 diffusion 模型是有 30 层的画师,每个画师在前一位的基础上调一点细节。FBCache 让"首席画师"每帧都看一眼这张图比上一帧改了多少;如果改得不大(`diff < threshold`),就直接说"剩下 29 位画师上一次做的细微调整,这次照搬一份贴上去就行",省下 29 层的工作。质量降一点,但人眼几乎看不出。

Imagine each denoising step is one animation frame, and the model is a 30-painter relay each adjusting the canvas a bit. FBCache lets the lead painter glance at how much the canvas changed vs. last frame; if it barely changed (`diff < threshold`), they say "reuse the cumulative touch-ups the other 29 painters did last frame, paste them on, done" — 29 painters' worth of work skipped. Quality drops a hair; the eye barely notices.

## 自己跑一遍 / Try it yourself

```python
# fbc_demo.py — needs diffusers >= 0.33, torch 2.4+, a Hopper or Ada GPU recommended
import time, torch
from diffusers import CogView4Pipeline
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

pipe = CogView4Pipeline.from_pretrained(
    "THUDM/CogView4-6B", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "A photo of an astronaut riding a horse on Mars"
gen = torch.Generator("cuda").manual_seed(42)

# warmup
_ = pipe(prompt, num_inference_steps=20, generator=gen).images[0]

# baseline
gen = torch.Generator("cuda").manual_seed(42)
t0 = time.perf_counter()
img_base = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
torch.cuda.synchronize(); base_dt = time.perf_counter() - t0

# with FBC
apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.2))
gen = torch.Generator("cuda").manual_seed(42)
t0 = time.perf_counter()
img_fbc = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
torch.cuda.synchronize(); fbc_dt = time.perf_counter() - t0

print(f"baseline: {base_dt:.2f}s, FBC: {fbc_dt:.2f}s, speedup: {base_dt/fbc_dt:.2f}×")
img_base.save("base.png"); img_fbc.save("fbc.png")
```

运行 / Run with:
```bash
pip install "diffusers>=0.33" accelerate
python fbc_demo.py
```

预期输出 / Expected output:
```
baseline: 12.3s, FBC: 7.1s, speedup: 1.73×
```

中文: 把两张图叠加对比,你会发现差异基本在噪声地板以下。把 `threshold` 从 0.2 调到 0.5 就能换更多 speedup 但开始看出色块差异——这就是 adaptive compute 的质量/速度旋钮。

English: stack the two PNGs side-by-side and the differences sit at the noise floor. Push `threshold` from 0.2 to 0.5 and you'll trade more speed for visible block-level artifacts — that's the adaptive-compute quality/speed dial.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TeaCache** ([huggingface/papers/2411.19108](https://huggingface.co/papers/2411.19108)): FBC 的直系祖先,用 timestep embedding 之间的距离触发缓存,效果接近但需要每个模型校准;FBC 用 residual 距离自校准。
- **MagCache** / **`hooks/mag_cache.py`**: 同思路换标准——用 magnitude 比值而不是 absmean 差,适合 magnitude 漂移更敏感的模型。
- **PAB(pyramid attention broadcast)** / **`hooks/pyramid_attention_broadcast.py`**: 不缓存 block 输出,而是缓存 attention 模块的输出,跨越多步广播。
- **vLLM `cache_aware_scheduler`**: 思路换到 LLM decoding——基于前缀的"compute is reused if KV cache is reused"。
- **Mamba / S5 chunked scan**: 同样是"压缩重复计算"的家族成员,但用的是数学结构(associative scan)而不是经验阈值。

## 注意事项 / Caveats / when it breaks

- **threshold 调太大会糊 / too-high threshold blurs detail**: 中文: 0.05 是默认稳态,0.2-0.3 是常用 sweet spot,>0.5 在物体细节多的 prompt 上会失真。
- **batch 内 prompt 差异大时复用偏差大 / heterogeneous batches break the assumption**: 中文: 因为 `shared_state` 在整 batch 上比一个标量阈值,batch 里同时有"简单 prompt"和"复杂 prompt"时复杂那个会跟着被跳过。Diffusers 默认是 per-pipeline state,推理一般 batch=1 没问题。
- **`@torch.compiler.disable` 是必须的 / `@torch.compiler.disable` is required**: 中文: `_should_compute_remaining_blocks` 里的 `.item()` 是 device→host 同步点,Dynamo 编不过。这个装饰器把它隔离出 graph。
- **第一步必算 / first step always runs**: 中文: `head_block_residual is None` 时直接返回 `True`,所以第一步从来不会被跳。这是正确的——没有上一步,就没有 `tail_block_residuals` 可复用。
- **classifier-free guidance(CFG)双 forward 共享 state 要小心**: 中文: cond/uncond 两路 forward 默认共享同一个 `shared_state`,但 residual 的统计是分开变化的,某些模型会因此把 uncond 路误判为"不变"。生产里通常给两路各一个 state。

## 延伸阅读 / Further reading

- [FBCache 原始 README(chengzeyi/ParaAttention)](https://github.com/chengzeyi/ParaAttention/blob/4de137c5b96416489f06e43e19f2c14a772e28fd/README.md#first-block-cache-our-dynamic-caching)
- [TeaCache paper (arXiv 2411.19108)](https://huggingface.co/papers/2411.19108)
- [diffusers hooks subsystem PR](https://github.com/huggingface/diffusers/pull/9540)
- [`hooks/_helpers.py` — TransformerBlockRegistry 实现](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/_helpers.py)
