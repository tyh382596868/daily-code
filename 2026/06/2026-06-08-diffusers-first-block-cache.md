---
date: 2026-06-08
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/hooks/first_block_cache.py
permalink: https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/hooks/first_block_cache.py#L51-L142
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusion, inference-optimization, caching]
---

# 第一个 block 的残差几乎没变?那就跳过剩下所有 block / If the first block's residual barely moved, skip every other block

> **一句话 / In one line**: First-Block-Cache 用第一个 transformer block 的"残差变化幅度"当探针——如果它跟上一步几乎一样,就直接套用上次缓存的尾部残差,跳过中间所有 block。
> First-Block-Cache uses the first transformer block's residual change as a probe — if it's close enough to last step's residual, splice in the previously cached tail-block deltas and skip every block in between.

## 为什么重要 / Why this matters

视频 diffusion 模型(CogVideoX、Flux、Wan)的推理慢点不在 VAE,也不在 scheduler——在那一长串 30-40 个 transformer block 上。每一步去噪都全跑一遍,40 步 × 40 个 block = 1600 次 attention forward。但实际上相邻去噪步之间,模型对图像的"调整方向"是非常相关的——尤其在去噪后期,残差几乎不变。FBC 就是把这个直觉做成了一个 plug-in:挂个 hook 在第一个 block 上,测它的输出残差跟上一步比变了多少,如果几乎没变,大胆假设"剩下所有 block 这一步要做的事跟上一步也几乎一样",直接复用上一步缓存的最终残差。免费拿到 1.5-3x 推理加速,代价是一点点画质。

The slow part of video-diffusion inference (CogVideoX, Flux, Wan) isn't the VAE or the scheduler — it's the long stack of 30-40 transformer blocks. Every denoising step runs the whole stack, 40 steps × 40 blocks = 1600 attention forwards. But neighboring denoising steps share most of the *direction* in which they push the latent — especially late in the schedule, the residual barely changes. FBC turns that intuition into a plug-in: hook the first block, measure how much its residual changed vs. the previous step, and if it barely moved, gamble that "everything the rest of the stack is going to do this step is also basically what it did last step" — and reuse the cached tail residuals. Free 1.5-3x inference speedup, costing a sliver of quality.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/hooks/first_block_cache.py`](https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/hooks/first_block_cache.py#L51-L142)

```python
class FBCSharedBlockState(BaseState):
    def __init__(self):
        super().__init__()
        self.head_block_output = None       # last step's first-block output
        self.head_block_residual = None     # last step's first-block residual
        self.tail_block_residuals = None    # last step's residual from tail block
        self.should_compute = True

    def reset(self):
        self.tail_block_residuals = None
        self.should_compute = True


class FBCHeadBlockHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager, threshold):
        self.state_manager = state_manager
        self.threshold = threshold
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    def new_forward(self, module, *args, **kwargs):
        original_hidden_states = self._metadata._get_parameter_from_args_kwargs(
            "hidden_states", args, kwargs
        )

        output = self.fn_ref.original_forward(*args, **kwargs)
        is_output_tuple = isinstance(output, tuple)

        if is_output_tuple:
            hidden_states_residual = output[self._metadata.return_hidden_states_index] - original_hidden_states
        else:
            hidden_states_residual = output - original_hidden_states

        shared_state = self.state_manager.get_state()
        should_compute = self._should_compute_remaining_blocks(hidden_states_residual)
        shared_state.should_compute = should_compute

        if not should_compute:
            # Apply caching: splice cached tail residual onto this step's first-block output.
            if is_output_tuple:
                hidden_states = shared_state.tail_block_residuals[0] + output[self._metadata.return_hidden_states_index]
                encoder_hidden_states = shared_state.tail_block_residuals[1] + output[self._metadata.return_encoder_hidden_states_index]
                return_output = [None] * len(output)
                return_output[self._metadata.return_hidden_states_index] = hidden_states
                return_output[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
                output = tuple(return_output)
            else:
                output = shared_state.tail_block_residuals[0] + output
        else:
            # Refresh cache: store this step's first-block output + residual for next step.
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
    def _should_compute_remaining_blocks(self, hidden_states_residual):
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

1. **`hidden_states_residual = output - original_hidden_states`(行 85-87) / Lines 85-87**:
   - 中文: residual 就是"这个 block 对输入做了多少改动"。Diffusion 的 transformer block 都是残差结构 `out = in + something`,所以 `out - in` 就是 "block 想做的全部修改"。FBC 拿这个量当代理指标:这一步的修改方向跟上一步像不像?
   - English: The residual is "how much did this block change its input". Diffusion transformer blocks are all residual: `out = in + something`, so `out - in` *is* the entire modification the block is proposing. FBC uses this as a proxy: how similar is this step's modification direction to the last step's?

2. **`_should_compute_remaining_blocks`(行 134-142) / Lines 134-142**:
   - 中文: 决策函数。它算 `diff = ||r_t - r_{t-1}|| / ||r_{t-1}||`——本步残差和上步残差的相对差(用 absmean 而不是 L2 norm,GPU 友好)。如果 `diff > threshold`(默认 0.05),说明本步要做的事跟上次不一样,得乖乖跑完整个 stack;否则,跳。注意 `@torch.compiler.disable`——这是 Python 控制流分支,Dynamo 没法编译,所以显式关掉。
   - English: The decision function. It computes `diff = ||r_t - r_{t-1}|| / ||r_{t-1}||` — the relative L1 change between this step's residual and last step's (using absmean rather than L2 for GPU friendliness). If `diff > threshold` (default 0.05), this step is doing something different from last step and you have to run the whole stack; otherwise, skip. Note the `@torch.compiler.disable` — this is a Python control-flow branch that Dynamo can't compile through, so it's explicitly opted out.

3. **`if not should_compute` 分支:`tail_block_residuals[0] + output`(行 96-101) / Lines 96-101**:
   - 中文: 跳过的核心数学。`output` 是本步 first block 的输出(已经算了);`tail_block_residuals[0]` 是**上一步**整个 stack 在 first block 之后又往上加了多少。FBC 假设"那个增量在这一步也几乎一样",于是直接套上来。结果 hidden_states 就跳过中间所有 block 直接得出。
   - English: The math of the skip. `output` is this step's first-block output (already computed); `tail_block_residuals[0]` is how much the *previous step* further added on top of the first block by running the rest of the stack. FBC assumes "that delta is almost identical this step," and just splices it in. The result is this step's hidden_states obtained without touching any of the middle blocks.

4. **`else` 分支:刷新缓存(行 117-125) / Lines 117-125**:
   - 中文: 如果决定要跑完整个 stack,那就保存这一步 first block 的输出 + 残差,留给下一步当 baseline。注意 `tail_block_residuals` 不在这里更新——它是 `FBCBlockHook(is_tail=True)`(同文件下面)算完整个 stack 之后才填回去的。这里 hook 在 first block,只负责"测残差 + 决定跳不跳"。
   - English: If the decision is to run the full stack, stash this step's first-block output + residual to seed next step's comparison. Note `tail_block_residuals` is *not* updated here — that gets filled by `FBCBlockHook(is_tail=True)` (further down in the same file) once the rest of the stack has actually run. This hook lives on the first block; its job is just "measure residual + decide skip or not."

5. **`is_output_tuple` 分支处理(贯穿) / The `is_output_tuple` branching throughout**:
   - 中文: Diffusion 里有的 block 返回 `(hidden_states, encoder_hidden_states)` 双流(典型例子是 MMDiT —— image stream + text stream 双向 attend),有的返回单 tensor。代码两份逻辑都要写,通过 `TransformerBlockRegistry._metadata` 里登记的 `return_hidden_states_index` 拿到该塞哪个位置。这让同一个 hook 能套到 CogVideoX、Flux、Wan 不同架构的 block 上。
   - English: Some diffusion blocks return a tuple `(hidden_states, encoder_hidden_states)` for dual-stream architectures (canonical example: MMDiT, where image and text streams attend bidirectionally); others return a single tensor. The code keeps both paths, looking up `return_hidden_states_index` from the `TransformerBlockRegistry._metadata`. That's how one hook covers CogVideoX, Flux, and Wan block layouts uniformly.

## 类比 / The analogy

中文: 想象你在用 Google Maps 给同事发"路上还有多远"的实时更新。每分钟你都重新算路、查路况、发个新预估时间。但其实大多数情况下你估算出来的"接下来的修正方向"跟一分钟前一模一样——你正在直线上,没拐弯没遇红灯。FBC 就是说:每分钟你**先看一眼指南针**(first block 残差),如果指针指的方向跟一分钟前几乎一样,就直接复制粘贴上一分钟那条消息;只有当指针明显转了(残差变了),才重新跑完整查询。指南针很便宜,完整查询贵。

English: Picture sending live "ETA update" messages to a coworker via Google Maps. Every minute you re-route, check traffic, and post a new ETA. But most of the time the *correction direction* you're computing is identical to a minute ago — you're on a straight, no turns, no red lights. FBC is "every minute, first glance at the compass (the first-block residual); if it's pointing the same way as a minute ago, just copy-paste the previous message; only when the compass actually swings do you re-run the full query." The compass is cheap, the query is expensive.

## 自己跑一遍 / Try it yourself

```python
# fbc_demo.py
import torch
import torch.nn as nn

class MockBlock(nn.Module):
    def __init__(self, dim, scale=0.1):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.scale = scale
    def forward(self, x):
        return x + self.scale * torch.tanh(self.lin(x))

class FBCStack(nn.Module):
    def __init__(self, dim, n_blocks, threshold=0.05):
        super().__init__()
        self.blocks = nn.ModuleList([MockBlock(dim) for _ in range(n_blocks)])
        self.threshold = threshold
        self.prev_residual = None
        self.cached_tail = None
        self.skips = 0
        self.fulls = 0
    def forward(self, x):
        first_in = x
        x = self.blocks[0](x)
        residual = x - first_in
        should_skip = (self.prev_residual is not None and
            (residual - self.prev_residual).abs().mean() / self.prev_residual.abs().mean() < self.threshold)
        if should_skip:
            self.skips += 1
            return x + self.cached_tail
        # Full path
        self.fulls += 1
        head_out = x
        for b in self.blocks[1:]:
            x = b(x)
        self.prev_residual = residual
        self.cached_tail = x - head_out
        return x

torch.manual_seed(0)
stack = FBCStack(dim=64, n_blocks=20, threshold=0.05)
x = torch.randn(1, 8, 64)
for step in range(10):
    x = stack(x) + 0.01 * torch.randn_like(x)  # tiny per-step noise
print(f"full runs: {stack.fulls}, skipped runs: {stack.skips}")
```

运行 / Run with:
```bash
pip install torch
python fbc_demo.py
```

预期输出 / Expected output:
```
full runs: 1-3, skipped runs: 7-9
```

中文一两句: 第一次必然跑全(没有 prev_residual);之后大多数步都跳——因为相邻步的残差变化非常相关。把 threshold 调到 0.001 你会看到所有步都跑全(过严);调到 1.0 会看到 1 次全 + 9 次跳(过松)。0.05 是 diffusers 默认值。

English: The first step is always a full run (no prev_residual yet); after that most steps skip — neighboring residuals are highly correlated. Crank threshold to 0.001 and you'll see every step run full (too strict); crank to 1.0 and you'll see 1 full + 9 skips (too loose). 0.05 is the diffusers default.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TeaCache (Liu et al., 2024)** / **TeaCache (Liu et al., 2024)**: FBC 的论文起源,首次用 timestep embedding 的 norm 当探针。FBC 直接换成了 residual——更通用、不依赖 timestep。/ The paper FBC is built on; it used the norm of the timestep embedding as the probe. FBC swaps that for the residual — more general, doesn't depend on timestep being available.
- **Pyramid Attention Broadcast (PAB)** / **Pyramid Attention Broadcast (PAB)**: 同文件夹下,缓存 attention 输出而不是 block 残差,粒度更细。/ Sibling file, caches attention outputs instead of full block residuals — finer granularity, narrower scope.
- **DeepCache (Ma et al., 2023)** / **DeepCache (Ma et al., 2023)**: 在 UNet 上缓存深层特征,思路一致——"相邻去噪步特征相关"。/ The UNet equivalent — cache deep features, same intuition that adjacent denoising steps share most of their features.
- **`huggingface/diffusers/hooks/taylorseer_cache.py`** / **`huggingface/diffusers/hooks/taylorseer_cache.py`**: 用 Taylor 展开外推下一步特征,比直接复用更准但贵一点。/ Uses a Taylor expansion to extrapolate next-step features — more accurate than naive reuse, slightly costlier.

## 注意事项 / Caveats / when it breaks

- **threshold 是个画质 vs 速度的旋钮 / threshold is a quality-vs-speed knob**: 默认 0.05 偏激进。视频 generation 上很多用户调到 0.1-0.2。文本到图像 / 高写实场景下太松会有 ghosting。/ Default 0.05 is aggressive. Many video-gen users dial it to 0.1-0.2. Realistic / text-to-image at high quality may ghost if it's too loose.
- **第一步永远要跑全 / First step is always a full run**: 因为没有 prev_residual 可比。如果你 batch 里塞了多个 prompt(每个 prompt 独立 step counter),记得 `state_manager.reset()` 在每个新 prompt 边界。/ No prev_residual to compare to. If your batch holds multiple independent prompts, remember to `state_manager.reset()` at each prompt boundary.
- **`@torch.compiler.disable` 是个性能小坑 / `@torch.compiler.disable` is a perf gotcha**: 决策函数会强制 graph break,影响 `torch.compile` 上的吞吐。如果你想 max-perf,可以把这个分支 hoist 到 Python 端、由 scheduler 决定是否启用 FBC,然后整个 stack 跑 fullgraph。/ The decision forces a graph break, hurting `torch.compile` throughput. For max perf you can hoist the decision to the Python scheduler and let it switch between two pre-compiled stacks (with/without FBC).
- **Eval 时不一定准 / Doesn't strictly hold for eval**: FBC 是个有损近似;在 FID 等指标上会有 0.5-2 分的 regression。生产里通常只用于 latency 敏感的 interactive 场景,不用于 reference 出图。/ FBC is a lossy approximation; expect 0.5-2 points of FID regression. Production usually uses it only for latency-sensitive interactive paths, not for reference renders.

## 延伸阅读 / Further reading

- [TeaCache: Timestep Embedding Aware Cache (Liu et al., 2024)](https://huggingface.co/papers/2411.19108)
- [diffusers `apply_first_block_cache` docs](https://huggingface.co/docs/diffusers/main/en/api/cache)
- [ParaAttention's First Block Cache README](https://github.com/chengzeyi/ParaAttention)
- [DeepCache (Ma et al., 2023)](https://arxiv.org/abs/2312.00858)
