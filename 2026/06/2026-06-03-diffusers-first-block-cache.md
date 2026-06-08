---
date: 2026-06-03
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/hooks/first_block_cache.py
permalink: https://github.com/huggingface/diffusers/blob/1d6199345801af2176f12596e9546353d7bbcb9b/src/diffusers/hooks/first_block_cache.py#L32-L142
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, huggingface, diffusers, dit-inference, caching, hooks]
---

# First-Block Cache: 拿首块残差当"风向标",其余 DiT 块直接跳过 / First-Block Cache: use the first DiT block's residual as a weathervane, skip every block in between

> **一句话 / In one line**: 每个去噪步先算第一块,看它的 residual 和上一步差多少;差得不多就别算中间块了,直接复用上一步缓存的尾块 residual。 / On each denoising step, run only the first block, measure how much its residual changed since the previous step; if the change is small, skip every middle block and replay the cached tail residual instead.

## 为什么重要 / Why this matters

DiT 类模型推理慢的核心是"50 个 timestep × N 个 transformer block"的乘法。已有的加速方案要么针对单个模型手写(TeaCache、DeepCache),要么压缩 timestep(蒸馏)。FBCache 选了第三条路:**模型不动,timestep 不动**,只在 forward 里挂 hook,用第一块的 residual 差作为"这一步的特征还在动吗"的代理信号。差得小 → 跳过所有中间块,直接拿上一步存好的尾块 residual 拼回来;差得大 → 老老实实算完整 forward 并更新缓存。这个想法本身不复杂,巧的是 diffusers 用统一的 `HookRegistry + ModelHook + StateManager` 把它做成了**模型无关**的插件 —— 一句 `apply_first_block_cache(transformer, FirstBlockCacheConfig(threshold=0.2))` 就能给 CogView4、Wan、Flux 上同样的加速,不用动模型代码。

DiT inference is slow because of the `50 timesteps × N transformer blocks` multiplication. Existing accelerators either hand-craft per-model fast paths (TeaCache, DeepCache) or compress the timestep schedule (distillation). FBCache takes a third route: **leave the model alone, leave the schedule alone**, just attach forward hooks that use the first block's residual difference as a proxy for "are features still moving?". Small difference → skip every middle block and replay the previously cached tail residual; large difference → run the full forward and refresh the cache. The idea is simple; what's elegant is that diffusers wraps it in a unified `HookRegistry + ModelHook + StateManager` so the result is **model-agnostic**: one line `apply_first_block_cache(transformer, FirstBlockCacheConfig(threshold=0.2))` accelerates CogView4, Wan, Flux, and friends with no model-side surgery.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/hooks/first_block_cache.py`](https://github.com/huggingface/diffusers/blob/1d6199345801af2176f12596e9546353d7bbcb9b/src/diffusers/hooks/first_block_cache.py#L32-L142)

```python
@dataclass
class FirstBlockCacheConfig:
    threshold: float = 0.05      # higher = more skipping = faster, lower quality


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
            hidden_states_residual = (
                output[self._metadata.return_hidden_states_index] - original_hidden_states
            )
        else:
            hidden_states_residual = output - original_hidden_states

        shared_state: FBCSharedBlockState = self.state_manager.get_state()
        should_compute = self._should_compute_remaining_blocks(hidden_states_residual)
        shared_state.should_compute = should_compute

        if not should_compute:
            # CACHE HIT: skip middle blocks, just add cached tail residual to head's own output
            if is_output_tuple:
                hidden_states = (
                    shared_state.tail_block_residuals[0]
                    + output[self._metadata.return_hidden_states_index]
                )
            else:
                hidden_states = shared_state.tail_block_residuals[0] + output
            if self._metadata.return_encoder_hidden_states_index is not None:
                encoder_hidden_states = (
                    shared_state.tail_block_residuals[1]
                    + output[self._metadata.return_encoder_hidden_states_index]
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
            # CACHE MISS: store head's residual so we can compare on the next timestep
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
        prev = shared_state.head_block_residual
        absmean = (hidden_states_residual - prev).abs().mean()
        prev_absmean = prev.abs().mean()
        diff = (absmean / prev_absmean).item()
        return diff > self.threshold
```

## 逐行讲解 / What's happening

1. **`FirstBlockCacheConfig.threshold`(默认 0.05)/ `FirstBlockCacheConfig.threshold` (default 0.05)**:
   - 中文: 这是整个算法**唯一**要调的旋钮。`0.05` 表示"如果首块 residual 的相对变化 < 5%,就跳过中间块"。`0.2` 更激进 —— 大部分步都缓存命中,速度快但质量降。
   - English: this is the **only** knob in the entire algorithm. `0.05` means "if the first block's residual changed less than 5%, skip the middle blocks". `0.2` is more aggressive — most steps become cache hits, faster but lower quality.

2. **`FBCSharedBlockState` —— 头块和尾块的共享黑板 / `FBCSharedBlockState` — shared blackboard between head and tail**:
   - 中文: head hook 写 `head_block_residual`、读 `tail_block_residuals`;tail hook 反过来。两者通过 `state_manager` 共享 state。`should_compute` 是 head 在自己 forward 末尾设的旗子,middle 和 tail block 的 hook 都看它决定要不要跑。
   - English: the head hook writes `head_block_residual` and reads `tail_block_residuals`; the tail does the opposite. Both share state via `state_manager`. `should_compute` is a flag the head raises at the end of its own forward; middle/tail block hooks read it to decide whether to run.

3. **`hidden_states_residual = output - original_hidden_states`**:
   - 中文: 这就是 transformer block 的"残差贡献"—— 输入减输出。FBCache 比较的不是激活值本身,而是**这一块加了多少**。同一时刻同一像素,如果残差贡献相比上一步几乎没变,说明这一步对它没什么"新意"。
   - English: this is the transformer block's *residual contribution* — input minus output. FBCache doesn't compare activations themselves; it compares **how much this block added**. If the residual barely changed from the previous timestep at the same spatial location, the block doesn't have much "new" to say this step.

4. **`_should_compute_remaining_blocks`(算法核心 / the algorithmic core)**:
   - 中文: 取这一步的残差和上一步的残差作差,算 absmean,再除以上一步残差的 absmean 做归一化(相对变化率)。大于阈值 → 必须计算;小于阈值 → 跳过。第一步因为没有 prev,自动算完整 forward。
   - English: take this step's residual minus last step's residual, take absmean, divide by last step's absmean to get a *relative* change rate. Above threshold → must compute; below → skip. The very first step has no prev, so the full forward runs.

5. **`@torch.compiler.disable`**:
   - 中文: 这个判断里有 Python `if` 和 `.item()`(GPU → CPU 同步),Dynamo 跟踪会崩。直接给这一段标 `disable`,周围的 forward 仍然可以 compile,只是这一小段走 eager。是个让"动态控制流"和 `torch.compile` 共存的经典做法。
   - English: this predicate contains a Python `if` and an `.item()` (GPU → CPU sync), both of which break Dynamo tracing. Marking just this method `disable` lets the surrounding forward still compile, while this dynamic decision falls back to eager. A clean pattern for mixing dynamic control flow with `torch.compile`.

6. **缓存命中分支 / Cache-hit branch (`if not should_compute`)**:
   - 中文: 拿上一步存的 `tail_block_residuals` 直接加到本步 head 的 output 上 —— 等价于"假装中间所有块运行了,但它们什么也没改"。注意:这里仍然跑了 head 的真实 forward,只是省下了中间 + 尾 block 的所有 attention 和 MLP。一个 DiT 通常 30 层,省下 28 层 forward 是很可观的。
   - English: take the previously stored `tail_block_residuals` and add it to *this* step's head output — equivalent to pretending all middle blocks ran but contributed nothing new. Note the head's real forward still ran; what we saved is every middle + tail block's attention and MLP. A DiT typically has 30 layers; skipping 28 of them is a meaningful win.

7. **`apply_first_block_cache`(没在 snippet 里但值得提 / not in snippet but worth mentioning)**:
   - 中文: 在 `apply_first_block_cache` 函数里(L193+),它会扫 `_ALL_TRANSFORMER_BLOCK_IDENTIFIERS` 里登记过的 `nn.ModuleList`,第一个挂 `FBCHeadBlockHook`,最后一个挂 `is_tail=True` 的 `FBCBlockHook`(它负责存 `tail_block_residuals`),中间的挂普通 `FBCBlockHook`(它读 `should_compute` 决定要不要跑)。这就是为什么一行 `apply_first_block_cache(model, cfg)` 能在任何注册了的 DiT 上即插即用。
   - English: in `apply_first_block_cache` (L193+), the function scans `nn.ModuleList`s named in `_ALL_TRANSFORMER_BLOCK_IDENTIFIERS`, attaches `FBCHeadBlockHook` to the first block, `FBCBlockHook(is_tail=True)` to the last (which writes `tail_block_residuals`), and plain `FBCBlockHook` to the middle ones (which read `should_compute` to decide whether to run). That's how one `apply_first_block_cache(model, cfg)` line works generically on any registered DiT.

## 类比 / The analogy

中文:想象一个**编辑校稿**。第一遍粗读他注意到这章和上一章相比"几乎只改了几个标点"(首块 residual 变化很小);他直接合上书,假设"中间所有段落都没变",拿上一次校稿的修改记录(`tail_block_residuals`)套用到这一章。如果哪天他粗读发现"这一章整段都重写了",他才会真的逐段细读,顺便更新自己的"上次改了什么"笔记。FBCache 就是这个偷懒但有效的编辑 —— 阈值 0.05 等于他对自己的承诺:"标点差异 5% 以内我就不细看了"。

English: think of an **editor proofreading**. On the first pass, they notice this chapter "barely changed from the last one — just a few punctuation tweaks" (small first-block residual change); they close the book, *assume* all middle paragraphs are unchanged, and apply last time's edit notes (`tail_block_residuals`) to this chapter. If the first pass instead reveals the chapter was wholly rewritten, only then do they read it line by line and refresh their "what changed last time" notebook. FBCache is that lazy-but-effective editor; the 0.05 threshold is their personal rule: "if it's within 5% I won't re-read."

## 自己跑一遍 / Try it yourself

```python
# Standalone: simulates FBCache on a tiny fake DiT to show the decision logic.
import torch

class TinyFBCache:
    def __init__(self, threshold=0.05):
        self.threshold, self.prev_residual, self.tail_residual = threshold, None, None
        self.skips, self.computes = 0, 0
    def step(self, x_in, head_out, full_tail_out):
        residual = head_out - x_in
        if self.prev_residual is None:
            should_compute = True
        else:
            diff = (residual - self.prev_residual).abs().mean()
            should_compute = (diff / self.prev_residual.abs().mean()).item() > self.threshold
        if should_compute:
            self.computes += 1
            self.prev_residual = residual
            self.tail_residual = full_tail_out - head_out   # would be set by tail hook
            return full_tail_out
        else:
            self.skips += 1
            return self.tail_residual + head_out            # cache hit: replay tail

cache = TinyFBCache(threshold=0.05)
torch.manual_seed(0)
x = torch.randn(1, 4, 8)
for t in range(20):
    head_drift = 0.01 * torch.randn_like(x)        # small per-step change
    head = x + head_drift                          # simulate head block output
    full = head + 0.5 * torch.tanh(head)           # simulate middle+tail
    out = cache.step(x, head, full)
    x = out
print(f"skipped {cache.skips} / computed {cache.computes} of 20 steps")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
skipped 18 / computed 2 of 20 steps
```

中文:模拟里首块 residual 之间几乎没差异,所以第一步算完之后,后面 18 步全是缓存命中。换成 `threshold=0.0001`(更严格),你会看到几乎每步都计算 —— 这正是质量 vs. 速度的旋钮。

English: in this simulation the head-block residuals barely differ between steps, so after step 0 the next 18 steps are cache hits. Set `threshold=0.0001` (much stricter) and you'll see nearly every step computes — exactly the quality-vs-speed knob.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TeaCache** / **TeaCache**: FBCache 的祖宗,但用 polynomial fit + per-model 调参;FBCache 是它的"一键通用版"。 / The ancestor of FBCache, but with polynomial fits and per-model tuning; FBCache is the "one-knob universal" variant.
- **DeepCache** / **DeepCache**: U-Net 时代的同类思路 —— 跳过中间的 down/mid block,只跑 up block;思想一致(时间步间相关性高),实现完全不同。 / The U-Net-era cousin — skip the middle down/mid blocks, only run up blocks; same insight (cross-step correlation), totally different implementation.
- **vipshop/cache-dit** / **vipshop/cache-dit**: 整个 repo 都在做这类 DiT 推理缓存,FBCache 是里头最朴素也最好部署的一个。 / A whole repo dedicated to this family of DiT inference caches; FBCache is the simplest and easiest to ship from the lineup.
- **diffusers 的 `pyramid_attention_broadcast`** / **diffusers `pyramid_attention_broadcast`**: 同一个 `HookRegistry` 框架下的另一个加速 hook —— 学习一次,所有 hook 都能写。 / Another acceleration hook under the same `HookRegistry` framework — learn this pattern once and you can write any of them.

## 注意事项 / Caveats / when it breaks

- **早期步质量损失最大 / Quality hit concentrates on early steps**: 去噪刚开始时残差变化最剧烈,这时跳块代价最高。调阈值时建议先观察前 10 步的命中率,确保不要太激进。 / Residuals change most violently early in denoising; skipping there hurts most. When tuning the threshold, eyeball the first 10 steps' hit rate and don't let early skipping get too aggressive.
- **批内不同 prompt 共享缓存 / Cache shared across batch elements**: 当前实现里 state 是 module-级的,batch 中所有元素共用同一个 should_compute 判断 —— 如果 batch 里有的 prompt 收敛快、有的慢,你只能照最慢的那个来。 / The state is module-scope, so all batch elements share one `should_compute` verdict. If some prompts converge fast and others slow, you can only cache as aggressively as the slowest one allows.
- **`@torch.compiler.disable` 不能省 / Don't drop `@torch.compiler.disable`**: 想 compile 整张图的话,这个装饰器是必须的,否则 Dynamo 会在 `.item()` 处 graph-break,而且 break 的位置不可预测。 / Required if you want to compile the surrounding pipeline; without it Dynamo graph-breaks at `.item()` in unpredictable ways.
- **只支持 register 过的 transformer 类 / Only works on registered transformer classes**: `TransformerBlockRegistry` 需要事先用 `_get_parameter_from_args_kwargs` 注册过这个块的输入/输出语义,自家自定义模型要手动注册。 / `TransformerBlockRegistry` needs prior registration with `_get_parameter_from_args_kwargs` describing how the block reads/returns hidden states. Custom models must register themselves.

## 延伸阅读 / Further reading

- [FBCache origin: ParaAttention's first-block-cache writeup](https://github.com/chengzeyi/ParaAttention/blob/main/README.md#first-block-cache-our-dynamic-caching)
- [TeaCache paper](https://huggingface.co/papers/2411.19108)
- [diffusers `hooks/` module overview](https://github.com/huggingface/diffusers/tree/main/src/diffusers/hooks)
- [Existing daily-code entry: Wan2.1's denoise loop](../../nano/wam/2026-05-29-wan21-denoise-loop.md) — pair this acceleration with that loop.
