---
date: 2026-06-04
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_block_refinement.py
permalink: https://github.com/huggingface/diffusers/blob/9b0818cf87413b4b9ca2501bf49406eed6d881af/src/diffusers/schedulers/scheduling_block_refinement.py#L51-L104
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, huggingface, diffusers, scheduler, masked-diffusion, llada, discrete-diffusion]
---

# diffusers 里多了一个"按置信度提交"的 scheduler / diffusers now ships a "commit-by-confidence" scheduler

> **一句话 / In one line**: `BlockRefinementScheduler` 不是噪声 scheduler —— 它给 masked-diffusion 语言模型用,每一步把模型最有把握的若干 mask 位置"提交"成 token,直到整块都填满。 / `BlockRefinementScheduler` is not a noise scheduler. It's for LLaDA-style masked-diffusion language models — each step commits the most confidently predicted mask positions to real tokens, until the whole block is filled.

## 为什么重要 / Why this matters

LLaDA、LLaDA-2、Mercury 这类"扩散语言模型"在 2025-2026 上来了 —— 它们不像 autoregressive LM 一个 token 一个 token 吐,而是一开始把整块都置成 `[MASK]`,然后多次"细化"。每一步模型对每个 mask 位置同时预测一个 logit,scheduler 决定"哪些位置足够确信,可以从 mask 升级为真 token"。如果你只看过 DDPM/DDIM/Euler 这种连续噪声 scheduler,会很惊讶 diffusers 居然能把同一接口套到完全离散的世界 —— 没有 betas、没有 noise level、没有 epsilon prediction,但 `step()` / `set_timesteps()` 的签名一模一样。这段代码值得读一遍,因为它告诉你 "scheduler" 在 diffusers 里其实是个抽象的 "iterative-refinement controller",远不止 noise schedule。

LLaDA, LLaDA-2, Mercury and other "diffusion language models" landed in 2025-2026. Instead of decoding one token at a time like an autoregressive LM, they fill a block by starting from all-`[MASK]` and refining it over several passes. Each step the model produces logits for every masked position, and the scheduler decides which positions are confident enough to be "promoted" from mask to real token. If you've only seen DDPM/DDIM/Euler-style continuous-noise schedulers, it's surprising that diffusers can wrap the same interface around a fully discrete process — no betas, no noise level, no epsilon prediction, but `step()` / `set_timesteps()` look identical. The takeaway is that a "scheduler" in diffusers is really an abstract iterative-refinement controller; noise schedules are just one instance of it.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_block_refinement.py`](https://github.com/huggingface/diffusers/blob/9b0818cf87413b4b9ca2501bf49406eed6d881af/src/diffusers/schedulers/scheduling_block_refinement.py#L51-L104)

```python
class BlockRefinementScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for block-wise iterative refinement (commit-by-confidence).

    At each step, the scheduler samples candidate tokens from model logits and commits those with the highest
    confidence. The number of tokens to commit per step is determined by evenly distributing the block length across
    the number of refinement steps.

    Optionally supports editing: after all mask tokens are resolved, tokens can be replaced if the model predicts a
    different token with confidence above a positive `editing_threshold` (`None`, `0.0`, or negative disables editing).
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        block_length: int = 32,
        num_inference_steps: int = 32,
        threshold: float = 0.95,
        editing_threshold: float | None = None,
        minimal_topk: int = 1,
    ):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, dtype=torch.long)
        self._transfer_schedule: torch.LongTensor | None = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        block_length: int | None = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        if block_length is None:
            block_length = self.config.block_length
        elif block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, device=device, dtype=torch.long)
        self._transfer_schedule = self.get_num_transfer_tokens(block_length, self.num_inference_steps).to(
            device=device if device is not None else "cpu"
        )

    def get_num_transfer_tokens(self, block_length: int, num_inference_steps: int) -> torch.LongTensor:
        """Evenly distribute `block_length` token commits across `num_inference_steps` steps."""
        if num_inference_steps <= 0:
            return torch.zeros((0,), dtype=torch.long)
        base = block_length // num_inference_steps
        remainder = block_length % num_inference_steps
        out = torch.full((num_inference_steps,), base, dtype=torch.long)
        out[:remainder] += 1
        return out
```

## 逐行讲解 / What's happening

1. **`order = 1`**:
   - 中文: diffusers 协议字段。表示这是一个 first-order scheduler(每步只看当前模型输出,不需要保留过去若干步的预测做 multi-step solver)。
   - English: A diffusers protocol field. Marks this as a first-order scheduler — each step depends only on the current model output, no need to keep previous-step predictions for a multi-step solver.
2. **`timesteps = torch.arange(num_inference_steps - 1, -1, -1, ...)`**:
   - 中文: 倒序时间步(31, 30, ..., 0)。这里 timestep 只是"步索引",不像 DDPM 那样代表噪声水平。
   - English: A reverse-order step index (31, 30, ..., 0). Here the timestep is just a *step counter*, not a noise level like in DDPM.
3. **`get_num_transfer_tokens(block_length, num_inference_steps)`** —— 整个 scheduler 的灵魂:
   - 中文: 用整除 + 余数把 `block_length` 个 token 名额平摊到 `num_inference_steps` 步上。`base = block_length // num_inference_steps` 是基础名额;`out[:remainder] += 1` 把多出来的 token 名额分给前几步。例如 `block=32, steps=8` → 每步 4;`block=33, steps=8` → 前一步 5、后七步 4。
   - English: Integer-divide + remainder to distribute `block_length` commit slots across `num_inference_steps` steps. `base` is the floor, then the first `remainder` steps each get one extra. E.g. `block=32, steps=8` → 4 per step; `block=33, steps=8` → 5 in the first step, 4 in the remaining seven.
4. **`set_timesteps(...)` 把上面这张 transfer 表存进 `self._transfer_schedule`**:
   - 中文: 每次推理之前调用,告诉 scheduler "总共多少步、每步提交多少个 mask"。把表预存,`step()` 只需 `self._transfer_schedule[step_index]` 查一下。
   - English: Called once before inference; precomputes the per-step quota so `step()` can just index into `self._transfer_schedule[step_index]`.
5. **下游 `step()` 的逻辑**(代码省略但要知道):
   - 中文: 每步用 `top_p` / `top_k` 从 logits 采样,得到候选 token + 它们的概率。只看 mask 位置的概率,挑最高的若干 commit,其余继续保持 mask。如果模型对足够多的 mask 都高置信(`> threshold`),就一次全 commit 它们;否则退回 top-k 强制 commit 配额数。
   - English: At each step, sample candidate tokens from `top_p` / `top_k`-filtered logits along with their probabilities. Look only at mask positions, commit the most confident ones, leave the rest masked. If at least the quota count is above `threshold`, commit them all; otherwise fall back to top-k to fill the quota.

## 类比 / The analogy

想象你在做一道 32 道填空题,有 8 次答题机会,但每次你可以同时盯着所有 32 道题想:"哪些我现在最有把握?" `get_num_transfer_tokens` 是考官给你的"配额表":第一次至少填 4 个,第二次再填 4 个,以此类推。每次你都对 32 道题都给出一个"猜测 + 自信度",然后挑自信度最高的 4 道写上去固定下来,剩下的下一次再想。优势是:容易的题先填(高置信),后面填难题时已经有了上下文(已经写下来的字给整段提供了线索),不像自回归一定要从左到右。

Imagine you have 32 fill-in-the-blank questions and 8 attempts. Each attempt you can re-examine all 32 simultaneously and ask "which ones am I most sure of now?" `get_num_transfer_tokens` is the proctor's quota: commit at least 4 in attempt 1, 4 more in attempt 2, etc. You produce a guess plus a confidence for every blank, then *lock in* the four with the highest confidence and leave the rest masked. The benefit: easy blanks get filled first; when you tackle the hard ones, the already-committed answers give the rest of the sentence as context. No left-to-right constraint like an autoregressive LM.

## 自己跑一遍 / Try it yourself

```python
import torch

def get_num_transfer_tokens(block_length, num_inference_steps):
    base = block_length // num_inference_steps
    remainder = block_length % num_inference_steps
    out = torch.full((num_inference_steps,), base, dtype=torch.long)
    out[:remainder] += 1
    return out

# A toy "block refinement" generator: 8 mask positions, 4 steps.
torch.manual_seed(0)
MASK = -1
block = torch.full((8,), MASK, dtype=torch.long)
quota = get_num_transfer_tokens(block_length=8, num_inference_steps=4)
print("transfer quota per step:", quota.tolist())  # [2, 2, 2, 2]

for step, n in enumerate(quota):
    # toy "model" — every step emits random logits over a vocab of 5
    logits = torch.randn(8, 5)
    probs = logits.softmax(-1)
    top_p, top_id = probs.max(-1)
    # only look at currently-masked positions
    masked_pos = (block == MASK).nonzero(as_tuple=True)[0]
    conf = top_p[masked_pos]
    # commit the `n` most confident masked positions
    _, top_idx = torch.topk(conf, k=int(n))
    chosen = masked_pos[top_idx]
    block[chosen] = top_id[chosen]
    print(f"step {step}: committed {chosen.tolist()} -> {block.tolist()}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
transfer quota per step: [2, 2, 2, 2]
step 0: committed [...] -> [..., -1, ..., -1, ...]
step 1: committed [...] -> [...]
step 2: committed [...] -> [...]
step 3: committed [...] -> [0, 3, 1, 2, ...]    <- all filled
```

中文重点:8 个 mask、4 步、每步提交 2 个 —— 最关键的发现是 token 的提交**顺序不固定**,每次取决于"当前最自信的"是哪几个,这就是和自回归本质区别。

The key thing to notice: with 8 masks and 4 steps committing 2 each, the *order* of commits is data-dependent — whichever positions are most confident this step — not fixed left-to-right. That's the core difference from autoregressive decoding.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MaskGIT (image generation)** / **MaskGIT**: 同样的"按置信度提交" idea,但用在 VQGAN 离散图像 token 上。 / Same commit-by-confidence idea, just applied to VQGAN discrete image tokens.
- **Lookahead decoding / speculative decoding** / **Lookahead / speculative decoding**: 也是"一次预测一块,挑能接受的提交",但是用 draft model 而不是 confidence threshold。 / Also "predict a block, accept the prefix that matches"; uses a draft model instead of confidence threshold.
- **diffusers `EulerDiscreteScheduler.step`** / **EulerDiscreteScheduler.step**: 接口签名几乎一致 —— 同样 `(model_output, timestep, sample) -> next_sample`,但 sample 是连续 latent,model_output 是 epsilon。 / Almost the same signature — `(model_output, timestep, sample) -> next_sample` — but `sample` is a continuous latent and `model_output` is epsilon.
- **iterative parallel decoding in DiffuLLaMA / Mercury** / **DiffuLLaMA, Mercury**: production-scale 扩散语言模型,本质上就是把这个 scheduler 套上一个 LLaMA backbone。 / Production-scale diffusion LMs that are essentially this scheduler wrapped around a LLaMA backbone.

## 注意事项 / Caveats / when it breaks

- **`threshold` 太高反而退化** / **threshold too high degrades to top-k**: 如果阈值卡到 0.99,大部分步都没人达到,最后所有 step 都退回 top-k 强制提交,scheduler 就和"分块 argmax"差不多了。常用 0.85 - 0.95。 / Cranking threshold to 0.99 means almost no step has enough above-threshold positions; everything falls back to top-k and you've effectively got "chunked argmax". The sweet spot is usually 0.85 - 0.95.
- **`block_length < num_inference_steps`** / **block shorter than steps**: `base = 0`,只有前 `remainder` 步会提交一个 token,后面的步都是空操作。一般避免。 / `base = 0`, so only the first `remainder` steps commit anything and the rest are no-ops. Usually avoid.
- **`editing_threshold` 是后置编辑** / **editing happens after the main commit**: 当 block 已经全填完后,editing 阶段才会触发,允许重写之前提交过的 token —— 适合长文本的"自我修正",但代价是多一轮 forward。 / Editing only runs after the block is fully filled. It can rewrite previously committed tokens — useful for long-form self-correction but costs an extra forward.
- **不要把 timestep 当 noise level 用** / **timestep is not noise level**: 这里的 timestep 只是 step index,和 DDPM 那种 `sigma_t` 没关系。如果你写 model wrapper,记得 timestep embedding 的输入是 step 序号,不是连续噪声。 / The timestep here is just a step index, not `sigma_t` from DDPM. If you wrap the model, the timestep embedding gets a step number, not a continuous noise level.

## 延伸阅读 / Further reading

- LLaDA paper (Large Language Diffusion Models): <https://arxiv.org/abs/2502.09992>
- MaskGIT: <https://arxiv.org/abs/2202.04200>
- diffusers Scheduler design doc: <https://huggingface.co/docs/diffusers/api/schedulers/overview>
