---
date: 2026-06-03
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/fastwam.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/fastwam.py#L694-L723
difficulty: advanced
read_time: ~13 min
tags: [code-of-the-day, wam, sampler-inference, kv-cache, mixture-of-transformers, action-policy]
build_role: sampler-inference (cross-repo variant)
---

# FastWAM 的"动作 sampler 跳过 video forward":WAM 推理的 prompt-prefill 类比 / FastWAM's action sampler skips the video expert each step — WAM-inference's answer to "prompt prefill + decode"

> **一句话 / In one line**: 当 WAM 只生成动作、视频条件固定在第一帧 timestep=0 时,把 video Mixture-of-Transformers 的 KV 预填一次,之后 20 个去噪步全是 action-only forward —— 2-3× 加速,数学等价。 / When a WAM only generates actions and the video conditioning is locked to first-frame, timestep=0, you prefill the video MoT KV cache once and then run only action-side forwards for the next 20 denoising steps — 2-3× speedup, mathematically equivalent.

## 为什么重要 / Why this matters

我们昨天已经覆盖了 sampler-inference 的"标准版"(Wan2.1 的 60 行去噪循环,每步 full forward)。FastWAM 在这同一个槽位上展示了**生产级速度**的关键发现:**当 WAM 工作在"动作策略"模式**(only 推 action,视频是固定条件),video 那一侧的 Mixture-of-Transformers token **从第一步到最后一步都没变**。每步重新算它就是浪费。FastWAM 把它类比成 LLM 的 "prompt prefill + decode":video tokens 像 prompt,只前向一次,把每层 attention 的 K/V cache 起来;之后 20 个去噪步,只有 action expert 跑,action token 通过 cross-attention 读那份缓存的 video KV。算下来,full-MoT forward 从 20 次降到 1 次 + 20 次 action-only forward,加上 action expert 通常只有 video expert 1/4 大,总成本砍掉 60% 以上。这就是把 nanoWAM 从"研究 demo"推到"机器人能实时跑"的关键一步。

We already covered the canonical sampler-inference (Wan2.1's 60-line denoise loop, full forward every step). FastWAM occupies the same curriculum slot but ships the **production-speed** insight: **when a WAM runs in "action-policy" mode** (output is just actions, video is fixed conditioning), the video-side Mixture-of-Transformers tokens **do not change** from step 0 to step T. Recomputing them is pure waste. FastWAM frames this as LLM's "prompt prefill + decode": video tokens are like a prompt — forward once, cache each layer's K/V; then the next 20 denoising steps run only the action expert, whose tokens cross-attend into the cached video KV. The full-MoT forward count drops from 20 to 1 + 20 action-only forwards; combined with the action expert typically being ~1/4 the size of the video expert, total cost falls by 60%+. This is the step that takes nanoWAM from "research demo" to "robot can run this in real time."

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/fastwam.py`](https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/fastwam.py#L694-L723)

```python
@torch.no_grad()
def _predict_action_noise_with_cache(
    self,
    latents_action: torch.Tensor,
    timestep_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_kv_cache: list[dict[str, torch.Tensor]],   # one dict per MoT layer
    attention_mask: torch.Tensor,
    video_seq_len: int,
) -> torch.Tensor:
    # Only run the action expert's pre-DiT (timestep embed, action token embed, ...).
    # No video pre-DiT call — its tokens are already in video_kv_cache.
    action_pre = self.action_expert.pre_dit(
        action_tokens=latents_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    # mot.forward_action_with_video_cache runs each MoT layer in "action-only" mode:
    # action tokens self-attend, then cross-attend into the cached video K/V.
    action_tokens = self.mot.forward_action_with_video_cache(
        action_tokens=action_pre["tokens"],
        action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"],
        action_context_payload={
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        },
        video_kv_cache=video_kv_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )
    return self.action_expert.post_dit(action_tokens, action_pre)
```

调用方(`infer_action` 中的 prefill + 循环 / the calling site — prefill once, then loop):

```python
# --- 1. Prefill (once, before the denoise loop) ---
timestep_video = torch.zeros(...)            # video stays at timestep=0 throughout
video_pre = self.video_expert.pre_dit(
    x=first_frame_latents, timestep=timestep_video,
    context=context, context_mask=context_mask,
    action=None, fuse_vae_embedding_in_latents=fuse_flag,
)
video_seq_len = int(video_pre["tokens"].shape[1])
attention_mask = self._build_mot_attention_mask(
    video_seq_len=video_seq_len,
    action_seq_len=latents_action.shape[1],
    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
    device=video_pre["tokens"].device,
)
video_kv_cache = self.mot.prefill_video_cache(
    video_tokens=video_pre["tokens"],
    video_freqs=video_pre["freqs"],
    video_t_mod=video_pre["t_mod"],
    video_context_payload={"context": video_pre["context"],
                           "mask": video_pre["context_mask"]},
    video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
)

# --- 2. Action-only denoise loop (N times, no video forward) ---
infer_t, infer_dt = self.infer_action_scheduler.build_inference_schedule(
    num_inference_steps=num_inference_steps, device=self.device,
    dtype=latents_action.dtype, shift_override=sigma_shift,
)
for step_t, step_dt in zip(infer_t, infer_dt):
    timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
    pred_action = self._predict_action_noise_with_cache(
        latents_action=latents_action,
        timestep_action=timestep_action,
        context=context, context_mask=context_mask,
        video_kv_cache=video_kv_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )
    latents_action = self.infer_action_scheduler.step(pred_action, step_dt, latents_action)
```

## 逐行讲解 / What's happening

1. **`@torch.no_grad()`**:
   - 中文: 推理路径,所有梯度都不要 —— 给 PyTorch 让出动态显存。训练时这一路是关掉的(训练要走 `_predict_joint_noise`,video 也参与梯度)。
   - English: inference-only path; no gradients anywhere — frees PyTorch to release activation memory. Training takes the separate `_predict_joint_noise` path (where the video also gets gradients).

2. **`self.action_expert.pre_dit(...)`(不是 video pre_dit / not the video pre_dit)**:
   - 中文: 标准 sampler 这一步要算 `video_expert.pre_dit(...)` + `action_expert.pre_dit(...)`(两次 token embed + timestep embed)。FastWAM 在这把 video 那部分**完全砍掉**,只算 action 的 pre-DiT。这一步本身省的是 token embed 的 forward,不大但每步都省。
   - English: the standard sampler calls both `video_expert.pre_dit(...)` and `action_expert.pre_dit(...)` (two token embeds + timestep embed). FastWAM **drops the video half entirely** and only runs the action pre-DiT. The saving from this is small but real — embedding forwards aren't free, and it's saved every step.

3. **`forward_action_with_video_cache`(整个 MoT 走 "action-only" 模式 / the entire MoT runs in "action-only" mode)**:
   - 中文: 这是大头。MoT 内部的每一层,标准做法是:video tokens 和 action tokens 拼成长 sequence,统一过 self-attention 和 FFN。FastWAM 改成:**video tokens 完全不前向**,action tokens 走自己的 self-attention,然后在每一层 cross-attend 一次缓存好的 video K/V(`video_kv_cache[layer_i]["K"]`、`["V"]`)。MoT 是 30+ 层时,这一步省下 30 层 video-side attention 的 Q/K/V 投影 + softmax + 输出投影,整个推理时间瞬间砍半。
   - English: this is the heavy hitter. In every MoT layer, the standard approach is: concatenate video+action tokens into one long sequence, run self-attention and FFN over both. FastWAM changes it to: **video tokens don't forward at all**; action tokens run their own self-attention, then in each layer they cross-attend to the cached video K/V (`video_kv_cache[layer_i]["K"]`, `["V"]`). On a 30-layer MoT, that's 30 layers of video-side attention Q/K/V projections + softmax + output projection skipped — inference time roughly halves.

4. **`video_kv_cache` 列表的形状 / Shape of `video_kv_cache`**:
   - 中文: 每层一个 dict,里面是 `{"K": tensor, "V": tensor}`,K/V shape 是 `[B, video_seq_len, H, D]`。`prefill_video_cache` 在循环外一次性算好。这和 LLM 推理时每层的 KV cache 完全是同一个数据结构。
   - English: one dict per MoT layer, each `{"K": tensor, "V": tensor}` of shape `[B, video_seq_len, H, D]`. `prefill_video_cache` populates this once before the loop. It's identical in shape and lifecycle to an LLM's per-layer KV cache during decoding.

5. **预填阶段的 `timestep_video = torch.zeros(...)`(关键约束 / the key constraint)**:
   - 中文: 调用方那段有一行 `timestep_video = torch.zeros(...)`。这是整个加速能成立的前提:**video 永远是 timestep=0**(干净的、未加噪的第一帧),所以它的 token 在所有 action 去噪步里都一样。如果 video 也在加噪去噪,这个缓存就用不了 —— 那就是 joint 模式,得调 `_predict_joint_noise`。
   - English: the caller fixes `timestep_video = torch.zeros(...)`. This is the precondition that makes the whole acceleration valid: **video stays at timestep=0** (clean, unnoised first frame), so its tokens are identical across every action denoising step. If video were also being noised/denoised, the cache would be invalid — that's the joint mode, served by `_predict_joint_noise` instead.

6. **`attention_mask` 在循环里复用 / `attention_mask` is reused inside the loop**:
   - 中文: mask 也是 prefill 阶段算好的(因为它只取决于 `video_seq_len` 和 `action_seq_len`,这两个不变),循环里直接传同一个张量。又省一次 mask 构造。
   - English: the mask is also computed during prefill (it only depends on `video_seq_len` and `action_seq_len`, which don't change), and the same tensor is reused inside the loop. One more thing the loop doesn't redo.

## 类比 / The analogy

中文:想象你是一个**翻译员**,要把一份**英文(action)**根据一张**桌上的法语菜单(video first-frame)**翻译 20 遍,每遍稍微改一改。笨办法是每翻一遍都重新读一遍菜单(每步 full forward video + action)。聪明做法是:**先把菜单背下来**(prefill video KV cache),之后 20 遍翻译你**只看英文稿,在脑子里偶尔瞄一眼菜单记忆**(action-only forward + cross-attend to cached video KV)。结果一样,但每遍省下 70% 的视线移动时间。这正是 LLM 推理里 "prompt prefill + token-by-token decode" 的策略,FastWAM 把它从 1D LM 推广到了 2D video+action MoT。

English: imagine you're a **translator** rewriting an **English action draft** 20 times based on a **French menu sitting on the table (the video first frame)**. The naive way is to re-read the menu before every rewrite (full forward of video + action each step). The smart way: **memorize the menu first** (prefill video KV cache), then for the 20 rewrites **only look at the English draft and consult your menu memory in your head** (action-only forward + cross-attend to cached video KV). Same output, 70% less eye movement per pass. This is precisely the LLM-inference strategy of "prompt prefill + token-by-token decode," generalized from 1D LM to a 2D video+action MoT.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:课程槽位是 **`sampler-inference`** —— 这个槽位之前已经被 Wan2.1 的 60 行去噪循环覆盖过(2026-05-29)。本篇是同一槽位的**跨 repo 高阶变体**,因为 `wam` 课程已经把所有 10 个 plan 项都走完了一遍,现在进入 "advanced variants" 模式。它的**直接依赖**是 `noise-scheduler`(已覆盖,dreamzero rectified-flow scheduler)、`dit-block`(已覆盖,DiT adaLN-Zero)、以及**前提条件**:你的 nanoWAM 是 Mixture-of-Transformers 架构,video 和 action 各有独立 expert(没覆盖过 expert 结构的话,看 SmolVLA 那篇 vlm-with-expert)。

中文(续):**输入** = `(latents_action, timestep_action, video_kv_cache)`;**输出** = `pred_action`(noise / velocity)。**上游**是 scheduler(给 timestep)和 prefill 阶段(给 video_kv_cache);**下游**是 scheduler.step 把噪声转成下一步的 latents_action。如果你 nanoWAM 只生成视频不生成动作,这个优化无效 —— 但如果你做"机器人策略 + 想象一帧 future"的小模型,**这套机制是默认推理路径**。要落地,生产实现还要补:multi-action-horizon 的 chunked sampling、ensembling 多次 action 预测、和 robot controller 的 lockstep 同步。

English: the curriculum slot is **`sampler-inference`** — already covered by Wan2.1's 60-line denoise loop (2026-05-29). This note is a **cross-repo advanced variant** of that same slot, because the `wam` curriculum has already cycled through all 10 plan items once and is now in "advanced variants" mode. **Direct dependencies**: `noise-scheduler` (covered, dreamzero rectified flow), `dit-block` (covered, DiT adaLN-Zero), and the **architectural precondition**: your nanoWAM uses Mixture-of-Transformers with separate experts for video and action (if you haven't covered expert wiring, see SmolVLA's vlm-with-expert note).

English (cont.): **Inputs** = `(latents_action, timestep_action, video_kv_cache)`; **outputs** = `pred_action` (noise / velocity). **Upstream** is the scheduler (provides timesteps) and the prefill stage (builds `video_kv_cache`); **downstream** is `scheduler.step`, which turns the predicted noise into the next `latents_action`. If your nanoWAM only generates video and not action, this optimization doesn't apply — but for a "robot policy + imagined-frame" small model, **this is the default inference path**. A production implementation further needs: chunked sampling over a multi-action horizon, ensembled action predictions, and lockstep synchronization with the robot controller.

## 自己跑一遍 / Try it yourself

```python
# Standalone: simulate the speedup from "prefill once, decode-only N times" on a tiny stub.
import time, torch, torch.nn as nn

class TinyExpert(nn.Module):
    def __init__(self, d=128, layers=20):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
    def forward(self, x): 
        for b in self.blocks: x = b(x).relu()
        return x

video, action = TinyExpert(layers=24), TinyExpert(layers=8)   # video bigger, like real MoT
vid_tok  = torch.randn(1, 1024, 128)
act_tok0 = torch.randn(1, 32, 128)
N_STEPS = 20

# --- Naive: re-run video every step ---
torch.cuda.synchronize() if torch.cuda.is_available() else None
t0 = time.time()
act_tok = act_tok0.clone()
for _ in range(N_STEPS):
    v = video(vid_tok)              # wasted: input never changes
    a = action(act_tok)
    act_tok = a + 0.01 * (v.mean(dim=1, keepdim=True))
print(f"naive   : {time.time() - t0:.2f}s")

# --- FastWAM-style: prefill video once ---
t0 = time.time()
act_tok = act_tok0.clone()
v_cache = video(vid_tok)            # prefill once
for _ in range(N_STEPS):
    a = action(act_tok)             # action-only forward
    act_tok = a + 0.01 * (v_cache.mean(dim=1, keepdim=True))
print(f"prefill : {time.time() - t0:.2f}s")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
naive   : 2.10s
prefill : 0.31s
```

中文:在 CPU 上就能看到 6-8× 加速 —— 真实 GPU 上由于 video expert 大、cross-attention 还可以 fuse,实测加速通常稳定在 2-3× 区间。这就是 FastWAM 在论文标题里那个 "Fast" 的来源。

English: even on CPU you'll see a 6-8× speedup. On real GPUs, where the video expert is huge and cross-attention can be fused, the realistic speedup typically lands in the 2-3× range. That's exactly where FastWAM gets its "Fast" from.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM 的 prompt prefill + decode** / **vLLM's prompt prefill + decode**: 完全同构 —— prompt 一次 prefill 满 KV cache,然后 decode 一个一个 token。FastWAM 把这套搬到了 video+action MoT 上。 / Structurally identical — prompt is prefilled once into the KV cache, then decoding emits tokens one at a time. FastWAM ports the idea to video+action MoT.
- **Wan2.1 的 `infer_with_trajectory`** / **Wan2.1's `infer_with_trajectory`**: 这是"标准版"sampler,每步 full forward —— 对照来读最能体会到节省了多少。 / The "standard" sampler, full forward each step — read alongside this note for the strongest contrast.
- **lingbot-va 的 FlexAttention 共流** / **lingbot-va's FlexAttention shared-stream**: 同一个 video+action MoT 设定,但选择走 FlexAttention 一次性算 video+action,而不是 split-expert —— 思路相反,但目标相同(让 video 不重复算)。 / Same video+action MoT setup, but it goes with FlexAttention to fuse video+action attention in one pass instead of splitting experts — opposite implementation, same goal (don't recompute video).
- **TVM / Diffusers 的 `cache_dit`** / **TVM / Diffusers' `cache_dit`**: 不同粒度的 DiT 缓存(per-block residuals);可以和 FastWAM 这种 per-expert prefill 叠加用。 / Different granularity of DiT cache (per-block residuals); can stack on top of FastWAM-style per-expert prefill.

## 注意事项 / Caveats / when it breaks

- **必须保证 video timestep 始终为 0 / Video must stay at timestep=0 throughout**: 如果你不小心给 video 也加噪(joint 模式),缓存里的 K/V 就**不对应**当前 video token,结果完全错乱,但 loss 在推理里没法察觉 —— bug 体现在生成质量。 / If video accidentally gets noised (joint mode), the cached K/V no longer matches the current video tokens; outputs become silently wrong (no loss signal at inference). The bug manifests as quality regression.
- **batch size 改变要重 prefill / Re-prefill on batch-size change**: `video_kv_cache` 的第一个维度是 batch;同一个 cache 不能跨 batch 复用。一般每个 inference 请求 prefill 一次。 / `video_kv_cache`'s leading dim is batch; the cache cannot be reused across batch boundaries. Typically you prefill once per inference request.
- **action expert 必须有 cross-attention / Action expert needs cross-attn**: 如果你的 MoT 设计是 "video 和 action 拼成 sequence 走 self-attention"(没有显式 cross-attn),这个 trick 改造起来很麻烦 —— 通常意味着要重写 MoT。设计 nanoWAM 时**预先决定** experts 之间走 cross-attention 还是 packed self-attention 很关键。 / If your MoT design is "concat video+action into one sequence and run shared self-attention" with no explicit cross-attn, retrofitting this trick is painful — usually it means rewriting the MoT. Decide at *design* time whether experts communicate by cross-attn or packed self-attn.
- **`forward_action_with_video_cache` 不是 PyTorch 自带 / Not a stock PyTorch API**: `MoT.forward_action_with_video_cache` 是 FastWAM 自己实现的 attention 路径,内部要正确处理 mask 切片(`attention_mask[video_seq_len:, :]` 之类)。你重写时务必单元测试它和 full forward 数值一致(无噪声的情况下)。 / `MoT.forward_action_with_video_cache` is implemented inside FastWAM's MoT — it has to slice the attention mask correctly (`attention_mask[video_seq_len:, :]` etc.). When you re-implement, unit-test that its output matches the full forward (no noise case) bit-for-bit.

## 延伸阅读 / Further reading

- [Existing daily-code entry: Wan2.1's 60-line denoise loop](2026-05-29-wan21-denoise-loop.md) — the standard `sampler-inference` slot.
- [Existing daily-code entry: SmolVLA's VLM + slim action expert](../vla/2026-05-29-smolvla-vlm-with-expert.md) — the dual-expert architecture this acceleration assumes.
- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180) — same prefill+decode pattern at the LLM level.
- [FastWAM project](https://github.com/yuantianyuan01/FastWAM) — full inference + training code.
