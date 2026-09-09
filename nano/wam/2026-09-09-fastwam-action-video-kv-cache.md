---
date: 2026-09-09
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/mot.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py#L370-L472
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, kv-cache, action-conditioning]
build_role: sampler-inference advanced variant
---

# FastWAM action cache：视频 K/V 只算一次，动作每步重算 / FastWAM Action Cache: Compute Video K/V Once, Recompute Actions Each Step

> **一句话 / In one line**: FastWAM 在视频分支预填每层 K/V，扩散采样时只重算 step-dependent action 分支。 / FastWAM prefills per-layer video K/V once, then recomputes only the step-dependent action branch during diffusion sampling.

## 为什么重要 / Why this matters

中文：WAM 的动作采样通常要走很多 denoising step。如果每一步都把视频 token 重新过一遍 attention，计算量会被固定的视觉上下文反复放大。FastWAM 把视频侧的 K/V 当成只读证据缓存，动作侧仍然按当前 noisy action 和 timestep 重算，再把两边拼到同一个 attention 里。

English: Action denoising in a WAM may require many steps. Recomputing the same video-side attention evidence at every step multiplies the cost of a fixed context. FastWAM treats per-layer video K/V as read-only evidence, recomputes the action branch from the current noisy action and timestep, and joins them in attention.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/mot.py`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py#L370-L472)

```python
    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x
```

## 逐行讲解 / What's happening

1. **第 396-415 行 / Lines 396-415**:
   - 中文: 先检查 action expert、cache 层数和联合 mask 的方形/长度契约，再只截取 action query 对应的 mask 行。
   - English: The method validates the action expert, cache depth, and square joint-mask contract, then keeps only the rows where action tokens are queries.
2. **第 419-438 行 / Lines 419-438**:
   - 中文: 每层仍然重建 action Q/K/V，因为 noisy action 和 timestep 会变；这也是缓存不能把 action 一起冻结的原因。
   - English: Action Q/K/V are rebuilt at every layer because the noisy action and timestep change. That is why the action branch cannot be frozen with the video branch.
3. **第 439-459 行 / Lines 439-459**:
   - 中文: 当前层取出视频 `k/v`，和新的 action `k/v` 沿序列维拼接，action query 通过联合 mask 读取两类 key/value。
   - English: The current layer retrieves video `k/v`, concatenates them with fresh action `k/v` along sequence length, and lets action queries read both through the joint mask.
4. **第 461-472 行 / Lines 461-472**:
   - 中文: attention 输出继续走 block 的 gated residual/MLP 后处理，结果成为下一层的 action tokens。
   - English: Attention output goes through the block’s gated residual/MLP post-processing, producing the action tokens for the next layer.

## 类比 / The analogy

中文：像看着一段固定的监控录像，同时反复修改一条动作脚本。录像的索引卡只需整理一次，脚本每次改稿都要重新读，但两者最后仍要在同一张时间表上对齐。

English: It is like editing an action script against a fixed surveillance clip. The video index cards are prepared once; the script is reread after every edit, but both still align on one shared timeline.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文：这是 `sampler-inference` 的 advanced variant，位于视频 latent prefill 和动作 denoising loop 之间。上游是已经 patchify 的 video tokens、当前 noisy action tokens、RoPE/timestep 条件和联合 attention mask；下游是下一层 action tokens，最终接到动作输出 head。没有缓存时，每个采样 step 都会重复 video expert 的 Q/K/V 和 block 计算，延迟随 denoising steps 线性放大。生产版需要补 cache 生命周期、不同 batch/分辨率的 key、显存复用、混合精度、mask 编译缓存，以及训练模式下不能错误复用带梯度的缓存。

English: This is an advanced `sampler-inference` variant between video-latent prefill and the action denoising loop. Inputs are patchified video tokens, current noisy action tokens, RoPE/timestep conditions, and a joint attention mask; the output is the next-layer action tokens, eventually consumed by the action head. Without the cache, every sampling step repeats video-expert Q/K/V and block work, so latency grows with the number of denoising steps. Production code needs cache lifetime management, batch/resolution keys, memory reuse, mixed precision, compiled-mask reuse, and strict protection against reusing gradient-carrying caches during training.

## 自己跑一遍 / Try it yourself

```python
def prefill_video(video_tokens):
    return {"k": [f"video-k:{x}" for x in video_tokens],
            "v": [f"video-v:{x}" for x in video_tokens]}

def action_step(action_tokens, cache):
    current_k = [f"action-k:{x}" for x in action_tokens]
    current_v = [f"action-v:{x}" for x in action_tokens]
    return list(zip(cache["k"] + current_k, cache["v"] + current_v))

cache = prefill_video(["f0", "f1"])
print(action_step(["a0"], cache))
print(action_step(["a1"], cache))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[('video-k:f0', 'video-v:f0'), ('video-k:f1', 'video-v:f1'), ('action-k:a0', 'action-v:a0')]
[('video-k:f0', 'video-v:f0'), ('video-k:f1', 'video-v:f1'), ('action-k:a1', 'action-v:a1')]
```

中文：两次 action step 看到的是同一份 video cache，但 action K/V 随当前动作变化；这就是“固定上下文 + 可变查询”的核心。

English: Both action steps reuse the same video cache while their action K/V changes with the current action. That is the essence of a fixed context with a changing query branch.

## 注意事项 / Caveats / when it breaks

- **cache 必须按 layer 对齐** / **The cache must align by layer**: 第 `i` 层不能读取别的层产生的 K/V。
- **video sequence length 是联合序列的前缀长度** / **`video_seq_len` is the joint-sequence prefix length**: 错一位会让 mask 行和 K/V 拼接错位。
- **action branch 不能缓存成常量** / **The action branch cannot be cached as a constant**: noisy action、timestep 或 action context 变化时必须重算。

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM video KV cache** / **FastWAM video KV cache**: 早期路径同样利用视频 prefill + action decode 的结构。
- **openpi prefix cache** / **openpi prefix cache**: VLA 中把静态语言/视觉前缀缓存起来，反复推进动作 suffix。
- **LLM KV-cache decoding** / **LLM KV-cache decoding**: token-by-token decode 复用历史 key/value，只计算最新 query。

## 延伸阅读 / Further reading

- [FastWAM MoT implementation](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py)
- [FastWAM video DiT helpers](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/wan_video_dit.py)
