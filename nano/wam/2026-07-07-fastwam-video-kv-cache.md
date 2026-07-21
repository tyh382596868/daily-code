---
date: 2026-07-07
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/mot.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/mot.py#L257-L445
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, kv-cache]
build_role: sampler-inference advanced variant
---

# FastWAM video KV cache：视频先预填，动作逐步解码 / FastWAM Video KV Cache: Prefill Video Once, Decode Actions Step by Step

> **一句话 / In one line**: FastWAM 先为视频分支缓存每层 K/V，再让动作 query 每个去噪步只拼接缓存的视频 K/V 和当前动作 K/V。 / FastWAM caches per-layer video K/V first, then each action denoising step concatenates cached video K/V with current action K/V.

## 为什么重要 / Why this matters

WAM 推理里，视频上下文通常不随动作去噪步改变。如果每一步都重算视频分支，会像 LLM decode 时每个 token 都重算 prompt 一样浪费。FastWAM 的 MoT cache 把视频分支当成 prefix prefill，动作分支当成 decode，从而减少重复计算。

In WAM inference, video context often stays fixed across action denoising steps. Recomputing the video branch every step is like recomputing the full prompt for each LLM decode token. FastWAM's MoT cache treats video as prefix prefill and actions as decode, reducing repeated work.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/mot.py`](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/mot.py#L257-L445)

```python
def prefill_video_cache(self, video_tokens, video_freqs, video_t_mod, video_context_payload, video_attention_mask):
    if "video" not in self.mixtures:
        raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
    if video_attention_mask.ndim != 2 or video_attention_mask.shape[0] != video_attention_mask.shape[1]:
        raise ValueError("`video_attention_mask` must be square")

    expert = self.mixtures["video"]
    x = video_tokens
    kv_cache: list[dict[str, torch.Tensor]] = []
    for layer_idx in range(self.num_layers):
        block = expert.blocks[layer_idx]
        q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing = \
            self._build_expert_attention_io(expert=expert, block=block, x=x, freqs=video_freqs, t_mod=video_t_mod)
        mixed = self._mixed_attention(q_cat=q, k_cat=k, v_cat=v, attention_mask=video_attention_mask)
        x = self._apply_post_with_optional_checkpoint(
            block=block, residual_x=residual_x, gate_msa=gate_msa,
            shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed, context_payload=video_context_payload,
        )
        kv_cache.append({"k": k, "v": v})
    return kv_cache


def forward_action_with_video_cache(self, action_tokens, action_freqs, action_t_mod, action_context_payload,
                                    video_kv_cache, attention_mask, video_seq_len):
    if len(video_kv_cache) != self.num_layers:
        raise ValueError(f"`video_kv_cache` must contain {self.num_layers} layers")
    action_seq_len = int(action_tokens.shape[1])
    total_seq_len = int(video_seq_len) + action_seq_len
    action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

    expert = self.mixtures["action"]
    x = action_tokens
    for layer_idx in range(self.num_layers):
        block = expert.blocks[layer_idx]
        q_action, k_action, v_action, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing = \
            self._build_expert_attention_io(expert=expert, block=block, x=x, freqs=action_freqs, t_mod=action_t_mod)
        k_video = video_kv_cache[layer_idx]["k"]
        v_video = video_kv_cache[layer_idx]["v"]
        k_cat = torch.cat([k_video, k_action], dim=1)
        v_cat = torch.cat([v_video, v_action], dim=1)
        mixed = self._mixed_attention(q_cat=q_action, k_cat=k_cat, v_cat=v_cat, attention_mask=action_attention_mask)
        x = self._apply_post_with_optional_checkpoint(
            block=block, residual_x=residual_x, gate_msa=gate_msa,
            shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed, context_payload=action_context_payload,
        )
    return x
```

## 逐行讲解 / What's happening

1. **第 1-6 行 / Lines 1-6 (prefill validation)**:
   - 中文: 视频 attention mask 必须是方阵，因为它只描述视频 token 之间的可见性。
   - English: the video attention mask must be square because it only describes visibility among video tokens.
2. **第 9-24 行 / Lines 9-24 (layer cache)**:
   - 中文: 每层都先跑视频 expert，保存该层的 `k/v`，同时更新视频 token 到下一层。
   - English: each layer runs the video expert, saves that layer's `k/v`, and updates video tokens for the next layer.
3. **第 28-32 行 / Lines 28-32 (action rows)**:
   - 中文: 联合 mask 里只取 action query 对应的行，因为当前 forward 只更新 action token。
   - English: only action-query rows are used from the joint mask because this forward updates action tokens only.
4. **第 38-41 行 / Lines 38-41 (concat K/V)**:
   - 中文: action query 可以看缓存视频 K/V，也可以看当前 action K/V。
   - English: action queries can attend to cached video K/V and current action K/V.
5. **第 42-50 行 / Lines 42-50 (post block)**:
   - 中文: attention 输出仍走 action expert 自己的 post block，不污染视频分支。
   - English: the attention output goes through the action expert's own post block, leaving the video branch untouched.

## 类比 / The analogy

像开卷考试：视频上下文是提前整理好的资料夹，动作去噪每一步只翻资料夹再写当前答案，不需要每次重新整理整本资料。

It is like an open-book exam: the video context is a prepared folder, and each action denoising step consults that folder while writing the current answer instead of rebuilding the folder.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，这是 `sampler-inference` 的高级优化。上游视频 encoder/DiT 产生固定视频 token；动作 sampler 多步更新动作 token；cache 层夹在两者之间，避免每个动作步重复视频 forward。生产实现还要处理 batch 内不同视频长度、cache dtype、offload 和失效条件。

In nanoWAM, this is an advanced `sampler-inference` optimization. Upstream video encoder/DiT produces fixed video tokens; the action sampler repeatedly updates action tokens; the cache sits between them to avoid repeated video forwards. A production version must handle variable video lengths, cache dtype, offload, and invalidation.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
video_k = np.ones((1, 3, 2))
action_k = np.full((1, 2, 2), 2.0)
full_k = np.concatenate([video_k, action_k], axis=1)
joint_mask = np.ones((5, 5), dtype=bool)
action_rows = joint_mask[3:5, :5]
print(full_k[:, :, 0].tolist())
print(action_rows.shape)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[1.0, 1.0, 1.0, 2.0, 2.0]]
(2, 5)
```

它展示了 action query 的 K/V 视野：前三个来自缓存视频，后两个来自当前动作。

It shows the action query's K/V view: the first three entries come from cached video, the last two from current action.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LLM KV cache** / **LLM KV cache**: prompt prefill 一次，decode token 逐步复用 prefix K/V。 / Prefill the prompt once, then reuse prefix K/V while decoding.
- **Video diffusion latent cache** / **Video diffusion latent cache**: 条件帧或参考帧固定时，也可以缓存编码结果。 / Fixed conditioning or reference frames can also cache encoded results.

## 注意事项 / Caveats / when it breaks

- **cache 必须逐层对齐** / **The cache must align per layer**: 层数或顺序不一致会把错误 K/V 喂给 action block。 / Mismatched layer count or order feeds wrong K/V to the action block.
- **视频条件变化会使 cache 失效** / **Changing video context invalidates the cache**: 新帧、新 mask、新文本条件都要重新 prefill。 / New frames, masks, or text conditions require a fresh prefill.

## 延伸阅读 / Further reading

- [FastWAM MoT cache](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/mot.py)
