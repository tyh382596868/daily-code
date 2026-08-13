---
date: 2026-08-13
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L381-L553
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, attention]
build_role: dit-block advanced variant, blockwise causal image/action/state attention
---

# DreamZero blockwise flash attention：视频块只看该看的控制信号 / DreamZero Blockwise Flash Attention: Video Blocks See Only the Right Control Signals

> **一句话 / In one line**: `_blockwise_causal_flash_attn()` 把序列切成 first image、image blocks、action blocks、state blocks，再为每类 query 拼接不同 K/V 上下文。 / `_blockwise_causal_flash_attn()` splits the sequence into first image, image blocks, action blocks, and state blocks, then builds different K/V contexts for each query type.

## 为什么重要 / Why this matters

WAM 不是普通视频模型：动作和状态 token 不能随便和所有未来帧互看，否则模型会偷看答案。DreamZero 把“谁能看谁”落成块级 K/V 拼接，保住 causal 语义。

A WAM is not a plain video model: action and state tokens cannot freely attend to future frames, or the model leaks answers. DreamZero turns “who may see whom” into block-level K/V assembly that preserves causal semantics.

## 代码 / The code

`dreamzero0/dreamzero` — [`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L381-L553)

```python
# Multi-modal structure: [first image] [image blocks] [action blocks] [state blocks]
# Calculate block structure
first_image_len = frame_seqlen
action_len = action_horizon
state_len = state_horizon
image_blocks_len = total_len - first_image_len - action_len - state_len

num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
num_action_blocks = action_horizon // num_action_per_block
num_state_blocks = state_horizon // num_state_per_block

assert num_image_blocks == num_action_blocks == num_state_blocks

# Token ranges
first_image_start = 0
first_image_end = first_image_len
image_blocks_start = first_image_end
image_blocks_end = image_blocks_start + image_blocks_len
action_start = image_blocks_end
action_end = action_start + action_len
state_start = action_end
state_end = state_start + state_len

# Visualize attention mask if requested
if visualize_mask:
    mask = self._visualize_attention_mask(
        total_len, first_image_len, image_blocks_len,
        action_len, state_len, num_image_blocks,

# ... visualization branch omitted ...
# OPTIMIZED: Pre-allocate output tensor and pre-compute all indices
output = torch.empty_like(q)

# Process first image (conditioning, can only self-attend)
output[:, first_image_start:first_image_end] = self.attn(
    q[:, first_image_start:first_image_end],
    k[:, first_image_start:first_image_end],
    v[:, first_image_start:first_image_end]
)

# Pre-compute all block indices for image blocks
image_block_starts = [image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
image_block_ends = [image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
if self.local_attn_size != -1:
    image_kv_starts = [max(image_blocks_start, end - self.local_attn_size * frame_seqlen) for end in image_block_ends]
else:
    image_kv_starts = [image_blocks_start] * num_image_blocks

# Pre-compute action and state block indices
action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
action_block_ends = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
state_block_starts = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
state_block_ends = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]

# Process each image block
for block_idx in range(num_image_blocks):
    block_start = image_block_starts[block_idx]
    block_end = image_block_ends[block_idx]
    image_kv_start = image_kv_starts[block_idx]
    action_block_start = action_block_starts[block_idx]
    action_block_end = action_block_ends[block_idx]
    state_block_start = state_block_starts[block_idx]
    state_block_end = state_block_ends[block_idx]

    # Build context: first image + relevant image blocks + current action + current state
    k_context = torch.cat([
        k[:, first_image_start:first_image_end],  # First image
        k[:, image_kv_start:block_end],  # Image blocks
        k[:, action_block_start:action_block_end],  # Current action block
        k[:, state_block_start:state_block_end]  # Current state block
    ], dim=1)
    v_context = torch.cat([
        v[:, first_image_start:first_image_end],
        v[:, image_kv_start:block_end],
        v[:, action_block_start:action_block_end],
        v[:, state_block_start:state_block_end]
    ], dim=1)

    output[:, block_start:block_end] = self.attn(
        q[:, block_start:block_end], k_context, v_context
    )
```

## 逐行讲解 / What's happening

1. **第 381-392 行 / Lines 381-392: 序列拆成四段，并要求 image/action/state block 数一致。 / The sequence is split into four regions, and image/action/state block counts must match.**
2. **第 394-402 行 / Lines 394-402: start/end 下标成为后续 attention 的边界合同。 / The start/end indices become the boundary contract for later attention calls.**
3. **第 458-466 行 / Lines 458-466: first image 只 self-attend，作为条件锚点。 / The first image only self-attends and acts as the conditioning anchor.**
4. **第 482-508 行 / Lines 482-508: image block 的 K/V 来自 first image、历史/当前 image、当前 action 和当前 state。 / An image block receives K/V from the first image, previous/current images, the matching action, and the matching state.**

## 类比 / The analogy

像剪电影时按场记板分镜：第 3 个镜头只能看开场定场镜、第 1-3 个镜头素材，以及第 3 条动作指令。

It is like editing a film from a shot list: shot 3 may use the establishing shot, shots 1-3, and action instruction 3.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `dit-block` 的 advanced variant，依赖 `patchify-positional`、`action-conditioning` 和 `sampler-inference`。在 nanoWAM 中，它是 video/action/state token 混合后的核心 attention 层；省掉块级因果约束，训练时会泄漏未来视觉 token。

This is an advanced `dit-block` variant depending on `patchify-positional`, `action-conditioning`, and `sampler-inference`. In a nanoWAM, it is the core attention layer after video/action/state token packing; without blockwise causality, training leaks future visual tokens.

## 自己跑一遍 / Try it yourself

```python
def contexts(num_blocks):
    first = ["I0"]
    image = [[f"I{i}"] for i in range(1, num_blocks + 1)]
    action = [[f"A{i}"] for i in range(1, num_blocks + 1)]
    state = [[f"S{i}"] for i in range(1, num_blocks + 1)]
    for i in range(num_blocks):
        ctx = first + sum(image[: i + 1], []) + action[i] + state[i]
        print(f"block {i+1}:", ctx)

contexts(3)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
block 1: ['I0', 'I1', 'A1', 'S1']
block 2: ['I0', 'I1', 'I2', 'A2', 'S2']
block 3: ['I0', 'I1', 'I2', 'I3', 'A3', 'S3']
```

输出里没有未来块，比如 block 2 看不到 `I3/A3/S3`。

There is no future block in the output; for example, block 2 cannot see `I3/A3/S3`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM group-causal attention** / **FastWAM group-causal attention**: 同样用分组约束让动作 token 和视频 token 按时间对齐。 / It also uses grouped constraints to align action and video tokens.
- **Wan2.1 varlen attention** / **Wan2.1 varlen attention**: 同样先整理 token 边界，再交给高效 attention kernel。 / It also prepares token boundaries before calling an efficient attention kernel.

## 注意事项 / Caveats / when it breaks

- **block 数必须一致** / **Block counts must match**: assert 失败说明 token 排布已经坏了。 / Assertion failure means the token layout is already broken.
- **局部窗口会改语义** / **Local windows change semantics**: `local_attn_size` 会限制 image K/V 历史范围。 / `local_attn_size` limits the image K/V history range.

## 延伸阅读 / Further reading

- [dreamzero0/dreamzero source](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L381-L553)
