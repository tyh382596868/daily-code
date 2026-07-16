---
date: 2026-07-16
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L365-L567
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, dreamzero, attention, action-conditioning]
build_role: action-conditioning advanced variant
---

# DreamZero blockwise attention：每个视频块只看对应动作和状态 / DreamZero Blockwise Attention: Each Video Block Sees Its Matching Action and State

> **一句话 / In one line**: DreamZero 把序列拆成 first image、image blocks、action blocks、state blocks，并为每类 query 手动拼出允许看的 KV 上下文。 / DreamZero splits the sequence into first image, image blocks, action blocks, and state blocks, then manually builds the allowed KV context for each query type.

## 为什么重要 / Why this matters

WAM 不是普通视频 DiT：它要让视频 token 看到动作和状态，但不能随便泄漏未来。这里的实现没有只依赖一个大 mask，而是按 block 预计算边界，再为 image/action/state 三类 token 分别调用 attention。这样更容易表达“第 i 个视频块只能看第 i 个动作/状态块和过去图像”。

A WAM is not a plain video DiT: video tokens should see actions and state, but future leakage must be controlled. This implementation does not rely only on one giant mask. It precomputes block boundaries and runs separate attention calls for image/action/state tokens, making the rule "block i sees action/state i plus previous images" explicit.

## 代码 / The code

`dreamzero0/dreamzero` — [`wan_video_dit_action_casual_chunk.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L365-L567)

```python
# Multi-modal structure: [first image] [image blocks] [action blocks] [state blocks]
first_image_len = frame_seqlen
action_len = action_horizon
state_len = state_horizon
image_blocks_len = total_len - first_image_len - action_len - state_len

num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
num_action_blocks = action_horizon // num_action_per_block
num_state_blocks = state_horizon // num_state_per_block
assert num_image_blocks == num_action_blocks == num_state_blocks

first_image_start = 0
first_image_end = first_image_len
image_blocks_start = first_image_end
image_blocks_end = image_blocks_start + image_blocks_len
action_start = image_blocks_end
state_start = action_start + action_len

output = torch.empty_like(q)
output[:, first_image_start:first_image_end] = self.attn(
    q[:, first_image_start:first_image_end],
    k[:, first_image_start:first_image_end],
    v[:, first_image_start:first_image_end],
)

for block_idx in range(num_image_blocks):
    block_start = image_block_starts[block_idx]
    block_end = image_block_ends[block_idx]
    action_block_start = action_block_starts[block_idx]
    action_block_end = action_block_ends[block_idx]
    state_block_start = state_block_starts[block_idx]
    state_block_end = state_block_ends[block_idx]

    k_context = torch.cat([
        k[:, first_image_start:first_image_end],
        k[:, image_kv_start:block_end],
        k[:, action_block_start:action_block_end],
        k[:, state_block_start:state_block_end],
    ], dim=1)
    output[:, block_start:block_end] = self.attn(q[:, block_start:block_end], k_context, v_context)
```

## 逐行讲解 / What's happening

1. **四段序列 / Four sequence regions**: 中文: first image 是全局条件，后面依次是视频块、动作块、状态块。 English: first image is conditioning, followed by image blocks, action blocks, and state blocks.
2. **block 数必须一致 / Block counts must match**: 中文: `assert` 保证第 i 个 image/action/state block 可以一一配对。 English: the `assert` guarantees blockwise pairing across image, action, and state.
3. **first image 自注意力 / First image self-attends**: 中文: 条件帧不读动作或未来图像。 English: the conditioning frame does not read action or future image context.
4. **image query 拼上下文 / Image queries build context**: 中文: 视频块看 first image、允许的历史图像、当前动作、当前状态。 English: each image block sees first image, allowed image history, current action, and current state.
5. **action/state 也分开处理 / Action and state are also separate**: 中文: 后续代码给 action block 和 state block 用不同上下文规则。 English: later code gives action and state blocks their own context rules.

## 类比 / The analogy

这像剪辑一段机器人视频。每个镜头可以看剧本开头、之前镜头、这一镜头对应的动作指令和状态记录，但不能偷看后面镜头的动作答案。

It is like editing a robot video. Each shot may see the opening scene, previous shots, and that shot's action/state notes, but it cannot peek at future action answers.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `action-conditioning` 的高级变体，位于 DiT self-attention 内部。nanoWAM 如果要联合建模视频、动作和状态，需要明确每类 token 的可见性：视频能不能看动作、动作能不能看视频、状态是否只是条件。

English: This is an advanced `action-conditioning` component inside DiT self-attention. A nanoWAM that jointly models video, actions, and state needs explicit visibility rules for every token type: whether video sees actions, actions see video, and whether state is conditioning only.

## 自己跑一遍 / Try it yourself

```python
first = ["I0"]
images = ["I1", "I2"]
actions = ["A1", "A2"]
states = ["S1", "S2"]

for i, image in enumerate(images):
    ctx = first + images[: i + 1] + [actions[i], states[i]]
    print(image, "<-", ctx)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
I1 <- ['I0', 'I1', 'A1', 'S1']
I2 <- ['I0', 'I1', 'I2', 'A2', 'S2']
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **causal video attention** / **causal video attention**: 视频块只能看过去和当前。 / Video blocks see only past and current context.
- **prefix-LM VLA masks** / **prefix-LM VLA masks**: 不同 token 区域有不同可见性。 / Different token regions get different visibility rules.

## 注意事项 / Caveats / when it breaks

- **block 对齐是硬前提 / Block alignment is required**: image/action/state block 数不等会直接失败。 / Unequal image/action/state block counts fail immediately.
- **`torch.cat` 上下文有成本 / Concatenating context has a cost**: 规则清晰，但会创建临时 KV。 / The rule is clear but temporary KV tensors cost memory.
- **local attention 会改变历史窗口 / Local attention changes history**: `local_attn_size` 限制 image history 范围。 / `local_attn_size` bounds the image-history window.

## 延伸阅读 / Further reading

- [DreamZero blockwise attention file](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py)
- [DreamZero repository](https://github.com/dreamzero0/dreamzero)
