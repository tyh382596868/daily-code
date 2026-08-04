---
date: 2026-08-04
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L57-L90
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, embodiment]
build_role: action-conditioning advanced variant, embodiment-specific action token encoder
---

# DreamZero action encoder：同一个 WAM 要听懂多种机器人 / DreamZero Action Encoder: One WAM Needs to Understand Many Robots

> **一句话 / In one line**: `MultiEmbodimentActionEncoder` 用 embodiment-specific linear layers 把不同机器人的 action 加上 timestep 后投影到统一 hidden space。 / `MultiEmbodimentActionEncoder` uses embodiment-specific linear layers to project each robot's actions plus timestep into one shared hidden space.

## 为什么重要 / Why this matters

World Action Model 如果只服务一种机器人，action encoder 可以很简单。但多 embodiment 训练里，不同机器人的 action 维度语义、关节顺序和控制尺度都可能不同。这里的做法是共享模块形状，但按 `cat_ids` 选择不同的线性权重，把“机器人身份”放进 action conditioning 的第一步。

For a World Action Model serving one robot, the action encoder can be simple. In multi-embodiment training, action dimensions, joint order, and control scales may differ across robots. This module shares the outer shape but selects category-specific linear weights by `cat_ids`, putting robot identity into the first step of action conditioning.

## 代码 / The code

`dreamzero0/dreamzero` — [`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L57-L90)

```python
class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x
```

## 逐行讲解 / What's happening

1. **第 64-66 行 / Lines 64-66 (category-specific projections)**:
   - 中文: `W1/W2/W3` 都是按 embodiment 选择权重的线性层，同一个 batch 里可以有不同机器人。
   - English: `W1/W2/W3` are linear layers whose weights are selected by embodiment, so one batch can contain different robots.
2. **第 79-82 行 / Lines 79-82 (action + time)**:
   - 中文: action 先投到 hidden width，timestep 用 sinusoidal encoding 变成同宽条件。
   - English: Actions are projected to hidden width, while the timestep becomes a sinusoidal conditioning vector of the same width.
3. **第 85-90 行 / Lines 85-90 (fusion MLP)**:
   - 中文: action embedding 和 time embedding 拼接后再过两层 category-specific MLP，输出可送入 DiT/WAM 的 token。
   - English: Action and time embeddings are concatenated, then passed through a two-layer category-specific MLP to produce tokens for the DiT/WAM.

## 类比 / The analogy

这像一个万能充电站：插座外观看起来一样，但内部会根据手机、相机或笔记本选择不同电压曲线，最后都输出设备能用的稳定电流。

It is like a universal charging station: the plug looks shared, but internally it selects a different voltage curve for a phone, camera, or laptop, then outputs stable power each device can use.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `action-conditioning` 的 advanced variant。上游是动作序列、扩散/流匹配 timestep 和机器人类别；下游是视频 DiT 的 action token stream。省掉 embodiment-specific 层，多机器人数据会被迫共享一套动作语义，容易把“夹爪开合”和“底盘速度”混成同一种维度。生产级实现还要维护每种 embodiment 的 action schema、normalizer 和缺失维度 mask。

This is an advanced variant of `action-conditioning`. Upstream are action sequences, diffusion or flow-matching timesteps, and robot category ids; downstream is the action token stream inside the video DiT. Without embodiment-specific layers, multi-robot data must share one action semantics and may confuse gripper, base, and arm dimensions. A production version also needs action schemas, normalizers, and missing-dimension masks per embodiment.

## 自己跑一遍 / Try it yourself

```python
WEIGHTS = {
    "arm": 2.0,
    "mobile": 0.5,
}

def encode(actions, timestep, robot):
    scale = WEIGHTS[robot]
    time = [timestep, 1 - timestep]
    out = []
    for a0, a1 in actions:
        action_emb = [a0 * scale, a1 * scale]
        out.append(action_emb + time)
    return out

print(encode([(1, 2)], 0.25, "arm"))
print(encode([(1, 2)], 0.25, "mobile"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[2.0, 4.0, 0.25, 0.75]]
[[0.5, 1.0, 0.25, 0.75]]
```

中文: 同样的原始 action，会因为 embodiment 不同进入不同的隐藏空间位置。
English: The same raw action lands in a different hidden-space location depending on the embodiment.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GR00T embodiment tags** / **GR00T embodiment tags**: 多机器人训练也需要显式告诉模型当前身体是谁。 / Multi-robot training also needs an explicit identity for the current body.
- **Action normalizers** / **Action normalizers**: 不同机器人通常先各自归一化，再进入共享模型。 / Different robots usually normalize actions separately before sharing a model.

## 注意事项 / Caveats / when it breaks

- **`cat_ids` 错就是错身体** / **Wrong `cat_ids` means wrong body**: 类别 id 对不上时，动作会过错误权重。 / If category ids are wrong, actions pass through the wrong weights.
- **不是 schema 的替代品** / **Not a schema replacement**: 它能选择权重，但仍需要上游保证 action 维度语义一致。 / It selects weights, but upstream still must define action dimension semantics.

## 延伸阅读 / Further reading

- DreamZero action encoder source: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L57-L90
