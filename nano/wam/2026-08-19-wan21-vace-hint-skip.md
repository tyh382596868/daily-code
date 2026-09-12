---
date: 2026-08-19
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vace_model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vace_model.py#L10-L62
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, control-hints]
build_role: action-conditioning advanced variant
---

# Wan2.1 VACE hints：控制分支用零初始化慢慢接入 / Wan2.1 VACE Hints: Add the Control Branch with Zero Init

> **一句话 / In one line**: VACE 给 Wan attention block 加一条控制/提示分支，零初始化投影让新分支从“不影响主干”开始学习。 / VACE adds a control/hint branch to Wan attention blocks, with zero-initialized projections so the new branch starts by not perturbing the backbone.

## 为什么重要 / Why this matters

WAM 的控制信号不一定只是文本或动作 token，也可能是编辑 hint、参考帧、mask、草图或外部状态。直接把新条件强塞进主干会破坏预训练模型。VACE 的做法是把控制流做成旁路：第一层可把 hint 和主干 latent 对齐，之后每层产出 `c_skip`，主干层再按 block id 把 hint 加回来。

WAM conditioning is not always just text or action tokens; it can be edit hints, reference frames, masks, sketches, or external state. Injecting a new condition directly into a pretrained backbone can destabilize it. VACE treats the control stream as a side branch: the first block aligns hints with the latent stream, each branch block emits `c_skip`, and base blocks add hints back by block id.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vace_model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vace_model.py#L10-L62)

```python
class VaceWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
```

## 逐行讲解 / What's happening

1. **第 25-31 行 / Lines 25-31 (zero init)**:
   - 中文: `before_proj` 和 `after_proj` 都零初始化，新分支刚接上时不会立刻改变主干输出。
   - English: `before_proj` and `after_proj` are zero-initialized, so the newly attached branch initially leaves the backbone unchanged.
2. **第 33-39 行 / Lines 33-39 (control block)**:
   - 中文: 第一个 block 把控制 token 投影后加到主干 `x` 上，再走标准 Wan attention block，最后吐出 skip hint。
   - English: The first block aligns control tokens with the backbone stream, runs the standard Wan attention block, then emits a skip hint.
3. **第 58-62 行 / Lines 58-62 (base block injection)**:
   - 中文: 主干 block 正常 forward 后，如果有对应 `block_id`，就把 hint 按 `context_scale` 加回去。
   - English: The base block runs normally first, then adds the corresponding hint scaled by `context_scale` when a `block_id` is assigned.

## 类比 / The analogy

像给老楼加一部外置电梯。电梯井先独立搭好，入口和每层楼的连接从封闭状态开始，确认安全后再逐步开放；老楼主体不需要一开始就被拆开重造。

It is like adding an external elevator to an old building. The shaft is built as a side structure, floor connections start closed, and access opens gradually after validation; the original building is not torn apart on day one.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoWAM 里，这是 `action-conditioning` 的高级变体：把动作、mask、参考图或编辑条件变成 hint stream，再在若干 DiT block 注入。上游是条件 encoder 和 patchify 后的视频 latent；下游是主干 Wan/DiT block。省掉它，所有控制都只能走文本 cross-attention，局部编辑或强结构条件会很弱。生产级实现要补 hint 的时空对齐、哪些层注入、`context_scale` 调度、以及多条件冲突时的优先级。

In a nanoWAM, this is an advanced `action-conditioning` variant: encode actions, masks, references, or edit controls as a hint stream, then inject them into selected DiT blocks. Upstream are condition encoders and patchified video latents; downstream are the main Wan/DiT blocks. Without this path, all control must pass through text cross-attention, which is weak for local edits or hard structural constraints. A production implementation needs spatial-temporal alignment, layer selection, `context_scale` scheduling, and conflict handling across multiple conditions.

## 自己跑一遍 / Try it yourself

```python
def zero_linear(v):
    return [0.0 for _ in v]

def vace_block(c, x, block_id):
    if block_id == 0:
        c = [a + b for a, b in zip(zero_linear(c), x)]
    c = [v + 1.0 for v in c]   # stand-in for WanAttentionBlock
    c_skip = zero_linear(c)
    return c, c_skip

def base_block(x, hints, block_id, context_scale=1.0):
    x = [v * 2 for v in x]     # stand-in for WanAttentionBlock
    if block_id is not None:
        x = [a + context_scale * b for a, b in zip(x, hints[block_id])]
    return x

c, hint = vace_block([9, 9], [1, 2], 0)
print(c, hint)
print(base_block([1, 2], {0: hint}, 0))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[2.0, 3.0] [0.0, 0.0]
[2.0, 4.0]
```

中文: 零初始化让 hint 一开始是 0，所以主干输出不被突然扰动。

English: Zero initialization makes the first hint zero, so the backbone output is not suddenly perturbed.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ControlNet** / **ControlNet**: 也用零初始化旁路把控制条件接到扩散主干上。 / It also uses zero-initialized side branches to attach controls to a diffusion backbone.
- **T2I-Adapter** / **T2I-Adapter**: 同样把外部结构条件编码成中间特征，再注入生成网络。 / It similarly encodes structural controls into intermediate features before injection.

## 注意事项 / Caveats / when it breaks

- **hint 尺寸必须对齐** / **Hint shapes must align**: `hints[self.block_id]` 必须和 `x` 同 shape，否则加法不是语义错误而是直接形状错误。 / `hints[self.block_id]` must have the same shape as `x`; otherwise the injection fails as a shape error, not a subtle semantic bug.
- **零初始化不是永久关闭** / **Zero init is not permanent disablement**: 训练后这些投影会学到非零权重，推理时要保存和加载对应参数。 / After training, these projections become nonzero and must be saved and loaded.

## 延伸阅读 / Further reading

- [Wan2.1 VACE model source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vace_model.py#L10-L62)
- [ControlNet paper](https://arxiv.org/abs/2302.05543)
