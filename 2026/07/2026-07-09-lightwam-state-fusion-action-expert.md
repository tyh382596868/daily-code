---
date: 2026-07-09
topic: robotics
source: trending
repo: L1ziang/Light-WAM
file: src/lightwam/models/wan22/state_fusion_action_expert.py
permalink: https://github.com/L1ziang/Light-WAM/blob/b2785f66e13fd9987e94ae1ecc1c441d5059c9ae/src/lightwam/models/wan22/state_fusion_action_expert.py#L229-L445
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, robotics, wam, action-head]
---

# Light-WAM StateFusionActionExpert：从多层视频状态直接预测动作 chunk / Light-WAM StateFusionActionExpert: Predict Action Chunks from Multi-Layer Video States

> **一句话 / In one line**: Light-WAM 把 backbone/adapted/delta 多层 token 池化压缩，再加 step position embedding，一次输出整个 action horizon。 / Light-WAM pools and compresses multi-layer backbone/adapted/delta tokens, adds step position embeddings, and predicts a full action horizon in one pass.

## 为什么重要 / Why this matters

很多 WAM/VLA 系统用重型 diffusion action expert 逐步去噪动作。Light-WAM 的这个模块走另一条路：直接读视频 backbone 的多层状态，用 MLP trunk 预测动作 chunk。它牺牲了一部分生成式灵活性，换来低延迟和低显存。

Many WAM/VLA systems use a heavy diffusion action expert that denoises actions iteratively. This Light-WAM module takes another route: read multi-layer video-backbone states and predict the action chunk with an MLP trunk. It trades some generative flexibility for lower latency and memory.

## 代码 / The code

`L1ziang/Light-WAM` — [`src/lightwam/models/wan22/state_fusion_action_expert.py`](https://github.com/L1ziang/Light-WAM/blob/b2785f66e13fd9987e94ae1ecc1c441d5059c9ae/src/lightwam/models/wan22/state_fusion_action_expert.py#L229-L445)

```python
class StateFusionActionExpert(nn.Module):
    """Direct action predictor over pooled multi-layer backbone/adapter states."""

    def __init__(self, video_hidden_dim: int, action_dim: int, num_fusion_layers: int, ...):
        super().__init__()
        ...
        global_feature_sources = _normalize_feature_source_spec(feature_sources)
        if layer_feature_sources is None:
            self.layer_feature_sources = tuple(
                global_feature_sources for _ in range(self.num_fusion_layers)
            )
        else:
            ...
        self.layer_input_dims = [
            self.pooler_output_dim * len(source_names)
            for source_names in self.layer_feature_sources
        ]
        self.fused_input_dim = self.per_layer_dim * self.num_fusion_layers

        self.layer_poolers = nn.ModuleList()
        for source_names in self.layer_feature_sources:
            source_poolers = nn.ModuleDict()
            if self.token_pooling_type == "learned_query":
                for source_name in source_names:
                    source_poolers[source_name] = LearnedQueryPooler(...)
            self.layer_poolers.append(source_poolers)

        self.layer_compressors = nn.ModuleList(
            [LayerFusionCompressor(in_dim=layer_input_dim, out_dim=self.per_layer_dim)
             for layer_input_dim in self.layer_input_dims]
        )
        self.fused_norm = nn.LayerNorm(self.fused_input_dim)
        self.fused_proj = nn.Linear(self.fused_input_dim, self.trunk_dim)
        self.trunk = nn.ModuleList(
            [ResidualMLPBlock(self.trunk_dim) for _ in range(self.num_trunk_blocks)]
        )
        self.step_pos_proj = nn.Sequential(...)
        self.output = nn.Sequential(
            nn.Linear(self.trunk_dim, self.trunk_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.trunk_dim, self.action_dim),
        )

    def forward(self, layer_states: Sequence[dict[str, Any]], action_horizon: int) -> torch.Tensor:
        compressed = []
        batch_size = None
        for idx, (layer_state, compressor) in enumerate(zip(layer_states, self.layer_compressors)):
            pooled_sources = []
            expected_sources = self.layer_feature_sources[idx]
            for source_name in expected_sources:
                pooled_sources.append(
                    self._pool_source_tokens(idx, source_name, layer_state[source_name])
                )
            feature = torch.cat(pooled_sources, dim=-1)
            compressed.append(compressor(feature))

        fused = torch.cat(compressed, dim=-1)
        state = self.fused_proj(self.fused_norm(fused))
        for block in self.trunk:
            state = block(state)

        positions = torch.arange(action_horizon, device=state.device, dtype=state.dtype)
        step_pos = sinusoidal_embedding_1d(self.step_pos_dim, positions)
        step_tokens = state.unsqueeze(1) + self.step_pos_proj(step_pos).unsqueeze(0)
        return self.output(self.output_norm(step_tokens))
```

## 逐行讲解 / What's happening

1. **第 317-339 行 / Lines 317-339 (source selection)**:
   - 中文: 每层可以选择 `backbone`、`adapted`、`delta` 中的一个或多个来源，输入维度随来源数量变化。
   - English: Each layer can select one or more of `backbone`, `adapted`, and `delta`, so input width follows the number of sources.
2. **第 345-358 行 / Lines 345-358 (poolers)**:
   - 中文: token pooling 可用 mean，也可用 learned queries；后者让模型自己挑哪些空间/时间 token 重要。
   - English: Token pooling can be mean pooling or learned-query pooling; the latter lets the model decide which space/time tokens matter.
3. **第 360-379 行 / Lines 360-379 (compress + trunk)**:
   - 中文: 每层先压到固定宽度，再拼接成一个全局 state，经 residual MLP trunk 融合。
   - English: Each layer is compressed to a fixed width, concatenated into one global state, and fused by a residual MLP trunk.
4. **第 439-445 行 / Lines 439-445 (horizon output)**:
   - 中文: `action_horizon` 的每个时间步加一份 sinusoidal position，然后一次输出 `[B, H, action_dim]`。
   - English: Each step in `action_horizon` receives a sinusoidal position embedding, producing `[B, H, action_dim]` in one pass.

## 类比 / The analogy

这像一个驾驶教练同时看仪表盘、后视镜和路线图：每个来源先总结成一句话，再合成驾驶意图，最后按未来几秒逐步写出方向盘和油门动作。

It is like a driving coach reading the dashboard, mirrors, and route map. Each source is summarized, fused into driving intent, then expanded into steering and throttle commands for the next few seconds.

## 自己跑一遍 / Try it yourself

```python
import math

layers = [
    {"backbone": [1, 2], "adapted": [2, 4], "delta": [1, 2]},
    {"backbone": [3, 1], "adapted": [4, 2], "delta": [1, 1]},
]
state = []
for layer in layers:
    pooled = [sum(v) / len(v) for v in layer.values()]
    state.append(sum(pooled))
for t in range(3):
    print(round(state[0] + state[1] + math.sin(t), 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
12.0
12.841
12.909
```

这个小例子把多层、多来源池化成一个 state，再用时间位置生成每一步动作。

This toy example pools multiple sources across layers into one state, then uses step position to generate per-step actions.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLA action chunking** / **SmolVLA action chunking**: 也一次生成一段动作，只是上游表示和 head 不同。 / It also predicts an action chunk, but with a different representation and head.
- **FastWAM video KV cache** / **FastWAM video KV cache**: 复用视频状态供动作生成读取。 / It also reuses video state for action generation.

## 注意事项 / Caveats / when it breaks

- **直接 head 少了迭代修正** / **Direct heads lose iterative refinement**: 它快，但不像 diffusion head 那样逐步修正动作分布。 / It is fast, but lacks the iterative refinement of diffusion heads.
- **layer/source 配置必须对齐** / **Layer/source config must align**: `layer_states` 数量和 key 必须匹配配置，否则会直接报错。 / The number of `layer_states` and their keys must match the configuration.

## 延伸阅读 / Further reading

- [Light-WAM `StateFusionActionExpert`](https://github.com/L1ziang/Light-WAM/blob/b2785f66e13fd9987e94ae1ecc1c441d5059c9ae/src/lightwam/models/wan22/state_fusion_action_expert.py#L229-L445)
