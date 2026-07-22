---
date: 2026-07-02
topic: vla
source: vla
repo: nv-tlabs/Gamma-World
file: gamma_world/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py
permalink: https://github.com/nv-tlabs/Gamma-World/blob/6a95de85c439d8ea73eae34c88fbfd4e89ea02e2/gamma_world/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py#L107-L160
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-normalization]
build_role: action preprocessing variant for a from-scratch VLA data pipeline
---

# Gamma-World 动作归一化：零方差维度也要可训练 / Gamma-World Action Normalization: Keep Zero-Variance Dimensions Trainable

> **一句话 / In one line**: 归一化不只是套公式；代码专门处理 `q01==q99`、`std==0`、`min==max` 这些真实机器人数据里的坏维度。 / Normalization is not just a formula; this code handles `q01==q99`, `std==0`, and `min==max`, which show up in real robot logs.

## 为什么重要 / Why this matters

VLA 的动作向量经常混合位置、旋转、夹爪、模式 bit。某些维度在一个数据集里可能完全不动，如果直接除以零，训练会炸；如果随便丢掉，又会破坏动作 schema。Gamma-World 的 `Normalizer` 把这些边界情况放在数据层集中处理。

VLA action vectors often mix position, rotation, gripper, and mode bits. Some dimensions may be constant in a dataset; dividing by zero breaks training, while dropping them breaks the action schema. Gamma-World's `Normalizer` centralizes these edge cases in the data layer.

## 代码 / The code

`nv-tlabs/Gamma-World` — [`state_action.py`](https://github.com/nv-tlabs/Gamma-World/blob/6a95de85c439d8ea73eae34c88fbfd4e89ea02e2/gamma_world/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py#L107-L160)

```python
class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            mask = q01 != q99
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask])
            normalized[..., mask] = 2 * normalized[..., mask] - 1
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            mask = std != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            mask = min != max
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (max[..., mask] - min[..., mask])
            normalized[..., mask] = 2 * normalized[..., mask] - 1
            normalized[..., ~mask] = 0
```

## 逐行讲解 / What's happening

1. **第 107-114 行 / Lines 107-114**: 中文: 统计量在初始化时转成 tensor，后续能跟 batch tensor 做索引和 dtype 对齐。 / English: Statistics become tensors at initialization so later code can index them and match the batch dtype.
2. **第 119-128 行 / Lines 119-128**: 中文: 分位数归一化把范围映射到 `[-1, 1]`，常量维度保持原值。 / English: Quantile normalization maps values to `[-1, 1]`, while constant dimensions keep their original values.
3. **第 130-139 行 / Lines 130-139**: 中文: mean/std 模式只除以非零 std，零方差维度不制造 NaN。 / English: Mean/std mode divides only by nonzero standard deviations, avoiding NaNs on constant dimensions.
4. **第 141-153 行 / Lines 141-153**: 中文: min/max 模式里常量维度被置 0，表示“归一化空间的中性值”。 / English: In min/max mode, constant dimensions become 0, the neutral point in normalized space.

## 类比 / The analogy

像给不同量程的尺子统一刻度：能缩放的尺子映射到同一范围，坏掉不动的尺子不能参与除法，只能保留或放到中间刻度。

It is like standardizing rulers with different ranges. Rulers that vary can be rescaled; a stuck ruler cannot be divided by its range, so it is preserved or placed at the neutral mark.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文: 这是 `training-step` 之前的数据预处理模块，输入是原始 state/action，输出是模型可学习的归一化动作。它依赖动作 schema 和数据集统计量；下游 action head 应该只看归一化后的连续向量。生产级实现还要保存统计量版本，并保证 inverse transform 与机器人控制接口一致。

English: This is the preprocessing module before the `training-step`: raw state/action goes in, normalized model-space actions come out. It depends on the action schema and dataset statistics; downstream action heads should see only normalized continuous vectors. A production version also versions the stats and keeps the inverse transform aligned with the robot control API.

## 自己跑一遍 / Try it yourself

```python
import torch
x = torch.tensor([[2.0, 5.0, 7.0]])
q01 = torch.tensor([0.0, 5.0, 3.0])
q99 = torch.tensor([4.0, 5.0, 11.0])
mask = q01 != q99
y = torch.zeros_like(x)
y[..., mask] = (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask])
y[..., mask] = 2 * y[..., mask] - 1
y[..., ~mask] = x[..., ~mask]
print(y)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
tensor([[0., 5., 0.]])
```

中文: 第二个维度 `q01==q99`，所以没有除以零，而是原样保留。

English: The second dimension has `q01==q99`, so it is preserved instead of divided by zero.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi `Normalize` / openpi `Normalize`**: 中文: 同样在数据 transform 层处理 z-score 和 quantile。 / English: It also handles z-score and quantile normalization in the data transform layer.
- **Diffusion Policy normalizers / Diffusion Policy normalizers**: 中文: 动作扩散模型通常先把控制量映射到固定范围。 / English: Action diffusion models usually map controls into a fixed range before training.

## 注意事项 / Caveats / when it breaks

- **统计量版本 / Stats versioning**: 中文: 训练和部署必须使用同一份统计量。 / English: Training and deployment must use the same statistics.
- **常量维度语义 / Constant-dimension semantics**: 中文: 保留原值还是置零要跟 inverse transform 一起设计。 / English: Preserve-versus-zero must be designed together with the inverse transform.

## 延伸阅读 / Further reading

- Source permalink above.
- Related tracked note: `2026-06-23-openpi-normalize-transform.md`.
