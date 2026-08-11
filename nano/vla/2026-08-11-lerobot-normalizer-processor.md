---
date: 2026-08-11
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/processor/normalize_processor.py
permalink: https://github.com/huggingface/lerobot/blob/59ab28620f3f2385f808bd4bcac7fc50cf14217a/src/lerobot/processor/normalize_processor.py#L302-L421
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, normalization]
build_role: training-step advanced variant, shared observation/action normalization processor
---

# LeRobot normalizer processor：观测和动作共用一张尺度表 / LeRobot Normalizer Processor: One Scale Table for Observations and Actions

> **一句话 / In one line**: `_apply_transform()` 根据 feature type 选择 normalization mode，把观测和动作稳定地映射到训练尺度。 / `_apply_transform()` selects a normalization mode by feature type and maps observations/actions into the scale expected by the policy.

## 为什么重要 / Why this matters

VLA 训练最常见的隐形 bug 是尺度错位：图像、关节、末端位姿、gripper action 全都混在一个 batch 里，但它们的数值范围完全不同。LeRobot 把归一化做成 processor step，让训练前处理和推理后处理共享同一份 dataset stats。

A common hidden VLA bug is scale mismatch: images, joints, end-effector poses, and gripper actions share a batch but not a numeric range. LeRobot puts normalization in a processor step so training pre-processing and inference post-processing use the same dataset statistics.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/processor/normalize_processor.py`](https://github.com/huggingface/lerobot/blob/59ab28620f3f2385f808bd4bcac7fc50cf14217a/src/lerobot/processor/normalize_processor.py#L302-L421)

```python
def _apply_transform(
    self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
) -> Tensor:
    norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
    if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
        return tensor

    if norm_mode not in (
        NormalizationMode.MEAN_STD,
        NormalizationMode.MIN_MAX,
        NormalizationMode.QUANTILES,
        NormalizationMode.QUANTILE10,
    ):
        raise ValueError(f"Unsupported normalization mode: {norm_mode}")

    if self._tensor_stats and key in self._tensor_stats:
        first_stat = next(iter(self._tensor_stats[key].values()))
        if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
            self.to(device=tensor.device, dtype=tensor.dtype)

    stats = self._tensor_stats[key]

    if norm_mode == NormalizationMode.MEAN_STD:
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        if mean is None or std is None:
            raise ValueError(
                "MEAN_STD normalization mode requires mean and std stats, please update the dataset with the correct stats"
            )

        mean, std = stats["mean"], stats["std"]
        denom = std + self.eps
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / denom

    if norm_mode == NormalizationMode.MIN_MAX:
        min_val = stats.get("min", None)
        max_val = stats.get("max", None)
        if min_val is None or max_val is None:
            raise ValueError(
                "MIN_MAX normalization mode requires min and max stats, please update the dataset with the correct stats"
            )

        min_val, max_val = stats["min"], stats["max"]
        denom = max_val - min_val
        denom = torch.where(
            denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
        )
        if inverse:
            return (tensor + 1) / 2 * denom + min_val
        return 2 * (tensor - min_val) / denom - 1

    if norm_mode == NormalizationMode.QUANTILES:
        q01 = stats.get("q01", None)
        q99 = stats.get("q99", None)
        if q01 is None or q99 is None:
            raise ValueError(
                "QUANTILES normalization mode requires q01 and q99 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
            )

        denom = q99 - q01
        denom = torch.where(
            denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
        )
        if inverse:
            return (tensor + 1.0) * denom / 2.0 + q01
        return 2.0 * (tensor - q01) / denom - 1.0
```

## 逐行讲解 / What's happening

1. **第 329-339 行 / Lines 329-339 (mode gate)**:
   - 中文: feature type 决定归一化模式；没有 stats 或显式 `IDENTITY` 就原样返回。
   - English: The feature type selects the mode; missing stats or explicit `IDENTITY` returns the tensor unchanged.
2. **第 341-347 行 / Lines 341-347 (device/dtype sync)**:
   - 中文: stats 会被搬到输入 tensor 的 device/dtype，避免 Accelerate/DDP 下 CPU stats 混进 GPU tensor。
   - English: Stats are moved to the input tensor's device/dtype, avoiding CPU-stat/GPU-tensor mismatches under Accelerate or DDP.
3. **第 349-362 行 / Lines 349-362 (mean/std)**:
   - 中文: 标准化走 `(x - mean) / (std + eps)`，反向则 `x * std + mean`。
   - English: Standardization uses `(x - mean) / (std + eps)` and inversion uses `x * std + mean`.
4. **第 364-384 行 / Lines 364-384 (min/max)**:
   - 中文: min/max 模式把数据映射到 `[-1, 1]`，零范围维度用 `eps` 避免除零。
   - English: Min/max mode maps data to `[-1, 1]`, replacing zero-width ranges with `eps`.
5. **第 386-401 行 / Lines 386-401 (quantiles)**:
   - 中文: quantile 模式用 `q01/q99` 截断极端值影响，更适合带 outlier 的 robot action。
   - English: Quantile mode uses `q01/q99`, which is more robust for robot actions with outliers.

## 类比 / The analogy

像把不同国家的尺子都换成同一把训练尺：有的用厘米，有的用英寸，有的用百分位刻度；processor 负责进门前换单位，出门前换回来。

It is like converting rulers from different countries into one training ruler. Some are centimeters, some inches, some percentile marks; the processor converts units on the way in and back on the way out.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `training-step` 和 `inference-loop` 之间的共享边界层。训练时它把 dataset transition 变成 policy 能吃的 normalized tensor；推理时对应的 unnormalizer 把 policy action 还原成机器人尺度。上游是 dataset/robot observation，下游是 VLM/action head；如果省掉它，模型会把“单位问题”误当成要学习的策略问题。

This is the shared boundary between `training-step` and `inference-loop`. During training it turns dataset transitions into normalized tensors; during inference the matching unnormalizer turns policy actions back into robot units. Upstream are dataset or robot observations; downstream are the VLM and action head.

## 自己跑一遍 / Try it yourself

```python
def minmax(x, lo, hi, inverse=False, eps=1e-8):
    denom = hi - lo if hi != lo else eps
    if inverse:
        return (x + 1) / 2 * denom + lo
    return 2 * (x - lo) / denom - 1

raw = [0.0, 0.5, 1.0]
norm = [round(minmax(x, 0.0, 1.0), 2) for x in raw]
back = [round(minmax(x, 0.0, 1.0, inverse=True), 2) for x in norm]
print(norm)
print(back)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[-1.0, 0.0, 1.0]
[0.0, 0.5, 1.0]
```

这个最小例子对应 LeRobot 的 `MIN_MAX` 分支：训练看 `[-1,1]`，机器人执行看原始单位。

This minimal example matches LeRobot's `MIN_MAX` branch: training sees `[-1,1]`, execution sees original units.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA `predict_action`** / **OpenVLA `predict_action`**: 生成的 action bins 也要按 `q01/q99` 反归一化。 / Generated action bins are also unnormalized with `q01/q99`.
- **openpi transforms** / **openpi transforms**: 输入/输出 transform group 同样把 normalization 放在模型边界。 / Transform groups likewise put normalization at model boundaries.

## 注意事项 / Caveats / when it breaks

- **stats 要和数据集一致** / **Stats must match the dataset**: 换机器人或数据集后复用旧 stats，会直接改变动作尺度。 / Reusing old stats on a new robot or dataset changes the action scale.
- **反向分支也要测试** / **Test the inverse branch too**: 只测训练 normalized 值不够，部署时最容易坏的是 unnormalize。 / Testing normalized training values is not enough; deployment often fails in unnormalization.

## 延伸阅读 / Further reading

- [LeRobot normalization processor](https://github.com/huggingface/lerobot/blob/59ab28620f3f2385f808bd4bcac7fc50cf14217a/src/lerobot/processor/normalize_processor.py#L302-L421)

