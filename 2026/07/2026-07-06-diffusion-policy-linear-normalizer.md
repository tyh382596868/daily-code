---
date: 2026-07-06
topic: robotics
source: tracked
repo: real-stanford/diffusion_policy
file: diffusion_policy/model/common/normalizer.py
permalink: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/normalizer.py#L9-L80
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, normalization]
---

# Diffusion Policy LinearNormalizer：一套参数管住多路观测 / Diffusion Policy LinearNormalizer: One Parameter Store for Many Observation Streams

> **一句话 / In one line**: 它把 action、state、image feature 等不同字段各自拟合 `scale/offset`，然后用同一个接口做 normalize 和 unnormalize。 / It fits separate `scale/offset` parameters for each action, state, or feature field, then exposes one interface for normalize and unnormalize.

## 为什么重要 / Why this matters

机器人数据不是一个整齐的向量：关节角、末端位姿、夹爪、图像特征的量纲都不同。如果把它们直接喂给 diffusion policy，模型会把数值范围最大的字段当成最重要的字段。`LinearNormalizer` 的关键不是公式复杂，而是它把“每个字段一套统计量”变成了可保存、可索引、可反向变换的模块。

Robot data is not one tidy vector: joints, end-effector poses, grippers, and image features all live on different scales. Feeding them directly to a diffusion policy lets large-valued fields dominate training. `LinearNormalizer` matters because it turns "one set of statistics per field" into a reusable module that can be saved, indexed, and inverted.

## 代码 / The code

`real-stanford/diffusion_policy` — [`diffusion_policy/model/common/normalizer.py`](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/normalizer.py#L9-L80)

```python
class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']

    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)
```

## 逐行讲解 / What's happening

1. **第 1 行 / Line 1 (`DictOfTensorMixin`)**:
   - 中文: normalizer 本身像一个小模型，内部参数能跟着 checkpoint 保存。
   - English: the normalizer behaves like a small model whose tensors can travel with checkpoints.
2. **第 4-25 行 / Lines 4-25 (`fit`)**:
   - 中文: 输入是 dict 时，每个 key 单独拟合；输入是单个 tensor 时，用 `_default` 兜底。
   - English: dict inputs get per-key statistics; plain tensors use the `_default` slot.
3. **第 31-34 行 / Lines 31-34 (`__getitem__`, `__setitem__`)**:
   - 中文: 可以只拿出 `action` 或 `state` 的 normalizer，方便在 policy 外部单独处理某一路数据。
   - English: callers can extract just the `action` or `state` normalizer and reuse it outside the policy.
4. **第 36-49 行 / Lines 36-49 (`_normalize_impl`)**:
   - 中文: 正向和反向共用一条路径，只靠 `forward=True/False` 切换公式。
   - English: normalize and unnormalize share one dispatch path, switching behavior with `forward=True/False`.

## 类比 / The analogy

像给厨房里的每个量杯贴自己的换算表：面粉、糖、油都能换成“标准杯”，用完还能换回原单位。你不会拿糖的换算表去量油。

It is like keeping a conversion card on every measuring cup in a kitchen. Flour, sugar, and oil can all be converted into "standard cups" and later converted back, but each ingredient keeps its own conversion.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

stats = {
    "state": np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]]),
    "action": np.array([[-1.0], [0.0], [1.0]]),
}
params = {}
for k, x in stats.items():
    mn, mx = x.min(axis=0), x.max(axis=0)
    scale = 2.0 / np.maximum(mx - mn, 1e-6)
    offset = -1.0 - mn * scale
    params[k] = (scale, offset)

def norm(k, x):
    scale, offset = params[k]
    return x * scale + offset

print(norm("state", np.array([[2.0, 14.0]])))
print(norm("action", np.array([[0.5]])))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[0. 0.]]
[[0.5]]
```

中间状态被映射到 0，动作 0.5 保持在 action 自己的尺度里，不会混用 state 的统计量。

The middle state maps to 0, while action 0.5 stays inside the action field's own scale instead of borrowing state statistics.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi transforms** / **openpi transforms**: 也会把输入归一化和输出反归一化拆成显式 transform。 / It also makes input normalization and output denormalization explicit transforms.
- **LeRobot dataset stats** / **LeRobot dataset stats**: policy 加载时携带 dataset statistics，推理前后都要走同一套尺度。 / Policies carry dataset statistics so inference uses the same scale before and after the model.

## 注意事项 / Caveats / when it breaks

- **统计量必须来自训练集** / **Statistics must come from training data**: 用评测集重新 `fit` 会泄漏分布信息。 / Refitting on evaluation data leaks distribution information.
- **零范围字段要保护** / **Zero-range fields need guards**: 常量维度必须用 `range_eps` 之类的下限避免除零。 / Constant dimensions need a floor such as `range_eps` to avoid division by zero.

## 延伸阅读 / Further reading

- [Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy)
- [LeRobot normalization utilities](https://github.com/huggingface/lerobot)
