---
date: 2026-07-24
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/parametrizations.py
permalink: https://github.com/pytorch/pytorch/blob/2f22fb9cbe91b75d78006d58c896cc3516fbf0b2/torch/nn/utils/parametrizations.py#L409-L515
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, parametrization, spectral-norm, power-method]
---

# PyTorch SpectralNorm：用 power iteration 管住权重尺度 / PyTorch SpectralNorm: Keep Weight Scale in Check with Power Iteration

> **一句话 / In one line**: `_SpectralNorm` 把权重摊平成矩阵，用缓存的 `u/v` 奇异向量近似最大奇异值，再返回 `weight / sigma`。 / `_SpectralNorm` flattens a weight into a matrix, approximates the top singular value with cached `u/v` vectors, and returns `weight / sigma`.

## 为什么重要 / Why this matters

谱归一化常用于稳定 GAN 判别器和 Lipschitz 受控模型。PyTorch 的实现有两个工程细节很值得学：`u/v` 是 buffer 并在训练时原地更新；真正归一化前又 clone 一份，避免两次 forward 的 autograd 版本冲突。

Spectral normalization is used to stabilize GAN critics and Lipschitz-controlled models. PyTorch's implementation has two useful engineering details: `u/v` are buffers updated in place during training, then cloned before normalization to avoid autograd version conflicts across multiple forwards.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/parametrizations.py`](https://github.com/pytorch/pytorch/blob/2f22fb9cbe91b75d78006d58c896cc3516fbf0b2/torch/nn/utils/parametrizations.py#L409-L515)

```python
class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )

        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim <= 1:
            raise AssertionError(
                f"Expected weight to have more than 1 dimension, got {weight.ndim}"
            )

        if self.dim != 0:
            weight = weight.permute(
                self.dim, *(d for d in range(weight.dim()) if d != self.dim)
            )

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        for _ in range(n_power_iterations):
            self._u = F.normalize(
                torch.mv(weight_mat, self._v),
                dim=0,
                eps=self.eps,
                out=self._u,
            )
            self._v = F.normalize(
                torch.mv(weight_mat.H, self._u),
                dim=0,
                eps=self.eps,
                out=self._v,
            )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            return weight / sigma
```

## 逐行讲解 / What's happening

1. **第 427-437 行 / Lines 427-437 (`_u`, `_v`)**: 中文: 初始化两个单位向量，分别近似最大左/右奇异向量。 English: two unit vectors approximate the top left and right singular vectors.
2. **第 439 行 / Line 439 (`15`)**: 中文: 注册时先跑 15 次，让随机向量靠近稳定方向。 English: registration runs 15 warmup iterations so random vectors start near a stable direction.
3. **第 448-454 行 / Lines 448-454 (`permute` + `flatten`)**: 中文: 卷积核等高维权重先把输出维挪到最前，再摊平成二维矩阵。 English: higher-dimensional weights move the output dimension first, then flatten to a matrix.
4. **第 459-475 行 / Lines 459-475 (`out=self._u`)**: 中文: `out=` 让 buffer 原地更新，这对 `DataParallel` 的共享存储很关键。 English: `out=` updates the buffers in place, which matters for shared storage under `DataParallel`.
5. **第 483-488 行 / Lines 483-488 (`clone` + `sigma`)**: 中文: clone 后再算 `sigma`，避免第二次 forward 改掉第一次 backward 需要的向量。 English: cloning before `sigma` avoids mutating vectors needed by a previous forward's backward pass.

## 类比 / The analogy

像给扩音器加一个自动限幅器。它不断估计当前最大音量方向，然后把整台设备的输出除以这个最大值，避免某个方向突然爆音。

It is like adding an automatic limiter to a speaker. It keeps estimating the loudest direction, then divides the output by that maximum so no direction spikes.

## 自己跑一遍 / Try it yourself

```python
import math

W = [[3.0, 0.0], [4.0, 0.0]]
v = [1.0, 1.0]
for _ in range(5):
    u = [W[0][0] * v[0] + W[0][1] * v[1], W[1][0] * v[0] + W[1][1] * v[1]]
    un = math.sqrt(sum(x * x for x in u))
    u = [x / un for x in u]
    v = [W[0][0] * u[0] + W[1][0] * u[1], W[0][1] * u[0] + W[1][1] * u[1]]
    vn = math.sqrt(sum(x * x for x in v))
    v = [x / vn for x in v]
sigma = sum(u[i] * sum(W[i][j] * v[j] for j in range(2)) for i in range(2))
print(round(sigma, 3))
print([[round(x / sigma, 3) for x in row] for row in W])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
5.0
[[0.6, 0.0], [0.8, 0.0]]
```

中文: 这块矩阵最大的尺度是 5，归一化后第一列长度变成 1。 English: the matrix's largest scale is 5, and normalization makes the first column length 1.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`torch.nn.utils.parametrize`** / **`torch.nn.utils.parametrize`**: 同一套机制也能把正交约束、非负约束挂到参数访问路径上。 / the same mechanism can attach orthogonal or non-negative constraints to parameter access.
- **EMA / SWA buffer updates** / **EMA / SWA buffer updates**: 训练时原地维护统计量，forward 时读稳定副本，是很多 PyTorch 工具的共同模式。 / maintaining statistics in place while reading stable copies is common in PyTorch utilities.

## 注意事项 / Caveats / when it breaks

- **只近似最大奇异值 / It approximates the top singular value**: `n_power_iterations=1` 很快，但不是精确 SVD。 / `n_power_iterations=1` is fast, not exact SVD.
- **eval 不更新方向 / Eval does not update directions**: eval 模式用当前缓存的 `u/v`。 / eval mode uses the currently cached `u/v`.
- **一维权重走特例 / Vectors use a special path**: 一维参数直接 L2 normalize，不跑 power iteration。 / one-dimensional weights are directly L2-normalized.

## 延伸阅读 / Further reading

- [PyTorch parametrizations.py](https://github.com/pytorch/pytorch/blob/2f22fb9cbe91b75d78006d58c896cc3516fbf0b2/torch/nn/utils/parametrizations.py#L409-L515)

