---
date: 2026-07-09
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/psoft/layer.py
permalink: https://github.com/huggingface/peft/blob/1598ecb8fc504bfcb08b9b232b295414a729d7ed/src/peft/tuners/psoft/layer.py#L28-L166
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, orthogonal-adapter]
---

# PEFT PSOFT OrthLayer：用 skew 矩阵生成可训练正交旋转 / PEFT PSOFT OrthLayer: Train an Orthogonal Rotation from a Skew Matrix

> **一句话 / In one line**: PSOFT 不直接训练一个完整方阵，而是训练上三角参数，构造 skew-symmetric `Q`，再用 Cayley transform 得到近似正交的 `R`。 / PSOFT does not train a dense square matrix directly; it trains upper-triangular parameters, builds a skew-symmetric `Q`, and turns it into an orthogonal-ish `R` with a Cayley transform.

## 为什么重要 / Why this matters

很多 adapter 方法只关心低秩增量，但 PSOFT 这里关注“旋转”：在 LoRA 的 A/B 之间插一个小的正交变换，让 adapter 有表达力，同时控制参数量和数值形状。这个实现还提供了 Neumann 近似路径，避免某些场景下直接求解线性系统。

Many adapter methods focus on low-rank deltas. PSOFT focuses on a small trainable rotation inserted between LoRA-style factors, adding expressivity while keeping the parameter count and geometry controlled. The implementation also offers a Neumann approximation path instead of always solving a linear system.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/psoft/layer.py`](https://github.com/huggingface/peft/blob/1598ecb8fc504bfcb08b9b232b295414a729d7ed/src/peft/tuners/psoft/layer.py#L28-L166)

```python
class OrthLayer(nn.Module):
    """
    r*r orthogonal transformation R used in PSOFT between A and B. Forward: output = input @ R.T
    """

    def __init__(self, size: int, orth: bool = True, mag_b: bool = True, mag_a: bool = True, ...):
        super().__init__()
        self.size = size
        self.orth = orth
        ...
        if orth:
            self.weight = nn.Parameter(torch.empty((size * (size - 1)) // 2))
            rows, cols = torch.triu_indices(size, size, 1)
            self.register_buffer("rows", rows, persistent=False)
            self.register_buffer("cols", cols, persistent=False)
        else:
            self.weight = nn.Parameter(torch.empty(size, size))

        self.vector_b = nn.Parameter(torch.empty(size)) if mag_b else None
        self.vector_a = nn.Parameter(torch.empty(size)) if mag_a else None

    def _skew_symmetric(self) -> torch.Tensor:
        Q = torch.zeros((self.size, self.size), device=self.weight.device, dtype=self.weight.dtype)
        Q = Q.index_put((self.rows, self.cols), self.weight)
        return Q - Q.transpose(0, 1)

    def _project_Q(self, Q: torch.Tensor, eps: float = 0.9) -> torch.Tensor:
        norm = torch.linalg.norm(Q, ord="fro")
        if torch.isfinite(norm) and norm > eps:
            Q = Q * (eps / (norm + 1e-12))
        return Q

    def get_matrix(self) -> torch.Tensor:
        if not self.orth:
            R = self.weight
        else:
            Q = self._skew_symmetric()
            id_mat = torch.eye(self.size, device=Q.device, dtype=Q.dtype)
            if self.use_cayley_neumann:
                ...
            else:
                cast_to_fp32 = orig_dtype in (torch.float16, torch.bfloat16)
                if cast_to_fp32:
                    Q = Q.float()
                R = torch.linalg.solve(id_mat - Q, id_mat + Q, left=False)

        if self.vector_b is not None:
            R = self.vector_b[:, None] * R
        if self.vector_a is not None:
            R = R * self.vector_a[None, :]
        return R
```

## 逐行讲解 / What's happening

1. **第 48-55 行 / Lines 48-55 (upper-triangle params)**:
   - 中文: `r*r` 正交矩阵不直接存 `r*r` 个参数，只存严格上三角的 `r(r-1)/2` 个自由度。
   - English: The orthogonal matrix is not stored as `r*r`; only the strict upper triangle with `r(r-1)/2` parameters is learned.
2. **第 107-110 行 / Lines 107-110 (`_skew_symmetric`)**:
   - 中文: 参数先放进上三角，再减去转置，得到 `Q = -Q.T`。
   - English: Parameters fill the upper triangle, then subtracting the transpose gives `Q = -Q.T`.
3. **第 128-159 行 / Lines 128-159 (`get_matrix`)**:
   - 中文: Cayley 形式把 skew 矩阵变成旋转矩阵；半精度时先转 fp32，因为求解器需要更稳。
   - English: The Cayley form turns the skew matrix into a rotation; half precision is promoted to fp32 for the solver.
4. **第 161-164 行 / Lines 161-164 (magnitude vectors)**:
   - 中文: 最后再用左右两个向量缩放行列，让“方向”和“幅度”分开调。
   - English: Two optional vectors scale rows and columns, separating direction from magnitude.

## 类比 / The analogy

这像调一面小镜子的角度：你主要训练的是旋转角，而不是重新制造整面镜子；如果还需要亮度变化，再用两个缩放旋钮补上。

It is like adjusting the angle of a small mirror. You train the rotation, not a whole new mirror; if brightness needs changing, two scaling knobs handle that separately.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

w = np.array([0.2, -0.1, 0.3])
Q = np.zeros((3, 3))
Q[np.triu_indices(3, 1)] = w
Q = Q - Q.T
I = np.eye(3)
R = np.linalg.solve((I - Q).T, (I + Q).T).T
print(np.round(Q + Q.T, 6))
print(np.round(R.T @ R, 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

第一块输出说明 `Q` 是反对称的；第二块说明 Cayley transform 保持了近似正交。

The first block shows that `Q` is skew-symmetric; the second shows that the Cayley transform keeps the matrix approximately orthogonal.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OFT adapters** / **OFT adapters**: 也用正交变换约束 adapter 的几何形状。 / They also constrain adapter geometry through orthogonal transforms.
- **Muon optimizer** / **Muon optimizer**: 用矩阵正交化控制更新方向。 / It orthogonalizes matrix updates to control their direction.

## 注意事项 / Caveats / when it breaks

- **Cayley 有数值前提** / **Cayley has numerical assumptions**: `I - Q` 太病态时求解会不稳，所以有 `_project_Q` 和 fp32 cast。 / If `I - Q` is ill-conditioned, solving can be unstable; `_project_Q` and fp32 casting help.
- **正交不等于万能** / **Orthogonal is not always enough**: 所以实现仍保留 `vector_a/vector_b` 来调整幅度。 / That is why the implementation still keeps `vector_a/vector_b` for magnitude control.

## 延伸阅读 / Further reading

- [PEFT `OrthLayer`](https://github.com/huggingface/peft/blob/1598ecb8fc504bfcb08b9b232b295414a729d7ed/src/peft/tuners/psoft/layer.py#L28-L166)
