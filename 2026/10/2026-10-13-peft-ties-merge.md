---
date: 2026-10-13
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/utils/merge_utils.py
permalink: https://github.com/huggingface/peft/blob/b3f11aa390319ca076182ae472dc9e08f11caf59/src/peft/utils/merge_utils.py#L103-L214
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, adapter-merge]
---

# PEFT TIES merge：先投票方向，再合并 adapter delta / PEFT TIES Merge: Vote on Direction Before Merging Adapter Deltas

> **一句话 / In one line**: TIES 先剪枝 task tensor，再按多数符号筛掉冲突方向，最后平均同向 adapter 更新。 / TIES prunes task tensors, filters conflicting directions by majority sign, and averages aligned adapter updates.

## 为什么重要 / Why this matters

多个 LoRA delta 直接相加会互相抵消；TIES 用“保留强信号、投票方向、只合并同向”的流程降低冲突。

Multiple LoRA deltas can cancel when summed directly; TIES reduces conflict by keeping strong signals, voting direction, and merging aligned updates only.

## 代码 / The code

`huggingface/peft` - [`src/peft/utils/merge_utils.py`](https://github.com/huggingface/peft/blob/b3f11aa390319ca076182ae472dc9e08f11caf59/src/peft/utils/merge_utils.py#L103-L214)

```python
def calculate_majority_sign_mask(
    tensor: torch.Tensor, method: Literal["total", "frequency"] = "total"
) -> torch.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`torch.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The majority sign mask.
    """

    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    """
    Merge the task tensors using disjoint merge.

    Args:
        task_tensors (`torch.Tensor`):The task tensors to merge.
        majority_sign_mask (`torch.Tensor`):The mask of the majority sign across the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)


def task_arithmetic(task_tensors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def magnitude_prune(task_tensors: list[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`): The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def ties(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    Merge the task tensors using `ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors
```

## 逐行讲解 / What's happening

1. **第 103-125 行 / Lines 103-125 (majority sign)**:
   - 中文: `total` 看幅度和，`frequency` 看正负票数。
   - English: `total` uses signed magnitude, while `frequency` counts signs.
2. **第 128-141 行 / Lines 128-141 (disjoint merge)**:
   - 中文: 反向更新被 mask 掉，只除以保留下来的数量。
   - English: Opposing updates are masked out, and division uses only preserved counts.
3. **第 185-214 行 / Lines 185-214 (TIES path)**:
   - 中文: 顺序是剪枝、选符号、加权、disjoint merge。
   - English: The order is pruning, sign election, weighting, and disjoint merge.

## 类比 / The analogy

这像多人修地图：先决定每条路的主方向，再只平均同意这个方向的笔迹。

It is like multiple people editing a map: first decide the main direction of each road, then average only marks that agree.

## 自己跑一遍 / Try it yourself

```python
tasks = [[3, -2, 0], [1, 4, -5], [-2, -1, -6]]
majority = [1 if sum(cols) >= 0 else -1 for cols in zip(*tasks)]
merged = []
for j, sign in enumerate(majority):
    kept = [row[j] for row in tasks if (1 if row[j] >= 0 else -1) == sign]
    merged.append(round(sum(kept) / len(kept), 2))
print(majority)
print(merged)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 1, -1]
[2.0, 4.0, -5.5]
```

中文: `total` 多数派会让大幅度更新拥有更高话语权。

English: The `total` majority gives larger updates more influence.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Model soups** / **Model soups**: Model soups。 / Model soups.
- **Sparse adapter routing** / **Sparse adapter routing**: Sparse adapter routing。 / Sparse adapter routing.

## 注意事项 / Caveats / when it breaks

- **符号不是语义** / **Same-sign deltas do not guarantee identical behavior**: 符号不是语义。 / Same-sign deltas do not guarantee identical behavior.
- **density 很敏感** / **The pruning density controls signal versus noise**: density 很敏感。 / The pruning density controls signal versus noise.

## 延伸阅读 / Further reading
- [PEFT merge_utils.py](https://github.com/huggingface/peft/blob/b3f11aa390319ca076182ae472dc9e08f11caf59/src/peft/utils/merge_utils.py#L103-L214)
