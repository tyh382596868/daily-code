---
date: 2026-06-30
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/tuners/lora/layer.py
permalink: https://github.com/huggingface/peft/blob/314e988557586d4aefccb0b4a006d1048209ede3/src/peft/tuners/lora/layer.py#L540-L682
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, lora]
---

# LoRA-GA 初始化：用一次梯度 SVD 给 adapter 指方向 / LoRA-GA Init: Use One Gradient SVD to Aim the Adapter

> **一句话 / In one line**: PEFT 的 LoRA-GA 从 full fine-tune 梯度里抽低秩方向，再反向抵消 base weight，保持初始函数不跳变。 / PEFT LoRA-GA extracts low-rank directions from a full fine-tuning gradient, then offsets the base weight so the initial function does not jump.

## 为什么重要 / Why this matters

普通 LoRA 初始化通常让 `B` 为零，所以 adapter 一开始不影响输出。LoRA-GA 更激进：它先看一眼全量梯度，把最有用的方向塞进 `A/B`，再从 base weight 里减掉同样的低秩增量，让函数值保持连续。

Plain LoRA usually initializes `B` to zero, so the adapter starts as a no-op. LoRA-GA is more informed: it looks at a full-gradient snapshot, places useful directions into `A/B`, then subtracts the same low-rank offset from the base weight so the function stays continuous.

## 代码 / The code

`huggingface/peft` — [`src/peft/tuners/lora/layer.py`](https://github.com/huggingface/peft/blob/314e988557586d4aefccb0b4a006d1048209ede3/src/peft/tuners/lora/layer.py#L540-L682)

```python
def lora_ga_init(self, adapter_name, lora_ga_config):
    base_layer = self.get_base_layer()

    if not hasattr(base_layer, "_peft_loraga_grad"):
        self.reset_lora_parameters(adapter_name, init_lora_weights=True)
        return

    grad = base_layer._peft_loraga_grad
    if lora_ga_config is None:
        raise ValueError(
            "lora_ga_config must be provided when init_lora_weights='lora_ga'. "
            "Please pass lora_ga_config=LoraGAConfig(...) to LoraConfig."
        )
    direction = lora_ga_config.direction
    scale = lora_ga_config.scale
    stable_gamma = lora_ga_config.stable_gamma
    dtype = self.get_base_layer().weight.dtype

    grad = grad.to(torch.float32)
    weight = self.get_base_layer().weight
    grad = transpose(grad, self.fan_in_fan_out)
    r = self.r[adapter_name]

    U, S, V = torch.svd_lowrank(grad, q=min(4 * r, min(grad.shape)), niter=4)
    Vh = V.t()
    U = U[:, : 2 * r]
    S = S[: 2 * r]
    Vh = Vh[: 2 * r, :]

    if direction == "ArBr":
        lora_A_weight = Vh[1 : 2 * r : 2, :]
        lora_B_weight = U[:, 0 : 2 * r : 2]
        S_B = S[0 : 2 * r : 2]
        lora_B_weight = lora_B_weight @ torch.diag(S_B)
    elif direction == "A2rBr":
        lora_A_weight = Vh[r : 2 * r, :]
        lora_B_weight = U[:, :r]
        S_B = S[:r]
        lora_B_weight = lora_B_weight @ torch.diag(S_B)
    elif direction == "ArB2r":
        lora_A_weight = Vh[:r, :]
        lora_B_weight = U[:, r : 2 * r]
        S_B = S[r : 2 * r]
        lora_B_weight = lora_B_weight @ torch.diag(S_B)
    elif direction == "random":
        indices = torch.randperm(2 * r)[:r]
        lora_A_weight = Vh[indices, :]
        S_B = S[indices]
        lora_B_weight = U[:, indices] @ torch.diag(S_B)

    scaling_factor = self.scaling[adapter_name]
    if scale == "gd_scale":
        lora_A_weight = lora_A_weight / scaling_factor
        lora_B_weight = lora_B_weight / scaling_factor

    lora_A_weight = lora_A_weight.to(dtype)
    lora_B_weight = lora_B_weight.to(dtype)
    self.lora_A[adapter_name].weight.data = lora_A_weight.contiguous()
    self.lora_B[adapter_name].weight.data = lora_B_weight.contiguous()

    weight_data = transpose(weight.data.to(torch.float32), self.fan_in_fan_out)
    weight_offset = scaling_factor * (lora_B_weight.float() @ lora_A_weight.float())
    weight_data = weight_data - weight_offset
    weight_data = transpose(weight_data.to(dtype), self.fan_in_fan_out)
    self.get_base_layer().weight.data = weight_data
    del base_layer._peft_loraga_grad
```

## 逐行讲解 / What's happening

1. **第 545-548 行 / Lines 545-548**:
   - 中文: 没有预处理梯度时回退到普通初始化，适配加载已有 adapter 的场景。
   - English: If the preprocessing gradient is missing, it falls back to regular initialization, which is useful when loading saved adapters.
2. **第 565-572 行 / Lines 565-572**:
   - 中文: 对梯度矩阵做 low-rank SVD，取 `2r` 个方向作为候选池。
   - English: It runs low-rank SVD on the gradient matrix and keeps `2r` directions as a candidate pool.
3. **第 574-596 行 / Lines 574-596**:
   - 中文: 不同 `direction` 决定哪些奇异向量放进 `A`，哪些放进 `B`。
   - English: Different `direction` modes decide which singular vectors go into `A` and which go into `B`.
4. **第 673-679 行 / Lines 673-679**:
   - 中文: `W_new = W_old - scaling * B @ A` 抵消 adapter 初始增量。
   - English: `W_new = W_old - scaling * B @ A` cancels the adapter's initial delta.

## 类比 / The analogy

这像给自行车加辅助电机：先看你最常往哪个方向用力，再把电机装在那个方向；同时微调刹车，让刚装上时车不会突然往前冲。

It is like adding an assist motor to a bicycle: first observe where the rider pushes hardest, mount the motor along that direction, then adjust the brake so the bike does not lurch forward immediately.

## 自己跑一遍 / Try it yourself

```python
import torch

torch.manual_seed(0)
grad = torch.randn(6, 4)
r = 2
U, S, V = torch.svd_lowrank(grad, q=4, niter=2)
A = V.t()[1 : 2 * r : 2, :]
B = U[:, 0 : 2 * r : 2] @ torch.diag(S[0 : 2 * r : 2])
delta = B @ A
print(A.shape, B.shape, delta.shape)
print(torch.linalg.matrix_rank(delta).item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
torch.Size([2, 4]) torch.Size([6, 2]) torch.Size([6, 4])
2
```

这个示例展示了 LoRA 的核心约束：`B @ A` 回到原权重形状，但秩被限制为 `r`。

The example shows LoRA's core constraint: `B @ A` returns to the base weight shape, but its rank is capped by `r`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LoftQ 初始化** / **LoftQ initialization**: 也在初始化阶段同时处理 base weight 和 adapter，以减少量化误差。 / It also updates both base weight and adapter during initialization to reduce quantization error.
- **PiSSA / EVA LoRA** / **PiSSA / EVA LoRA**: 同样用数据或权重统计来决定低秩子空间，而不是随机起步。 / They also use data or weight statistics to choose the low-rank subspace instead of starting randomly.

## 注意事项 / Caveats / when it breaks

- **需要预处理梯度** / **It needs a preprocessing gradient**: 没有 `_peft_loraga_grad` 就不会真正使用 LoRA-GA。 / Without `_peft_loraga_grad`, the method falls back to standard init.
- **SVD 有成本** / **SVD has a cost**: 大层上初始化更慢，但训练初期方向更好。 / Initialization is slower on large layers, but early training receives a better direction.

## 延伸阅读 / Further reading

- [PEFT LoRA layer](https://github.com/huggingface/peft/blob/314e988557586d4aefccb0b4a006d1048209ede3/src/peft/tuners/lora/layer.py)

