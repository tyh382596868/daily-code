---
date: 2026-08-20
topic: vla
source: vla
repo: starVLA/starVLA
file: starVLA/model/modules/action_model/discrete_diffusion/action_binning.py
permalink: https://github.com/starVLA/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/model/modules/action_model/discrete_diffusion/action_binning.py#L17-L163
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer, discrete-actions]
build_role: action-tokenizer advanced variant
---

# StarVLA ActionBinning：连续动作先落进桶里 / StarVLA ActionBinning: Put Continuous Actions into Bins First

> **一句话 / In one line**: 这段代码把 `[-1, 1]` 的连续动作离散成 bin index，也支持用 8 个 bit logits 表示 256 个桶。 / This code discretizes continuous actions in `[-1, 1]` into bin indices and can also represent 256 bins through 8 bit logits.

## 为什么重要 / Why this matters

VLA 动作 head 不一定直接回归连续值。把动作分箱后，模型可以像预测 token 一样预测动作，离散 diffusion 或 masked token 解码也更自然。StarVLA 同时保留普通 `bin` 和 `bit` 两种表示，方便在 logits 维度和表达能力之间取舍。

A VLA action head does not have to regress continuous values directly. Once actions are binned, the model can predict them like tokens, which fits discrete diffusion or masked token decoding. StarVLA keeps both `bin` and `bit` representations, trading off logit width and expressiveness.

## 代码 / The code

`starVLA/starVLA` — [`starVLA/model/modules/action_model/discrete_diffusion/action_binning.py`](https://github.com/starVLA/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/model/modules/action_model/discrete_diffusion/action_binning.py#L17-L163)

```python
def continuous_to_bins(actions: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Map continuous actions in [low, high] to bin indices in [0, num_bins-1]. Ref: floor((x+1)/2 * num_bins)."""
    clamped = actions.clamp(low + 1e-6, high - 1e-6)
    x = (clamped - low) / (high - low)  # [0, 1)
    bins = (x * num_bins).floor().long().clamp(0, num_bins - 1)
    return bins


def bins_to_continuous(bin_indices: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Map bin indices to continuous actions (bin centers). Ref: (i+0.5)/num_bins * 2 - 1 for [-1,1]."""
    scale = (high - low) / num_bins
    centers = (bin_indices.float() + 0.5) * scale + low
    return centers


class ActionBinning(nn.Module):
    def __init__(
        self,
        num_bins: int,
        action_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        representation: str = "bin",
        num_bits: int = 8,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.low = low
        self.high = high
        self.representation = representation
        self.num_bits = num_bits

        if representation == "bit":
            expected_bins = 1 << num_bits
            assert num_bins == expected_bins, f"bit representation requires num_bins={expected_bins}, got {num_bins}"

        # Bin centers for decoding
        scale = (high - low) / num_bins
        centers = (torch.arange(num_bins, dtype=torch.float32) + 0.5) * scale + low
        self.register_buffer("bin_centers", centers)

        # Powers of 2 for bit->bin conversion
        if representation == "bit":
            self.register_buffer("bit_powers", torch.arange(num_bits, dtype=torch.long))

    @property
    def logits_dim(self) -> int:
        """Per-position model output dimension (last dim of logits)."""
        return self.num_bits if self.representation == "bit" else self.num_bins

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode continuous actions to discrete bin indices. (B, T, action_dim) -> (B, T, action_dim) long."""
        return continuous_to_bins(actions, self.num_bins, self.low, self.high)

    def decode(self, indices: torch.Tensor, logging: bool = False) -> torch.Tensor:
        """Decode discrete bin indices to continuous actions (bin centers)."""
        flat = indices.reshape(-1)
        decoded = self.bin_centers[flat].to(indices.device)
        B = indices.shape[0]
        output = decoded.reshape(B, -1, self.action_dim)
        if logging:
            print(f"indices output shape: {indices.shape}")
            print(f"indices output first batch first action: {indices[0, :]}")
        return output

    def _logits_to_indices_bin(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (indices, selected_probs)."""
        safe_temp = max(temperature, 1e-8)
        probs = F.softmax(logits / safe_temp, dim=-1)
        if deterministic:
            indices = logits.argmax(dim=-1)
            selected_probs = probs.max(dim=-1).values
        else:
            flat = probs.reshape(-1, self.num_bins)
            sampled_flat = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
            indices = sampled_flat.reshape(*logits.shape[:-1])
            selected_probs = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return indices, selected_probs
```

## 逐行讲解 / What's happening

1. **第 17-22 行 / Lines 17-22 (`continuous_to_bins`)**:
   - 中文: 先 clamp 到开区间，再归一化到 `[0, 1)`，最后乘桶数并 floor。
   - English: The action is clamped into an open interval, normalized to `[0, 1)`, multiplied by bin count, then floored.
2. **第 25-29 行 / Lines 25-29 (`bins_to_continuous`)**:
   - 中文: 解码用桶中心，不用桶边界，避免动作落在区间边缘。
   - English: Decoding uses bin centers rather than edges, avoiding boundary-valued actions.
3. **第 57-68 行 / Lines 57-68 (`bit` mode)**:
   - 中文: bit 表示要求 `num_bins == 2 ** num_bits`，8 个 bit 就覆盖 256 个桶。
   - English: Bit mode requires `num_bins == 2 ** num_bits`; 8 bits cover 256 bins.
4. **第 90-108 行 / Lines 90-108 (`_logits_to_indices_bin`)**:
   - 中文: deterministic 走 argmax，采样模式走 `multinomial`，同时返回选中概率给调度器使用。
   - English: Deterministic mode uses argmax; sampling mode uses `multinomial`, and selected probabilities are returned for scheduling.

## 类比 / The analogy

像把机械臂油门从连续旋钮改成带刻度的档位。训练时预测“第几个档”，执行时再把档位还原成档位中心对应的油门值。

It is like replacing a continuous robot throttle knob with numbered detents. Training predicts the detent index, and execution converts that index back to the center value of the detent.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoVLA 里，这是 `action-tokenizer` 的高级变体。上游是归一化后的连续 action，输出是离散 index 或 bit target；下游可以是离散 diffusion head、masked action model 或语言模型 token 头。省掉它就只能回归连续动作，无法复用 token 预测、mask schedule 和离散采样工具。生产级还要补 per-dimension bounds、非均匀分箱、动作 mask 和去归一化统计。

In a nanoVLA, this is an advanced `action-tokenizer` variant. Upstream is normalized continuous action; output is a discrete index or bit target; downstream can be a discrete diffusion head, masked action model, or language-model-style token head. Without it, you are limited to continuous regression and cannot reuse token prediction, mask scheduling, or discrete sampling tools. A production version needs per-dimension bounds, non-uniform bins, action masks, and denormalization stats.

## 自己跑一遍 / Try it yourself

```python
def encode(x, bins=4, low=-1.0, high=1.0):
    x = min(max(x, low + 1e-6), high - 1e-6)
    return int(((x - low) / (high - low)) * bins)

def decode(i, bins=4, low=-1.0, high=1.0):
    return (i + 0.5) * ((high - low) / bins) + low

values = [-1.0, -0.2, 0.2, 0.99]
ids = [encode(v) for v in values]
print(ids)
print([round(decode(i), 2) for i in ids])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0, 1, 2, 3]
[-0.75, -0.25, 0.25, 0.75]
```

中文: 解码回来的不是原值，而是桶中心；这是离散化误差的来源，也是分类训练的代价。

English: Decoding returns bin centers, not the original values. That quantization error is the cost of classification-style action prediction.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA action tokenizer** / **OpenVLA action tokenizer**: 把连续动作映射到离散 token，再接到语言模型输出空间。 / It maps continuous actions into discrete tokens attached to the language-model output space.
- **pi0-FAST / FAST tokenizer** / **pi0-FAST / FAST tokenizer**: 用更复杂的 action tokenization 压缩动作序列。 / It uses a more complex action tokenization scheme to compress action sequences.

## 注意事项 / Caveats / when it breaks

- **边界 clamp 会改变极值** / **Boundary clamp changes extremes**: `1.0` 会被压到 `high - 1e-6`，避免落到越界桶。 / `1.0` is clamped to `high - 1e-6` to avoid an out-of-range bin.
- **bit 独立假设较强** / **Bit independence is a strong assumption**: bit 模式把一个桶拆成多个 Bernoulli 预测，桶之间相关性不如 full softmax 直接。 / Bit mode decomposes one bin into Bernoulli predictions, which represents bin correlations less directly than a full softmax.

## 延伸阅读 / Further reading

- [StarVLA `ActionBinning`](https://github.com/starVLA/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/model/modules/action_model/discrete_diffusion/action_binning.py#L17-L163)
- [OpenVLA action tokenizer](https://github.com/openvla/openvla)
