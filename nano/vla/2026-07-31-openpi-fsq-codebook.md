---
date: 2026-07-31
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models/utils/fsq_tokenizer.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/utils/fsq_tokenizer.py#L15-L124
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer, fsq]
build_role: action-tokenizer advanced variant
---

# openpi FSQ codebook：把连续向量写成混合进制 token / openpi FSQ Codebook: Write Continuous Vectors as Mixed-Radix Tokens

> **一句话 / In one line**: `FsqCodebook` 先把向量投到少数几个量化维度，再用混合进制把每维 bin 合成一个整数 token。 / `FsqCodebook` projects vectors into a few quantized dimensions, then packs per-dimension bins into one integer token with mixed radix.

## 为什么重要 / Why this matters

VLA 可以输出连续动作，也可以把动作离散成 token 后交给语言模型式 head。FSQ/LFQ 的价值在于：不用维护一个大 embedding table，就能把连续空间压成稳定的离散 ID，并通过 straight-through trick 保持训练梯度。

A VLA can emit continuous actions, or discretize actions into tokens for a language-model-like head. FSQ/LFQ is useful because it turns continuous space into stable discrete IDs without a large learned embedding table, while the straight-through trick preserves gradients during training.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models/utils/fsq_tokenizer.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/utils/fsq_tokenizer.py#L15-L124)

```python
class FsqCodebook(nn.Module):
    input_dim: int
    target_codebook_size: int
    codebook_type: Literal["fsq", "lfq"]

    _bins_per_dim: tuple[int] | None = None

    @property
    def bins_per_dim(self) -> tuple[int]:
        if self._bins_per_dim is not None:
            return self._bins_per_dim
        if self.codebook_type == "fsq":
            return self._get_bins_fsq(self.target_codebook_size)
        elif self.codebook_type == "lfq":
            return self._get_bins_lfq(self.target_codebook_size)
        elif self.codebook_type == "custom":
            return self._get_bins_custom(self.target_codebook_size)
        else:
            raise ValueError(f"Codebook type {self.codebook_type} not supported.")

    @property
    def place_values(self) -> jnp.ndarray:
        place_values = [1]
        for b in self.bins_per_dim[:-1]:
            place_values.append(place_values[-1] * b)
        return jnp.array(place_values)

    def setup(self):
        self.proj_down = nn.Dense(len(self.bins_per_dim))
        self.proj_up = nn.Dense(self.input_dim)

    def encode(self, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        bases = jnp.array(self.bins_per_dim)
        x = self.proj_down(inputs)
        z = jnp.tanh(x)
        digits = jnp.round((z + 1) * (bases - 1) / 2).astype(jnp.int32)
        tokens = self.undigitize(digits)
        return tokens, z

    def decode(self, tokens: jnp.ndarray, z_grad: jax.Array | None = None) -> jnp.ndarray:
        bases = jnp.array(self.bins_per_dim)
        digits = self.digitize(tokens)
        z_q = digits / (bases - 1) * 2 - 1
        if z_grad is not None:
            chex.assert_equal_shape([z_q, z_grad])
            z_q = jax.lax.stop_gradient(z_q - z_grad) + z_grad
        return self.proj_up(z_q)

    def undigitize(self, digits: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(digits * jnp.array(self.place_values), axis=-1)

    def digitize(self, tokens: jnp.ndarray) -> jnp.ndarray:
        return (tokens[..., None] // jnp.array(self.place_values)) % jnp.array(self.bins_per_dim)
```

## 逐行讲解 / What's happening

1. **第 22-34 行 / Lines 22-34 (`bins_per_dim`)**:
   - 中文: 码本大小不是直接变成一个维度，而是拆成每个量化维度的 bin 数。
   - English: The codebook size is not used as one flat dimension; it becomes a tuple of bins across quantized dimensions.
2. **第 36-41 行 / Lines 36-41 (`place_values`)**:
   - 中文: `place_values` 是混合进制权重，作用类似十进制里的个位、十位、百位。
   - English: `place_values` are mixed-radix weights, like ones, tens, and hundreds in decimal notation.
3. **第 93-103 行 / Lines 93-103 (`encode`)**:
   - 中文: 输入先降维，再 `tanh` 限制到 `[-1, 1]`，最后按每维 bin 四舍五入成 digit。
   - English: Inputs are projected down, bounded to `[-1, 1]` with `tanh`, then rounded into per-dimension digits.
4. **第 105-115 行 / Lines 105-115 (`decode`)**:
   - 中文: token 先拆回 digits，再映射回 `[-1, 1]`；`stop_gradient` 让前向像量化、反向像连续。
   - English: Tokens are unpacked into digits and mapped back to `[-1, 1]`; `stop_gradient` makes the forward pass quantized and the backward pass continuous.

## 类比 / The analogy

这像把一个坐标写成门牌号：楼栋、楼层、房间号各有取值范围，合起来就是一个唯一地址。

It is like writing a coordinate as an apartment address: building, floor, and room each have a range, and together they form one unique address.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-tokenizer` 的 advanced variant。上游是连续动作或动作隐藏向量，下游是离散 token head 或自回归策略；如果省掉它，你只能走连续回归/扩散 head，不能复用 LM 的 token 建模能力。生产级版本还要处理动作维度归一化、非法 token mask、跨机器人动作空间对齐。

This is an advanced variant of the `action-tokenizer` slot. Upstream are continuous actions or action hidden states; downstream is a discrete-token head or autoregressive policy. Without it, you are limited to continuous regression or diffusion heads and cannot reuse LM-style token modeling. A production version still needs action normalization, invalid-token masks, and cross-robot action-space alignment.

## 自己跑一遍 / Try it yourself

```python
def place_values(bases):
    out = [1]
    for b in bases[:-1]:
        out.append(out[-1] * b)
    return out

def undigitize(digits, bases):
    return sum(d * p for d, p in zip(digits, place_values(bases)))

def digitize(token, bases):
    return [(token // p) % b for p, b in zip(place_values(bases), bases)]

bases = (8, 6, 5)
digits = [3, 4, 2]
token = undigitize(digits, bases)
print(token)
print(digitize(token, bases))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
131
[3, 4, 2]
```

一个整数 token 可以无损还原成多个量化维度。

One integer token can be losslessly unpacked into multiple quantized dimensions.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA bin tokenizer** / **OpenVLA bin tokenizer**: 把动作值映射到离散 bin，但通常是一维逐动作维度分箱。 / It maps action values into discrete bins, often with one-dimensional bins per action dimension.
- **MolmoAct2 DCT tokenizer** / **MolmoAct2 DCT tokenizer**: 先压缩动作轨迹频率，再交给 BPE，目标也是让动作进入 token 世界。 / It compresses action trajectories in frequency space before BPE; the goal is also to move actions into token space.

## 注意事项 / Caveats / when it breaks

- **bin 设计会限制精度** / **Bins cap precision**: 码本太小会把细动作压平。 / A codebook that is too small flattens fine-grained actions.
- **离散 token 仍需物理约束** / **Discrete tokens still need physical constraints**: 解码后要检查关节范围、速度和安全约束。 / Decoded actions still need joint-limit, velocity, and safety checks.

## 延伸阅读 / Further reading

- [openpi FSQ tokenizer source permalink](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/utils/fsq_tokenizer.py#L15-L124)
