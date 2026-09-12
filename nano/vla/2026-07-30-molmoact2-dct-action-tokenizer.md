---
date: 2026-07-30
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/molmoact2/molmoact2_hf_model/action_tokenizer.py
permalink: https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/policies/molmoact2/molmoact2_hf_model/action_tokenizer.py#L51-L137
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer, dct]
build_role: action-tokenizer advanced variant
---

# MolmoAct2 action tokenizer：先做 DCT，再交给 BPE / MolmoAct2 Action Tokenizer: DCT First, BPE Second

> **一句话 / In one line**: `UniversalActionProcessor` 把连续动作 chunk 变成 DCT 系数，再量化成字符序列交给 BPE tokenizer。 / `UniversalActionProcessor` turns continuous action chunks into DCT coefficients, then quantizes them into character strings for a BPE tokenizer.

## 为什么重要 / Why this matters

把动作直接逐维离散化会浪费 token：相邻时间步高度相关。MolmoAct2 先用 DCT 把一段动作压到频率空间，让 BPE 更容易复用“平滑轨迹”的重复模式。

Directly discretizing each action dimension wastes tokens because neighboring timesteps are highly correlated. MolmoAct2 first moves a chunk into DCT frequency space, making it easier for BPE to reuse repeated smooth-trajectory patterns.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/molmoact2/molmoact2_hf_model/action_tokenizer.py`](https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/policies/molmoact2/molmoact2_hf_model/action_tokenizer.py#L51-L137)

```python
class UniversalActionProcessor(ProcessorMixin):
    attributes: ClassVar[list[str]] = ["tokenizer"]
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        scale: float = 10,
        vocab_size: int = 1024,
        min_token: int = 0,
        *,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ):
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token

        # Action horizon and dimension needed during decoding. These can be specified
        # in three ways (in order of priority):
        # 1. passed in as kwargs to decode()
        # 2. in the constructor
        # 3. cached from the last time decode() was called
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

        super().__init__(tokenizer)
        self.bpe_tokenizer = self.tokenizer

    def __call__(self, action_chunk: np.array) -> np.array:
        from scipy.fft import dct

        assert action_chunk.ndim <= 3, "Only 3 dimensions supported: [batch, timesteps, action_dim]"
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]

        # Cache the time horizon and action dimension for decoding
        self.called_time_horizon = action_chunk.shape[-2]
        self.called_action_dim = action_chunk.shape[-1]

        dct_coeff = dct(action_chunk, axis=1, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.scale)
        tokens = []
        for elem in dct_coeff:
            token_str = "".join(map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
            tokens.append(self.bpe_tokenizer(token_str)["input_ids"])
        return tokens

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.array:
        from scipy.fft import idct

        self.time_horizon = time_horizon or self.time_horizon or self.called_time_horizon
        self.action_dim = action_dim or self.action_dim or self.called_action_dim

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert self.time_horizon is not None and self.action_dim is not None, (
            "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."
        )

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.min_token
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert decoded_dct_coeff.shape == (
                    self.time_horizon,
                    self.action_dim,
                ), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({self.time_horizon}, {self.action_dim})"
                )
            except Exception:
                logger.warning("Error decoding tokens: %s", token, exc_info=True)
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions)
```

## 逐行讲解 / What's happening

1. **第 65-80 行 / Lines 65-80 (shape memory)**:
   - 中文: processor 记住 `time_horizon` 和 `action_dim`，这样 decode 时才能还原矩阵形状。
   - English: The processor remembers `time_horizon` and `action_dim` so decoding can restore matrix shape.
2. **第 82-99 行 / Lines 82-99 (`__call__`)**:
   - 中文: encode 路径先做 DCT，再乘 scale 四舍五入，最后把整数映射成字符给 BPE。
   - English: The encode path applies DCT, scales and rounds coefficients, then maps integers to characters for BPE.
3. **第 101-137 行 / Lines 101-137 (`decode`)**:
   - 中文: decode 反过来：BPE id 转字符串，字符转整数矩阵，再用 IDCT 回到动作空间。
   - English: Decode reverses it: BPE ids to string, chars to integer matrix, then IDCT back to action space.
4. **第 123-136 行 / Lines 123-136 (fallback)**:
   - 中文: 解码失败时返回零动作，保证推理接口不因单个坏 token 崩掉。
   - English: If decoding fails, it returns zero actions so inference does not crash on one bad token sequence.

## 类比 / The analogy

这像把一段旋律先写成“低音多一点、高音少一点”的乐谱，再压缩成常见乐句；比逐个采样点记录更省。

It is like writing a melody as low and high frequency notes, then compressing common phrases; it is cheaper than recording every sample point independently.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoVLA 里，这属于 `action-tokenizer` 的 advanced variant。上游是 action chunk，输出是一串 LM token；下游是 VLM/LLM 的 next-token loss 或 generation。省掉这一层，就只能让模型直接回归连续动作，不能复用语言模型的离散生成能力。

In a nanoVLA, this is an advanced `action-tokenizer` variant. Upstream is an action chunk, output is a sequence of LM tokens, and downstream is next-token loss or generation inside the VLM/LLM. Without this layer, the model must directly regress continuous actions and cannot reuse discrete language-model generation.

## 自己跑一遍 / Try it yourself

```python
import math

def toy_dct(xs):
    n = len(xs)
    return [round(sum(xs[t] * math.cos(math.pi/n * (t + 0.5) * k) for t in range(n)), 3) for k in range(n)]

action = [0.0, 0.5, 1.0, 0.5]
coeff = toy_dct(action)
quant = [round(c * 10) for c in coeff]
chars = ''.join(chr(q + 64) for q in quant)
print(coeff)
print(quant, chars)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[2.0, -0.653, -0.707, 0.271]
[20, -7, -7, 3] T99C
```

平滑轨迹的大部分信息集中在少数低频系数里，这就是先 DCT 的动机。

Most information in a smooth trajectory sits in a few low-frequency coefficients; that is why DCT comes first.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA action tokenizer** / **OpenVLA action tokenizer**: 把连续动作分箱到语言 token。 / Bins continuous actions into language tokens.
- **openpi FAST tokenizer** / **openpi FAST tokenizer**: 同样把动作序列转成 LM 能生成的 token 段。 / Also turns action sequences into token spans an LM can generate.

## 注意事项 / Caveats / when it breaks

- **scale 是精度旋钮** / **Scale controls precision**: 太小会丢动作细节，太大会扩大字符 alphabet。 / Too small loses action detail; too large expands the character alphabet.
- **形状必须可恢复** / **Shape must be recoverable**: 没有 horizon/action_dim，token 流无法 reshape 回动作矩阵。 / Without horizon/action_dim, the token stream cannot reshape back to an action matrix.

## 延伸阅读 / Further reading

- [LeRobot MolmoAct2 action tokenizer](https://github.com/huggingface/lerobot)
- [Source permalink](https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/policies/molmoact2/molmoact2_hf_model/action_tokenizer.py#L51-L137)
