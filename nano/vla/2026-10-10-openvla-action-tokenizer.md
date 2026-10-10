---
date: 2026-10-10
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/vla/action_tokenizer.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py#L13-L70
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer]
build_role: action-tokenizer advanced variant
---
# OpenVLA action tokenizer：连续动作借用 LLM 词表尾部 / OpenVLA Action Tokenizer: Continuous Actions Borrow the Tail of the LLM Vocabulary

> **一句话 / In one line**: OpenVLA 把每个连续动作维度离散成 bin，再映射到 tokenizer 词表最后一段 token。 / OpenVLA discretizes each continuous action dimension into a bin, then maps it onto the final tokens of the tokenizer vocabulary.

## 为什么重要 / Why this matters

让 VLM 生成机器人动作的直接办法，是先把连续值变成 token。OpenVLA 用固定范围、均匀分箱和词表尾部 token，让动作监督复用 next-token 训练路径。

A direct way to make a VLM generate robot actions is to turn continuous values into tokens. OpenVLA uses a fixed range, uniform bins, and tail vocabulary tokens so action supervision can reuse next-token training.

## 代码 / The code

`openvla/openvla` — [`prismatic/vla/action_tokenizer.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py#L13-L70)

```python
class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
```

## 逐行讲解 / What's happening

1. **入口 / Entry**:
   - 中文: 先看函数签名和输入，它定义了这个模块承担的边界职责。
   - English: Start from the signature and inputs; they define the module boundary.
2. **核心状态 / Core state**:
   - 中文: 中间变量保存的是工程约束，例如 cache、rank、bin 或时间线。
   - English: The intermediate variables encode engineering constraints such as cache, rank, bins, or timeline state.
3. **返回值 / Return value**:
   - 中文: 输出不是孤立结果，而是给下游模块继续消费的契约。
   - English: The output is not an isolated result; it is the contract consumed downstream.

## 类比 / The analogy

像给机械臂旋钮贴 256 个刻度贴纸，然后让语言模型先学会说贴纸编号。

It is like putting 256 numbered stickers around a robot dial, then asking the language model to name the sticker.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 `action-tokenizer` 的高级变体：它把上游数据变成下游模型能稳定消费的表示。省掉它，nano 系统仍可跑通玩具例子，但会在真实动作空间或视频 chunk 边界上迅速暴露问题。

This is an advanced variant of `action-tokenizer`. It converts upstream data into a representation the downstream model can consume reliably. Without it, a nano system may run toy examples but will fail quickly on real action spaces or video chunk boundaries.

## 自己跑一遍 / Try it yourself

```python
edges = [-1 + i * (2 / 7) for i in range(8)]
centers = [(a + b) / 2 for a, b in zip(edges, edges[1:])]

def digitize(x):
    x = min(1, max(-1, x))
    return sum(x >= edge for edge in edges)

action = [-0.7, 0.2, 1.0]
ids = [1000 - digitize(x) for x in action]
back = [round(centers[min(1000 - tid - 1, len(centers) - 1)], 2) for tid in ids]
print(ids)
print(back)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[998, 995, 992]
[-0.57, 0.29, 0.86]
```

这个小例子保留了源码里最重要的控制结构，方便你先写最小版再回到工程实现。

The small example keeps the most important control structure from the source, so you can write the minimal version before returning to the production implementation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **训练/推理边界** / **Training/inference boundaries**: 这类代码常把数学步骤转换成工程契约。 / This kind of code turns a mathematical step into an engineering contract.
- **nano 系统实现** / **Nano-system implementation**: 从这些片段抽象出的接口可以直接变成你自己的最小模块。 / The interface abstracted from these snippets can become your own minimal module.

## 注意事项 / Caveats / when it breaks

- **边界条件** / **Boundary cases**: 空输入、短 batch、短视频或越界动作通常最容易出 bug。 / Empty inputs, short batches, short videos, or out-of-range actions are the easiest places to break.
- **契约要写测试** / **Test the contract**: 这些函数依赖调用方遵守 shape、顺序和状态生命周期。 / These functions rely on callers respecting shapes, ordering, and state lifetime.

## 延伸阅读 / Further reading

- [openvla/openvla source](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py#L13-L70)
