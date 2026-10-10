---
date: 2026-10-10
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L18-L36
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, vae-cache]
build_role: vae-encoder-decoder advanced variant
---
# Wan2.1 CausalConv3d：视频 VAE 只从过去补帧 / Wan2.1 CausalConv3d: A Video VAE Pads from the Past

> **一句话 / In one line**: 这层把 Conv3d 的时间 padding 改成只在过去侧补，并允许用 cache 接上上一段视频。 / This layer changes Conv3d temporal padding to past-only padding and can stitch in cached frames from the previous chunk.

## 为什么重要 / Why this matters

视频 VAE 分块编码时不能偷看未来。Wan2.1 先把历史 cache 拼到当前 chunk 前面，再扣掉对应 padding，让普通 Conv3d 获得因果上下文。

Chunked video VAEs must avoid peeking into the future. Wan2.1 concatenates cached history before the current chunk, then reduces padding so ordinary Conv3d receives causal context.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L18-L36)

```python
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)
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

像剪辑视频时接片头：新片段开头缺上下文，就拿上一段最后几帧垫上；不够时才补黑场。

It is like editing a clip: borrow the previous segment’s final frames for context, and only add blank padding if history is missing.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 `vae-encoder-decoder` 的高级变体：它把上游数据变成下游模型能稳定消费的表示。省掉它，nano 系统仍可跑通玩具例子，但会在真实动作空间或视频 chunk 边界上迅速暴露问题。

This is an advanced variant of `vae-encoder-decoder`. It converts upstream data into a representation the downstream model can consume reliably. Without it, a nano system may run toy examples but will fail quickly on real action spaces or video chunk boundaries.

## 自己跑一遍 / Try it yourself

```python
def causal_window(chunk, cache=None, kernel=3):
    need = kernel - 1
    padded = ([] if cache is None else list(cache)[-need:]) + list(chunk)
    while len(padded) < len(chunk) + need:
        padded = [0] + padded
    return [padded[i:i+kernel] for i in range(len(chunk))]
print(causal_window([10, 11], cache=[7, 8, 9]))
print(causal_window([10, 11]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[8, 9, 10], [9, 10, 11]]
[[0, 0, 10], [0, 10, 11]]
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

- [Wan-Video/Wan2.1 source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L18-L36)
