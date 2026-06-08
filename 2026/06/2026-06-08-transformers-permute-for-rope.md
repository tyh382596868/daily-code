---
date: 2026-06-08
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/core_model_loading.py
permalink: https://github.com/huggingface/transformers/blob/d6a82ba896fcc59e6c40af95cbbd167dcf736e2f/src/transformers/core_model_loading.py#L381-L412
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, huggingface, transformers, rope, checkpoint]
---

# 把困扰 Llama 移植半年的 RoPE 重排压成两行 view + transpose / The two-line `view` + `transpose` that fixes Llama's RoPE port nightmare

> **一句话 / In one line**: 上游 Llama 权重把 RoPE 的 (real, imag) 通道交错存,transformers 内部用「先全 real,再全 imag」存,这段 30 行的 `PermuteForRope` 就是中间的格式转换器。 / Llama upstream stores RoPE channels as interleaved `(r, i, r, i, …)`; transformers stores them split as `(r…r, i…i)`. `PermuteForRope` is the 30-line converter between the two formats.

## 为什么重要 / Why this matters

任何手写过 Llama 推理代码的人都见过这个 bug:你从官方权重加载,生成出来的文字全是乱码,但 loss 看起来又"差不多对"。原因几乎一定是 RoPE 通道格式不对——meta 官方的实现里, attention head 的偶数维和奇数维当成复数的实部和虚部交错存放,而 transformers / GPT-NeoX / Mistral 等用「前半 dim 全是实部,后半全是虚部」的 split 格式。这俩在数值上几乎一样,但矩阵乘出来的结果完全不同。HF transformers 把这件事从「每个 model file 自己手写 `permute_for_llama` 函数」升级成了一个 `ConversionOps` graph 里的一个节点——四行代码,工程上可组合。

If you've ever ported Llama weights by hand, you've seen this bug: load the official `consolidated.pth`, generate, get gibberish, but the loss looks "almost right." The cause is almost always the RoPE channel format. Meta's reference treats even/odd dims of an attention head as the *real* and *imaginary* parts of a complex number, interleaved as `(r₀, i₀, r₁, i₁, …)`. transformers / GPT-NeoX / Mistral use the *split* layout `(r₀, r₁, …, i₀, i₁, …)`. The two have the same numbers but produce different matmuls. transformers just lifted this from a per-model hand-written `permute_for_llama()` into a single node in its new `ConversionOps` graph — four lines of tensor reshape, fully composable.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/core_model_loading.py`](https://github.com/huggingface/transformers/blob/d6a82ba896fcc59e6c40af95cbbd167dcf736e2f/src/transformers/core_model_loading.py#L381-L412)

```python
class PermuteForRope(ConversionOps):
    """
    Applies the permutation required to convert complex RoPE weights to the split sin/cos format.
    """

    def __init__(self):
        pass

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        dim1, dim2 = tensor.shape
        n_heads = self.config.getattr("num_attention_heads", 1)

        tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        tensor = tensor.transpose(1, 2).reshape(dim1, dim2)
        return tensor

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        self.config = config
        output: dict[str, list[torch.Tensor]] = {}
        for key, tensors in input_dict.items():
            if len(tensors) != 1:
                raise ValueError("PermuteForRope expects a single tensor per key.")
            output[key] = [self._apply(tensors[0])]
        return output
```

## 逐行讲解 / What's happening

整段算法只有两行:

The whole algorithm is two lines:

```python
tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
tensor = tensor.transpose(1, 2).reshape(dim1, dim2)
```

我们一步步看在 `q_proj.weight` (形状 `[n_heads * head_dim, hidden]`,head_dim = 128, n_heads = 32 → `[4096, 4096]`) 上发生了什么 / Let's walk through what happens to `q_proj.weight` with shape `[n_heads * head_dim, hidden] = [4096, 4096]` (head_dim=128, n_heads=32):

1. **`view(n_heads, dim1 // n_heads // 2, 2, dim2)`** → 形状变成 `[32, 64, 2, 4096]` / shape becomes `[32, 64, 2, 4096]`
   - 中文: 把第 0 维 `4096` 拆成 `(头数 32, 半个 head_dim 64, 通道 2)`。这里的 `2` 就是「交错的实/虚」那一对。
   - English: Split axis 0 of `4096` into `(num_heads=32, half_head_dim=64, ri_channel=2)`. That `2` is the "interleaved real/imag" pair.

2. **`transpose(1, 2)`** → 形状变成 `[32, 2, 64, 4096]` / shape becomes `[32, 2, 64, 4096]`
   - 中文: 把「实/虚通道」和「半 head dim」交换。这一步是核心:之前 `(64, 2)` 表示 "对于每一对儿,先 real 后 imag",换完是 `(2, 64)` 表示 "先列出所有 real,再列出所有 imag"。
   - English: Swap the `ri_channel` axis with the `half_head_dim` axis. This is the key. Before: `(64, 2)` meant "for each of 64 pairs, list real then imag." After: `(2, 64)` means "list all 64 reals, then list all 64 imags." Same numbers, different memory layout.

3. **`reshape(dim1, dim2)`** → 形状回到 `[4096, 4096]` / shape goes back to `[4096, 4096]`
   - 中文: flatten 回原始权重形状,但通道顺序已经从「interleaved」变成了「split half」。后续 `nn.Linear` 拿到的权重和 transformers 写的 RoPE 函数预期的就对上了。
   - English: Flatten back to the original weight shape, but the per-head channel ordering is now "split half" instead of "interleaved." Downstream `nn.Linear` matmuls against the RoPE rotation in the format transformers expects.

4. **`convert(input_dict, source_patterns, target_patterns, config)`**
   - 中文: 这是 `ConversionOps` 的统一接口——输入是 `{glob pattern: [tensor]}`,输出同形状的 dict。`config` 进来后存到 `self.config`,因为 `n_heads` 是模型相关的、checkpoint 自己不知道。
   - English: This is the unified `ConversionOps` interface: input is `{glob pattern: [tensor]}`, output is the same shape. The model `config` is passed in so `_apply` can read `num_attention_heads` — the checkpoint file itself doesn't know this.

5. **`raise ValueError("PermuteForRope expects a single tensor per key.")`**
   - 中文: 防御性检查。其他 ConversionOps 比如 `MergeModulelist` 一个 key 对应多 tensor (合并多个 expert),`PermuteForRope` 只对一份权重做就近 permutation。
   - English: Defensive: other `ConversionOps` like `MergeModulelist` map one key to many tensors (e.g. stacking MoE experts). `PermuteForRope` operates per-weight, so it asserts exactly one tensor.

## 类比 / The analogy

想象你买了一盒色拉酱,瓶子上的标签是「红色甜的——绿色酸的——红色甜的——绿色酸的」一颗一颗交错涂的圆点。结账员的扫码枪 (transformers 的 RoPE 函数) 只认「前半全是红圆点,后半全是绿圆点」的标签。`PermuteForRope` 就是收银台里那个把瓶子标签重画一遍的小机器——瓶子里装的酱完全没动,只是外面的图案重新排了一下顺序。

Picture a salad-dressing bottle whose label is dotted "red sweet — green sour — red sweet — green sour" in alternating pattern. The checkout scanner (transformers' RoPE function) only knows how to read labels where *all the reds come first, then all the greens*. `PermuteForRope` is the little re-stickering machine at the till: the sauce inside the bottle is unchanged, only the pattern on the outside gets rearranged.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch

def permute_for_rope(W, n_heads):
    """Llama-interleaved → transformers-split."""
    dim1, dim2 = W.shape
    W = W.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    W = W.transpose(1, 2).reshape(dim1, dim2)
    return W

n_heads, head_dim, hidden = 4, 8, 16
# Build a weight that ENCODES which dim is real vs imag in the leading axis:
# rows 0,2,4,... = real channels, rows 1,3,5,... = imag channels (per head, interleaved)
W = torch.zeros(n_heads * head_dim, hidden)
for h in range(n_heads):
    for d in range(head_dim):
        W[h * head_dim + d] = h * 100 + d  # encode (head, channel) as a value

print("BEFORE permute, head 0 channels (interleaved):")
print(W[:head_dim, 0].tolist())          # [0,1,2,3,4,5,6,7] = r0,i0,r1,i1,r2,i2,r3,i3

W2 = permute_for_rope(W, n_heads)
print("AFTER permute, head 0 channels (split):")
print(W2[:head_dim, 0].tolist())         # [0,2,4,6,1,3,5,7] = all reals then all imags
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
BEFORE permute, head 0 channels (interleaved):
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
AFTER permute, head 0 channels (split):
[0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]
```

中文:看后半段的输出——偶数索引 `[0,2,4,6]` 全跑到前面,奇数索引 `[1,3,5,7]` 全跑到后面。这就是 interleaved → split 转换的全部数学内容。

English: Look at the second output line — even indices `[0,2,4,6]` move to the front, odd `[1,3,5,7]` to the back. That's the entire math of "interleaved → split half."

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Meta 官方 Llama 推理脚本里的 `permute()`** / **Meta's official Llama inference `permute()`**: 几乎一模一样的代码,只是不是抽象成 ConversionOp。 / Nearly identical code, just not abstracted into a ConversionOp.
- **vLLM 的 `weight_loader_v2`** / **vLLM's `weight_loader_v2`**: 用同样的 view-transpose 在 GPU 上做 in-place 转换。 / Does the same view-transpose, but in place on the GPU at load time.
- **GPT-NeoX 的 `rotate_half`** / **GPT-NeoX's `rotate_half`**: 反过来——neox 在 RoPE 函数里用 split 格式,所以从 neox 加载 weights 不需要 permute (而从 llama-original 加载需要)。 / The mirror case — neox already uses the split layout in its RoPE function, so neox weights need no permute, but Llama-original weights do.
- **`mistral-common` 的 `convert_weights.py`** / **`mistral-common`'s `convert_weights.py`**: 同样的 permute,套了一层 `safetensors` API,因为 Mistral 沿用了 Llama 的 interleaved 格式。 / Same permute, wrapped in a `safetensors` API, because Mistral inherited Llama's interleaved layout.

## 注意事项 / Caveats / when it breaks

- **必须对 q_proj 和 k_proj 都做** / **Must run on BOTH q_proj and k_proj**: 只对 q 做不对 k 做,attention 还是错——`q · kᵀ` 出来的就是噪声。 / Permuting only `q_proj` and not `k_proj` is just as broken as not permuting at all — `q · kᵀ` then computes nonsense.
- **GQA 模型 head_dim 不变,但 n_heads 要小心** / **In GQA, `n_heads` for k_proj is `num_kv_heads`, not `num_attention_heads`**: ConversionOps graph 在配 `q_proj` 时用 `num_attention_heads`,配 `k_proj` 时要换成 `num_kv_heads`,否则 view 会拆出错误形状然后悄悄报错。 / The graph wiring must pass `num_kv_heads` to `k_proj`'s `PermuteForRope`. Use `num_attention_heads` for both and the view at the wrong size silently produces garbage.
- **不要对 v_proj 做** / **DO NOT permute v_proj**: v 不参与 RoPE,permute 它就是无端洗乱了权重。 / `v_proj` doesn't participate in RoPE — permuting it just scrambles weights for no reason.

## 延伸阅读 / Further reading

- [RoPE: Roformer paper (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [Meta's reference `permute()` in llama/model.py](https://github.com/meta-llama/llama/blob/main/llama/model.py)
- [transformers WeightTransform graph design doc](https://github.com/huggingface/transformers/blob/main/src/transformers/core_model_loading.py)
