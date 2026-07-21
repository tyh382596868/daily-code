---
date: 2026-07-15
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_inductor/fx_passes/group_batch_fusion.py
permalink: https://github.com/pytorch/pytorch/blob/00e2f26bad872c79e06d1b36926bbce8a2b05fa2/torch/_inductor/fx_passes/group_batch_fusion.py#L509-L608
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, inductor, graph-rewrite]
---

# PyTorch Inductor：先探测 tensor subclass，再合并 Linear / PyTorch Inductor: Probe Tensor Subclasses Before Fusing Linear

> **一句话 / In one line**: `BatchLinearLHSFusion` 把多个共享输入的 linear 合成一次大 `mm`，但先用一次小 `torch.cat` 探测 tensor subclass 是否支持该融合。 / `BatchLinearLHSFusion` merges several linears with the same input into one large `mm`, but first probes whether tensor subclasses can survive the required `torch.cat`.

## 为什么重要 / Why this matters

编译器优化经常不是“看到模式就替换”这么简单。这里的融合会把多个 `linear(x, w_i)` 的权重沿输出维拼起来，做一次 `x @ cat(w_i).T`，再 split 回多个结果。普通 tensor 没问题；但量化权重可能是 tensor subclass，不一定实现 `aten.cat`。PyTorch 的做法不是一刀切禁用 subclass，而是先试一个极小的 cat，能跑就融合，不能跑就放弃。

Compiler rewrites are rarely just “match and replace.” This pass concatenates the weights from several `linear(x, w_i)` calls, runs one `x @ cat(w_i).T`, then splits the output back. Plain tensors work, but quantized weights may be tensor subclasses without `aten.cat` support. PyTorch avoids a blanket ban: it probes a tiny cat first, fuses if it works, and skips otherwise.

## 代码 / The code

`pytorch/pytorch` — [`torch/_inductor/fx_passes/group_batch_fusion.py`](https://github.com/pytorch/pytorch/blob/00e2f26bad872c79e06d1b36926bbce8a2b05fa2/torch/_inductor/fx_passes/group_batch_fusion.py#L509-L608)

```python
@register_fusion("batch_linear_lhs")
class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> tuple[str, int | None, Any] | None:
        if CallFunctionVarArgs([torch.nn.functional.linear, torch._C._nn.linear]).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            # Skip fusion when weight is a tensor subclass that can't handle aten.cat.
            # fuse() calls torch.cat on the weight example_values, which fails for
            # subclasses lacking aten.cat dispatch.  Probe it here so subclasses that
            # do implement aten.cat (e.g. future torchao versions) still get fused.
            weight_val = weight.meta.get("example_value", weight.meta.get("val"))
            if weight_val is not None and type(weight_val) not in (
                torch.Tensor,
                FakeTensor,
            ):
                try:
                    _ = torch.cat([weight_val[:1], weight_val[:1]], dim=0)
                except (RuntimeError, TypeError):
                    return None
            bias_tensor = None
            if bias is not None:
                bias_tensor = bias.meta.get("val", bias.meta.get("example_value"))
            bias_dim = None if bias_tensor is None else bias_tensor.ndim
            group_key = ("batch_linear_lhs", bias_dim, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_input = None
        batch_weights, batch_weights_meta = [], []
        batch_biases, batch_biases_meta = [], []
        split_sections = []
        for node in subset:
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                if batch_input is not input:
                    raise AssertionError(
                        f"expected batch_input to be input, got {batch_input}"
                    )
            batch_weights.append(weight)
            batch_weights_meta.append(weight.meta["example_value"])
            if bias:
                batch_biases.append(bias)
                batch_biases_meta.append(bias.meta["example_value"])
            split_sections.append(weight.meta["example_value"].shape[0])

        with graph.inserting_before(subset[0]):
            cat_weights = graph.call_function(
                torch.cat, args=(batch_weights,), kwargs={"dim": 0}
            )
            cat_weights.meta["example_value"] = torch.cat(batch_weights_meta, dim=0)
            transposed_weights = graph.call_function(
                torch.transpose, args=(cat_weights, 0, 1)
            )
```

## 逐行讲解 / What's happening

1. **`group_key = ("batch_linear_lhs", bias_dim, input)`**:
   - 中文: 只有共享同一个 `input`、bias 维度兼容的 linear 才会被放进同一组。
   - English: Only linears sharing the same `input` and compatible bias rank land in the same fusion group.
2. **subclass probe**:
   - 中文: 对非普通 `torch.Tensor` / `FakeTensor`，先切一行做 `torch.cat`；失败就返回 `None`，让原图保持不变。
   - English: For non-plain tensor values, it slices one row and tries `torch.cat`; failure returns `None`, leaving the graph unchanged.
3. **`split_sections`**:
   - 中文: 每个原始 weight 的输出通道数被记录下来，后面大 `mm` 的输出才能 split 回原来的多路结果。
   - English: Each weight's output width is recorded so the large `mm` output can be split back into the original result tensors.
4. **metadata update**:
   - 中文: graph rewrite 不只要插节点，还要同步 `example_value`，后续 pass 才能继续用形状和 dtype 做判断。
   - English: The rewrite inserts graph nodes and updates `example_value`, so later passes still have shape and dtype metadata.

## 类比 / The analogy

这像把多个小包裹合成一个大箱子发快递：前提是所有包裹都去同一个仓库，且胶带能粘住这种材质。tensor subclass probe 就是先拿边角料试粘一下；粘不住，就别把整批货打包。

It is like consolidating many small parcels into one large shipment: they must go through the same warehouse, and the tape must work on that material. The tensor-subclass probe tests a scrap first; if the tape fails, the shipment stays split.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

x = np.array([[1., 2.]])
w1 = np.array([[1., 0.], [0., 1.]])
w2 = np.array([[2., 0.]])

separate = [x @ w1.T, x @ w2.T]
cat_w = np.concatenate([w1, w2], axis=0)
fused = x @ cat_w.T
split = np.split(fused, [w1.shape[0]], axis=1)

print(separate)
print(split)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[array([[1., 2.]]), array([[2.]])]
[array([[1., 2.]]), array([[2.]])]
```

中文: 大矩阵乘和多个小 `linear` 结果一致；编译器优化的价值在于少一次次 launch 和调度。

English: The large matmul matches the separate linears; the optimization saves repeated launches and scheduling overhead.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TorchDynamo guards** / **TorchDynamo guards**: 先验证形状、dtype、对象身份，再决定缓存图能不能复用。 / It validates shape, dtype, and identity before reusing a cached graph.
- **Inductor pattern rewrites** / **Inductor pattern rewrites**: 很多 pass 都用 metadata 先证明 rewrite 安全，再改 FX graph。 / Many passes use metadata to prove a rewrite is safe before mutating the FX graph.

## 注意事项 / Caveats / when it breaks

- **probe 不是免费 / The probe is not free**: 它发生在编译期，不在每步运行期，但对大量候选节点仍有成本。 / It runs at compile time rather than every iteration, but many candidates still add cost.
- **只证明 `cat` 能跑 / It only proves `cat` works**: subclass 还可能在后续 `mm` 或 transpose 路径出问题。 / A subclass may pass `cat` but still fail in later `mm` or transpose paths.
- **共享输入是假设 / Shared input is the assumption**: 如果输入不是同一个 FX node，这个 LHS fusion 不适用。 / If the linears do not share the same FX input node, this LHS fusion does not apply.

## 延伸阅读 / Further reading

- [PyTorch `group_batch_fusion.py`](https://github.com/pytorch/pytorch/blob/00e2f26bad872c79e06d1b36926bbce8a2b05fa2/torch/_inductor/fx_passes/group_batch_fusion.py)
- [PR 189509 commit](https://github.com/pytorch/pytorch/commit/26628087023e5102381849e9a27c3f09b6d485a4)
