---
date: 2026-08-31
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/dce_extra_outputs.py
permalink: https://github.com/pytorch/pytorch/blob/4f0443d2fde3511401d06c083a607cd4d3581a01/torch/_dynamo/dce_extra_outputs.py#L31-L82
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, torch-dynamo, fx, dead-code-elimination]
---

# PyTorch HOP DCE：子图多余输出要一起删 / PyTorch HOP DCE: Remove Extra Subgraph Outputs Together

> **一句话 / In one line**: 这段 pass 找出 higher-order-op 子图没人用的额外输出，同时重写所有调用点的 `getitem` 索引。 / This pass finds unused extra outputs from higher-order-op subgraphs and rewrites every caller's `getitem` indices.

## 为什么重要 / Why this matters

TorchDynamo 和 FX pass 会把一段图包进 `invoke_subgraph`、checkpoint 这类 higher-order op。为了表达副作用，子图可能返回一串额外张量；如果父图没有使用这些输出，它们会让图更大、元数据更乱、后续优化更难。

TorchDynamo and FX passes can wrap regions in higher-order ops such as `invoke_subgraph` or checkpoint. To preserve side effects, subgraphs may return extra tensors. If parent graphs never use some outputs, they bloat the graph and complicate later optimization.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/dce_extra_outputs.py`](https://github.com/pytorch/pytorch/blob/4f0443d2fde3511401d06c083a607cd4d3581a01/torch/_dynamo/dce_extra_outputs.py#L31-L82)

```python
def dce_hop_extra_outputs(gm: torch.fx.GraphModule) -> bool:
    """
    Remove unused extra outputs from HOP calls in all submodules.

    For each subgraph output, check if any caller has a getitem for that index
    with users. If no caller uses it, remove the output.
    """
    subgraph_id_to_callers: dict[
        int, list[tuple[torch.fx.GraphModule, str, torch.fx.Node]]
    ] = collections.defaultdict(list)
    _collect_all_subgraph_usages(gm, subgraph_id_to_callers)

    if not subgraph_id_to_callers:
        return False

    modified = False

    for callers in subgraph_id_to_callers.values():
        parent_gm, subgraph_name, _ = callers[0]
        subgraph = getattr(parent_gm, subgraph_name)

        if not isinstance(subgraph, torch.fx.GraphModule):
            continue

        output_node = next(n for n in subgraph.graph.nodes if n.op == "output")
        output_args = output_node.args[0]
        if not isinstance(output_args, (tuple, list)):
            continue

        num_outputs = len(output_args)
        used_indices: set[int] = set()

        for idx in range(num_outputs):
            if _is_output_used(idx, callers):
                used_indices.add(idx)

        if 0 < len(used_indices) < num_outputs:
            if _dce_subgraph(subgraph, callers, used_indices):
                modified = True

    return modified
```

## 逐行讲解 / What's happening

1. **第 31 行 / Line 31 (`dce_hop_extra_outputs`)**:
   - 中文: pass 入口吃一个 `GraphModule`，返回 bool 表示图有没有被改。
   - English: The pass takes a `GraphModule` and returns whether it mutated the graph.
2. **第 46-50 行 / Lines 46-50 (collect callers)**:
   - 中文: 同一个子图可能被多个 HOP 节点调用，所以要按子图对象 ID 收集所有 caller。
   - English: A subgraph can be called from multiple HOP nodes, so callers are grouped by subgraph object id.
3. **第 64-67 行 / Lines 64-67 (only tuple outputs)**:
   - 中文: 只有 tuple/list 输出能安全按索引裁剪；单值输出没有“删第几个”的概念。
   - English: Only tuple/list outputs can be trimmed by index. A single output has no removable slots.
4. **第 72-80 行 / Lines 72-80 (all-callsite liveness)**:
   - 中文: 一个输出只要被任意 caller 的有效 `getitem` 使用，就必须保留。
   - English: If any caller has a live `getitem` for an output index, that output must stay.

## 类比 / The analogy

像餐厅后厨给每桌都上固定套餐。后来发现所有桌子都没人动第 4 道菜，厨房就把菜单改掉，同时服务员手里的“第几道菜”编号也要一起更新。

It is like a kitchen serving a fixed tasting menu. If no table eats course 4, the kitchen removes it, and every waiter's course numbering must be updated too.

## 自己跑一遍 / Try it yourself

```python
used = {0, 2, 4}
old_outputs = ["loss", "tmp_a", "logits", "tmp_b", "state"]
old_to_new = {}
new_outputs = []

for old_idx, value in enumerate(old_outputs):
    if old_idx in used:
        old_to_new[old_idx] = len(new_outputs)
        new_outputs.append(value)

print(new_outputs)
print(old_to_new)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['loss', 'logits', 'state']
{0: 0, 2: 1, 4: 2}
```

中文: 这就是 `_dce_subgraph` 的核心：先决定活跃输出，再建立旧索引到新索引的映射。

English: This is the core of `_dce_subgraph`: determine live outputs, then build the old-index to new-index map.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Compiler DCE** / **Compiler DCE**: 中文: 经典编译器也会先做 liveness，再删掉没人读的值。 / English: Classic compilers also compute liveness before removing unread values.
- **FX graph rewrites** / **FX graph rewrites**: 中文: 每次改节点输出形状，都必须同步更新下游索引和 metadata。 / English: Whenever node output structure changes, downstream indices and metadata must be updated together.

## 注意事项 / Caveats / when it breaks

- **多 caller 必须一起看 / Multiple callers must be checked together**: 中文: 只看一个调用点会误删另一个调用点需要的输出。 / English: Looking at one caller can delete an output another caller still needs.
- **索引重写不能漏 / Index rewrites cannot be partial**: 中文: 子图输出少了，父图的 `getitem(3)` 可能立刻变成错位访问。 / English: Once subgraph outputs shrink, a stale `getitem(3)` can point to the wrong value.

## 延伸阅读 / Further reading

- FX GraphModule docs: https://pytorch.org/docs/stable/fx.html
- PyTorch source: https://github.com/pytorch/pytorch/blob/4f0443d2fde3511401d06c083a607cd4d3581a01/torch/_dynamo/dce_extra_outputs.py
