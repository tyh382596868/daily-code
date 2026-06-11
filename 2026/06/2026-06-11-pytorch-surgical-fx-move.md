---
date: 2026-06-11
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_inductor/fx_passes/overlap_manual_scheduling.py
permalink: https://github.com/pytorch/pytorch/blob/19791183fec14fa4a6dc3a82004ed29cd4dc6704/torch/_inductor/fx_passes/overlap_manual_scheduling.py#L43-L112
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, pytorch, inductor, fx-graph, fsdp, compute-comm-overlap]
---

# PyTorch Inductor 把"全图拓扑排序"换成了"局部 BFS 搬运" — 18000 个节点中只动几十个 / PyTorch Inductor replaces a whole-graph topo sort with two tiny BFS helpers — out of 18 000 nodes it now moves only a few dozen

> **一句话 / In one line**: 重排 FSDP all-gather/reduce-scatter 用来 overlap compute 时,不要重新拓扑排序整张图;只要正向 BFS 收集"必须在 RS-wait 之后的节点",反向 BFS 收集"必须在 AG-start 之前的节点",然后用 `fx.Node.prepend` / `append` 就地搬运。 / Reordering FSDP all-gather / reduce-scatter for compute-comm overlap doesn't need a full topo sort — just two small BFSes (one forward from the RS wait, one backward from the AG start) and use `fx.Node.prepend` / `append` to surgically slide those chains into place.

## 为什么重要 / Why this matters

`torch.compile` 编译 FSDP 训练时,会在 FX 图上插入 collective op(all-gather、reduce-scatter)。要让通信和计算 overlap,需要把 AG 的"开始"提前到能与某段计算并行,把 RS 的"等待"延后到这段计算之后。**老版**的 `ManualOverlapScheduler` 是"先标记移动方向,再对整张图做稳定拓扑排序"。在 Llama-70B 上这意味着 **18,624 个 FX 节点里有 18,147 个被搬动** — 97% 的节点其实只是被同一个全图排序"路过"了一遍,真正需要换位的只有 ~30 条 AG/RS 链。PR #184711 做的事情非常漂亮:**把一次全局排序拆成 K 次局部 BFS**,K 是你要重排的链的数量。这是编译器里一类经典优化模式 — "**O(N+E) 全图改写 → O(K·V) 表层修补**" — 跟 LLVM 里的 dominator-tree update 比"重建支配树"省的功夫是一个味道。代码里两个 BFS 是教科书级别的简洁,值得作为"FX 图改写"的入门读本。

When `torch.compile` lowers an FSDP training step, it sprinkles collective ops (all-gather, reduce-scatter) into the FX graph. To overlap compute and comm, you need to slide the AG-start earlier (so it runs alongside some forward compute) and slide the RS-wait later (so the matmul that follows it can run while RS is in flight). The **old** `ManualOverlapScheduler` did this by labeling intended moves and then **re-topo-sorting the entire graph**. On Llama-70B that meant **18,147 of 18,624 FX nodes were physically moved** — 97% of those moves were no-ops, the topo sort just happened to walk past them. Only ~30 AG/RS chains actually had to shift. PR #184711 reframes the whole thing: **replace one global sort with K small BFSes**, where K is the number of chains to move. It's a classic compiler optimization pattern — "**rewrite O(N+E) of the graph → patch only O(K·V)**" — the same vibe as updating an LLVM dominator tree instead of rebuilding it. The two BFS helpers are textbook-tidy and well worth reading as an FX graph-rewrite primer.

## 代码 / The code

`pytorch/pytorch` — [`torch/_inductor/fx_passes/overlap_manual_scheduling.py`](https://github.com/pytorch/pytorch/blob/19791183fec14fa4a6dc3a82004ed29cd4dc6704/torch/_inductor/fx_passes/overlap_manual_scheduling.py#L43-L112)

```python
def _collect_nodes_must_be_after(node: fx.Node) -> list[fx.Node]:
    """BFS forward collecting node and its transitive users with no external inputs."""
    result: list[fx.Node] = [node]
    result_set: OrderedSet[fx.Node] = OrderedSet([node])
    i = 0
    while i < len(result):
        for user in result[i].users:
            if user not in result_set and all(
                inp in result_set for inp in user.all_input_nodes
            ):
                result_set.add(user)
                result.append(user)
        i += 1
    return result


def _collect_nodes_must_be_before(
    node: fx.Node, node_positions: dict[fx.Node, int]
) -> list[fx.Node]:
    """BFS backward collecting node and its non-placeholder dependencies, topo-sorted."""
    visited: OrderedSet[fx.Node] = OrderedSet()
    queue = [node]
    while queue:
        cur = queue.pop()
        if cur in visited or cur.op == "placeholder":
            continue
        visited.add(cur)
        queue.extend(cur.all_input_nodes)
    return sorted(visited, key=lambda n: node_positions[n])


def _move_overlap_nodes(
    graph: fx.Graph,
    overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    bucketed_node_types: dict[fx.Node, str],
) -> None:
    if not overlap_deps:
        return

    rs_defer: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    ag_prefetch: dict[fx.Node, list[fx.Node]] = defaultdict(list)

    for target, sources in overlap_deps.items():
        for source in sources:
            source_type = bucketed_node_types.get(source, "")
            if source_type.startswith("bucketed_reduce_scatter"):
                rs_defer[target].append(source)
            elif source_type.startswith("bucketed_all_gather"):
                ag_prefetch[target].append(source)

    node_positions = {n: i for i, n in enumerate(graph.nodes)}

    for rs_wait, rs_starts in rs_defer.items():
        latest_rs_start = max(rs_starts, key=lambda n: node_positions[n])
        node_insert_after = latest_rs_start
        for node in _collect_nodes_must_be_after(rs_wait):
            node_insert_after.append(node)
            node_insert_after = node

    # Recompute positions after RS moves
    node_positions = {n: i for i, n in enumerate(graph.nodes)}

    for ag_wait, ag_prefetch_starts in ag_prefetch.items():
        ag_wait_pos = node_positions[ag_wait]
        sorted_starts = sorted(ag_prefetch_starts, key=lambda n: node_positions[n])
        for ag_start in sorted_starts:
            if node_positions[ag_start] < ag_wait_pos:
                continue
            for node in _collect_nodes_must_be_before(ag_start, node_positions):
                ag_wait.prepend(node)
```

## 逐行讲解 / What's happening

1. **`_collect_nodes_must_be_after`(正向 BFS)**:
   - 中文: 从某个节点出发,沿着 `users`(谁用了我的输出)向前 BFS,但**只收闭包成员** — 一个 user 只有在它的所有输入都已经在结果集里时才被加进来。换句话说,这个集合永远是"自包含"的:把它整体搬到任何地方,所有依赖关系仍然成立。这是搬运 RS-wait 之后那条链(unpack、cast 等小算子)的关键。
   - English: forward BFS along `users`, but **only admit closure members** — a user joins the result set only if *all* its inputs are already in. The output set is therefore self-contained: move it anywhere as a unit and every dependency stays intact. This is what lets us slide the RS-wait's downstream chain (unpacks, casts, dtype conversions) into a new spot.

2. **`_collect_nodes_must_be_before`(反向 BFS + 拓扑排序)**:
   - 中文: 从一个 AG-start 节点反着走 `all_input_nodes`,直到遇到 `placeholder`(图的输入)就停。然后用提前算好的 `node_positions`(原图里的位置)对结果排序 — 这样最后插入图里时**保持依赖顺序**。
   - English: backward BFS along `all_input_nodes`, stopping at `placeholder` ops (graph inputs). Then sort by the original `node_positions` dict so the chain is reinserted in topological order. The trick is using the *pre-recorded* positions to topo-sort *after the fact* without re-walking the whole graph.

3. **`rs_defer` 和 `ag_prefetch` 的分类**:
   - 中文: 同一个 `overlap_deps` 字典里同时编码"AG 要提前到 X 之前"和"RS 要延后到 Y 之后"。代码先按 `bucketed_node_types[source]` 的前缀拆成两个字典 — `bucketed_reduce_scatter`(RS 延后)和 `bucketed_all_gather`(AG 提前)。
   - English: a single `overlap_deps` dict encodes both "AG must precede X" and "RS must follow Y" intents. The first 9 lines split it by the `bucketed_node_types[source]` prefix into two dicts: `rs_defer` (post-RS chains to push later) and `ag_prefetch` (pre-AG chains to pull earlier).

4. **`latest_rs_start = max(rs_starts, key=...)` + `_collect_nodes_must_be_after(rs_wait)`**:
   - 中文: 对每个 RS-wait 节点,先找到这一组 RS-start 中**位置最靠后**的那个(因为 wait 必须排在所有 starts 之后),然后把 RS-wait 自带的依赖闭包用 `fx.Node.append(node)` 一个一个挂到它后面。`node_insert_after = node` 这一行让链像穿珠子一样按顺序串起来。
   - English: for each RS-wait, pick the **latest** RS-start (the wait must come after *all* starts), then thread the wait's downstream closure behind it using `fx.Node.append`. The `node_insert_after = node` reassignment is the trick that strings the closure into a clean linked list — each new node hangs off the previous one.

5. **`node_positions = {...}` 重新计算位置**:
   - 中文: RS 链搬完之后,图的拓扑位置变了。重新构建一次 `node_positions` 字典(就是把 `graph.nodes` 枚举一遍 — 仍是 O(N) 但只跑一次,而不是 O((N+E)·logN))。然后 AG 的逻辑用更新后的位置继续工作。
   - English: after the RS moves the graph's node ordering has changed. Rebuild `node_positions` once (linear enumeration — still O(N), but a single pass — vs. the old code's full re-topo-sort each call). Then the AG step uses the updated positions.

6. **`ag_wait.prepend(node)` 提前 AG**:
   - 中文: 对每个 AG-wait,反向 BFS 收集它依赖的 AG-start 闭包,然后**整体塞到 AG-wait 前面**。如果某个 AG-start 已经在 AG-wait 之前就用 `continue` 跳过 — 已经满足约束,不要白动。这条 `continue` 优化掉了一大半冗余搬运。
   - English: for each AG-wait, BFS backward from its AG-start, collect the closure, and **prepend** it just before AG-wait. If an AG-start is already before AG-wait in the current order (`node_positions[ag_start] < ag_wait_pos`), `continue` — constraint already satisfied, no work. That one `continue` skips a huge fraction of redundant moves.

## 类比 / The analogy

想象 18000 行的 Python 文件,你要把其中**几段函数**搬到合适的位置 — 比如 `def all_gather_start` 要挪到 `def matmul` 之前,`def rs_wait` 要挪到一段计算之后。**老方法**像是把整个文件读到内存,根据"先后约束"重新排版整本书 — 排好后大部分行其实没动,但你浪费了把 18000 行扫一遍的功夫。**新方法**就是 IDE 里的 "Move method" 重构:你说"这个函数及其所有局部依赖一起搬",IDE 找出最小闭包,挑出原文件里的几行,贴到新位置 — 没动到的代码完全不知道发生了什么。

Picture an 18 000-line Python file where you want to move **a few functions** around — say `def all_gather_start` should land before `def matmul`, and `def rs_wait` should sit after a compute block. The **old approach** is "load the whole file, re-sort everything by some constraint" — most lines don't actually move, but you've still done the work of touching all 18 000. The **new approach** is the IDE's "Move method" refactor: you say "move this function plus its private closure to here," the IDE figures out the minimal set, cuts those lines, pastes them to the new spot, and untouched code never knew anything happened.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.fx as fx
from collections import OrderedDict

class M(torch.nn.Module):
    def forward(self, x, y):
        a = x + 1
        b = a * 2
        c = y + 10
        d = c * 3
        return b + d

gm = fx.symbolic_trace(M())
print("before:")
for n in gm.graph.nodes: print(" ", n.name, "<-", [i.name for i in n.all_input_nodes])

# move `d` to right before the return (it's already there, but pretend it isn't)
target = next(n for n in gm.graph.nodes if n.name == "mul_1")
ret = next(n for n in gm.graph.nodes if n.op == "output")
# tiny "must-be-before" BFS
def collect_before(node):
    seen, q = OrderedDict(), [node]
    while q:
        cur = q.pop()
        if cur in seen or cur.op == "placeholder": continue
        seen[cur] = None; q.extend(cur.all_input_nodes)
    pos = {n: i for i, n in enumerate(gm.graph.nodes)}
    return sorted(seen, key=lambda n: pos[n])

for n in collect_before(target): ret.prepend(n)
gm.graph.lint()
print("after:")
for n in gm.graph.nodes: print(" ", n.name)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
before:
  x <- []
  y <- []
  add <- [x]
  ...
after:
  x
  y
  ...
  output
```

中文:`graph.lint()` 在这里很关键 — 它会立刻爆出"如果有节点在它的输入之前出现"。换句话说,只要 BFS 闭包正确,你随便 `prepend` / `append`,这个 lint 帮你检查正确性。

English: `graph.lint()` is the unsung hero — it complains the instant any node appears before its inputs. So as long as the BFS closures are correct, you can `prepend` / `append` freely and let lint verify topological soundness.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LLVM `DominatorTree::recalculate` vs incremental updates** / **LLVM 支配树的增量更新**: 中文: 早期 LLVM 改一处 CFG 就重建整棵支配树;现在用 `DomTreeUpdater` 只更新受影响的节点。 / English: early LLVM rebuilt the entire dominator tree on any CFG edit; today's `DomTreeUpdater` patches only affected nodes — same "surgical, not global" philosophy.
- **MLIR's `IRRewriter` patterns** / **MLIR 的 IR 重写模式**: 中文: 一个 Rewriter 只重写匹配到的子图,而不是重建整个 module。 / English: a rewriter touches only the matched subgraph, not the enclosing module.
- **Postgres' `pg_rewrite` partial index updates** / **Postgres 部分索引的增量重建**: 中文: 写入一行只更新这一行涉及到的索引页,不会重建整张索引。 / English: a write only touches the index pages containing the affected row, not the whole index.
- **Git's `git rebase --onto` vs full history rewrite** / **Git 局部 rebase**: 中文: rebase 只搬运你指定的那几个 commit,而不是把整个分支线性化一遍。 / English: rebase moves only the commits you specify, not the whole branch — surgical history edits, same philosophy.

## 注意事项 / Caveats / when it breaks

- **`_collect_nodes_must_be_after` 的"自包含"约束 / The closure constraint**:
  - 中文: 这个函数要求 user 的**所有输入**都已在集合里,否则不收。它不是普通的 BFS — 漏掉这个约束,你搬运一组节点时会扯断其它依赖。
  - English: a user joins only if **all** its inputs are already in the closure. This is *not* a plain BFS — drop that condition and you'll move a set whose dependencies dangle outside.
- **`node_positions` 必须及时重算 / Recompute positions after each phase**:
  - 中文: 代码在 RS 阶段和 AG 阶段之间显式重建了一次 `node_positions`。如果你忘记,AG 阶段的位置信息是过时的,可能会导致"已经满足"的 fast path 误判。
  - English: the code explicitly rebuilds `node_positions` between the RS and AG phases. Skip this step and AG's "already satisfied" `continue` shortcut becomes unreliable, sometimes moving things that don't need moving.
- **`graph.lint()` 在 dev 环境很有用 / Lint is your safety net**:
  - 中文: 任何手动 FX 改写都建议在 dev 模式跑一遍 lint。线上把它关掉 — 它对一张 18000 节点的图本身就要 O(N) 时间。
  - English: any hand-rolled FX rewrite should be lint-checked in dev. Disable in prod, though — lint itself is O(N) on an 18 000-node graph.
- **`fx.Node.prepend` / `append` 是 O(1) 的双向链表操作 / They're O(1) doubly-linked list ops**:
  - 中文: 这就是为什么这个 PR 能从 O(N) 降到 O(K·V) — `fx.Node` 内部用的是 prev/next 指针,不是数组。如果换成数组就完全没有意义了。
  - English: this is why the PR's complexity drops from O(N) to O(K·V) — `fx.Node` is intrusively doubly-linked, so prepend/append are O(1). If it were a list, the cost model would collapse back to the old behavior.

## 延伸阅读 / Further reading

- [PR #184711 — "Replace whole-graph topo sort with surgical FX node moves"](https://github.com/pytorch/pytorch/pull/184711)
- [PyTorch docs — `torch.fx.Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph)
- [FSDP2 design — compute/comm overlap fundamentals](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- [LLVM blog — Incremental DominatorTree updates](https://blog.llvm.org/posts/2017-12-19-incremental-dominator-tree-update/)
