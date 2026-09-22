---
date: 2026-09-22
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/distributed/device_mesh.py
permalink: https://github.com/pytorch/pytorch/blob/1a080e5af57ac14b5c73058c7e1b2e889a371745/torch/distributed/device_mesh.py#L910-L974
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, distributed, device-mesh, failure-handling]
---

# PyTorch DeviceMesh abort：失败时先拆掉通信网 / PyTorch DeviceMesh Abort: Tear Down the Communication Mesh on Failure

> **一句话 / In one line**: `DeviceMesh.abort()` 收集 root mesh 拥有的全部 process group，去重后协调 abort，避免某个 rank 提前退出而让其它 rank 永久等待。 / `DeviceMesh.abort()` collects every process group owned by the root mesh, deduplicates them, and coordinates their abort so one failed rank does not leave the others waiting forever.

## 为什么重要 / Why this matters

分布式训练最难受的错误往往不是异常本身，而是“有人已经死了，其他 rank 还在 collective 里等”。旧式的串行 `for group in mesh.get_all_groups(): group.abort()` 可能在 NCCL 多通信器场景下再次死锁，因为不同 communicator 的终止需要 group semantics。

The most painful distributed failures are often not the original exception but the hang that follows it. A naïve serial loop over `mesh.get_all_groups()` can deadlock when several NCCL communicators must be aborted together, because communicator teardown needs group semantics.

这个新 API 把失败路径也当成一等公民：root mesh 才能负责整个作用域；dimension group 和 flattened mesh group 都要收集；同一个 process group 只能 abort 一次；最后还要清理 PyTorch 的全局 process-group 状态。

The new API treats the failure path as a first-class protocol. Only a root mesh has an unambiguous scope; both dimension groups and flattened-mesh groups are collected; each process group is aborted once; and PyTorch's global process-group state is cleaned afterward.

## 代码 / The code

`pytorch/pytorch` — [`torch/distributed/device_mesh.py`](https://github.com/pytorch/pytorch/blob/1a080e5af57ac14b5c73058c7e1b2e889a371745/torch/distributed/device_mesh.py#L910-L974)

```python
def abort(self) -> None:
    """
    Abort all process groups associated with this DeviceMesh.

    When a rank in a DeviceMesh exits prematurely, other ranks can call
    this method to abort the mesh's process groups instead of hanging.

    This method must be called on a root mesh. Calling it on a submesh
    raises ``RuntimeError`` because the scope of a submesh abort is
    ambiguous.
    """
    from torch.distributed.distributed_c10d import (
        _cleanup_process_group_global_state,
    )

    if self._root_mesh is not None:
        raise RuntimeError(
            "abort() is not supported on a submesh. "
            "Call abort() on the root mesh instead."
        )

    if not hasattr(self, "_dim_group_names"):
        return

    seen: set[str] = set()
    pgs: list[ProcessGroup] = []
    for name in self._dim_group_names:
        if name not in seen:
            seen.add(name)
            pg = _resolve_process_group(name)
            if pg is not None:
                pgs.append(pg)

    for flat_mesh in self._flatten_mapping.values():
        if hasattr(flat_mesh, "_dim_group_names"):
            for name in flat_mesh._dim_group_names:
                if name not in seen:
                    seen.add(name)
                    pg = _resolve_process_group(name)
                    if pg is not None:
                        pgs.append(pg)

    if not pgs:
        return

    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    try:
        pgs[0]._start_coalescing(device)
        coalescing = True
    except RuntimeError:
        coalescing = False

    for pg in pgs:
        pg.abort()

    if coalescing:
        pgs[0]._end_coalescing(device)

    for pg in pgs:
        _cleanup_process_group_global_state(pg)
```

## 逐行讲解 / What's happening

1. **第 930-937 行 / Lines 930-937 (scope checks)**:
   - 中文: submesh 不能猜“自己的 abort 范围”是不是还包括 root 的其它 group，所以直接失败；未初始化的 mesh 则安静返回。
   - English: A submesh cannot know whether its abort scope should include groups owned by the root, so it fails explicitly. An uninitialized mesh returns quietly.
2. **第 939-955 行 / Lines 939-955 (collect and deduplicate)**:
   - 中文: `seen` 按名字去重，先扫普通 dimension groups，再扫 flatten mapping，避免同一个 communicator 被重复处理。
   - English: `seen` deduplicates by group name while scanning both ordinary dimension groups and flattened meshes, preventing repeated treatment of one communicator.
3. **第 960-968 行 / Lines 960-968 (coalesced abort)**:
   - 中文: 尝试打开 coalescing；如果后端不支持就退回普通串行 abort。循环本身保持简单，后端差异被压在两个边界调用里。
   - English: The method tries coalescing and falls back to ordinary abort if the backend rejects it. Backend differences stay at the two boundary calls while the loop remains simple.
4. **第 970-974 行 / Lines 970-974 (cleanup)**:
   - 中文: abort 不是终点；每个 group 的全局 registry 也要清理，否则后续重建或错误处理可能看到过期状态。
   - English: Aborting is not the end. Global registry state must also be cleaned so later recovery or teardown does not observe stale process groups.

## 类比 / The analogy

想象一座有多条桥的工厂。某个工人发现主电源已经断了，不能只关掉自己那条传送带；要先通知所有桥梁同时落闸，再从总控台删除这些桥的登记。

Imagine a factory connected by several bridges. When one worker discovers that the main power is gone, shutting down only one conveyor is not enough. Every bridge must close under one coordinated signal, and the control room must remove the bridge registrations afterward.

## 自己跑一遍 / Try it yourself

```python
class Group:
    def __init__(self, name):
        self.name, self.calls = name, 0
    def abort(self):
        self.calls += 1

def abort_groups(names):
    groups = {name: Group(name) for name in set(names)}
    for group in groups.values():
        group.abort()
    return [(name, group.calls) for name, group in groups.items()]

print(sorted(abort_groups(["dp", "tp", "dp", "flat_dp"])))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[('dp', 1), ('flat_dp', 1), ('tp', 1)]
```

中文: 失败清理的第一条规则是“一个资源只终止一次”；真实 API 还把这个规则扩展到了 NCCL coalescing 和全局状态。 / English: The first teardown rule is “terminate each resource once”; the real API extends that rule to NCCL coalescing and global state.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NCCL group semantics** / **NCCL group semantics**: 中文: 多个 communicator 的操作需要成组提交，否则一个 rank 可能在中途阻塞。 / English: Operations across several communicators may need grouped submission or one rank can block halfway through teardown.
- **CUDA graph cleanup** / **CUDA graph cleanup**: 中文: 图执行失败后也要区分“停止执行”和“清理注册状态”。 / English: Failed graph execution similarly separates stopping work from cleaning registered state.
- **distributed job supervisors** / **distributed job supervisors**: 中文: 外层 launcher 通常负责 kill 全局进程，mesh 则负责释放自己拥有的通信资源。 / English: An outer launcher may kill processes globally, while the mesh owns the communication-resource cleanup.

## 注意事项 / Caveats / when it breaks

- **root scope 是硬契约** / **Root scope is a hard contract**: 中文: 对 submesh 调 `abort()` 会报错，不要用异常吞掉它再继续 collective。 / English: Calling `abort()` on a submesh is an error; do not swallow it and continue issuing collectives.
- **abort 不等于恢复** / **Abort is not recovery**: 中文: 它阻止 hang，但不会自动重建 optimizer、rank state 或 checkpoint。 / English: It prevents a hang but does not rebuild the optimizer, rank state, or checkpoint.
- **后端能力不同** / **Backends differ**: 中文: `_start_coalescing` 可能不支持；生产调用方要把 abort 当作 best-effort failure path。 / English: `_start_coalescing` may be unsupported, so callers should treat abort as a best-effort failure path.

## 延伸阅读 / Further reading

- [PyTorch `DeviceMesh.abort()`](https://github.com/pytorch/pytorch/blob/1a080e5af57ac14b5c73058c7e1b2e889a371745/torch/distributed/device_mesh.py#L910-L974)
- [PyTorch distributed collectives](https://pytorch.org/docs/stable/distributed.html)
- [NCCL group calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
