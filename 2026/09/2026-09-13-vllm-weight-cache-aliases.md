---
date: 2026-09-13
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/model_executor/model_loader/weight_cache/daemon.py
permalink: https://github.com/vllm-project/vllm/blob/2671fedfc7ae604761990603fc736c0c4f21de57/vllm/model_executor/model_loader/weight_cache/daemon.py#L63-L101
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, weight-cache, cuda-ipc, tied-parameters, zero-copy]
---

# vLLM 权重缓存：同一块 tensor 只导出一次 / vLLM Weight Cache: Export One Tensor, Preserve Every Alias

> **一句话 / In one line**: vLLM 用 tensor 对象身份区分“新权重”和“同一权重的另一个名字”，把共享参数导出成一个实体加一张 alias 表。 / vLLM uses tensor object identity to distinguish a new weight from another name for the same weight, exporting one entity plus an alias table.

## 为什么重要 / Why this matters

中文：语言模型经常让 `lm_head.weight` 和 `embed_tokens.weight` 共享同一组参数。若权重缓存只按名字导出，加载时就会为两个名字分配两块内存，既浪费显存，也破坏 tied weight 的语义。`export_entries` 先允许重复名字出现，再用 `id(tensor)` 找到对象级别的共享关系。

English: Language models often tie `lm_head.weight` and `embed_tokens.weight` to the same parameter. A name-only export would allocate two copies and lose the tied-weight contract. `export_entries` keeps duplicate names visible, then uses `id(tensor)` to recover object-level sharing.

## 代码 / The code

`vllm-project/vllm` — [`vllm/model_executor/model_loader/weight_cache/daemon.py`](https://github.com/vllm-project/vllm/blob/2671fedfc7ae604761990603fc736c0c4f21de57/vllm/model_executor/model_loader/weight_cache/daemon.py#L63-L101)

```python
def export_entries(
    model: torch.nn.Module,
) -> tuple[dict[str, TensorEntry], dict[str, str]]:
    """Export a model's tensors, preserving tied-parameter aliases.

    ``named_parameters``/``named_buffers`` are iterated with
    ``remove_duplicate=False`` so tied weights (e.g. ``lm_head.weight`` sharing
    storage with ``embed_tokens.weight``) are not silently dropped. Each unique
    tensor is exported once; every additional name that refers to the same
    tensor object is recorded in the returned alias map so the client can
    re-establish the shared identity instead of allocating uninitialized
    memory for it.
    """
    entries: dict[str, TensorEntry] = {}
    aliases: dict[str, str] = {}
    canonical_by_id: dict[int, str] = {}

    def _add(name: str, tensor: torch.Tensor, kind: str) -> None:
        canonical = canonical_by_id.get(id(tensor))
        if canonical is not None:
            aliases[name] = canonical
            return
        canonical_by_id[id(tensor)] = name
        entries[name] = TensorEntry.from_tensor(tensor, kind)

    for name, param in model.named_parameters(remove_duplicate=False):
        _add(name, param, "param")
    # named_buffers includes non-persistent buffers (e.g. rotary embedding
    # caches) that state_dict would miss.
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in entries or name in aliases:
            continue
        _add(name, buffer, "buffer")
    return entries, aliases
```

## 逐行讲解 / What's happening

1. **第 63-83 行 / Lines 63-83**:
   - 中文：函数同时返回 `entries` 和 `aliases`。前者描述真正需要传输或缓存的 tensor，后者只描述名字之间的关系。
   - English: The function returns `entries` and `aliases` separately. The first describes tensors that must be transferred or cached; the second describes only name relationships.
2. **第 85-91 行 / Lines 85-91 (`_add`)**:
   - 中文：`canonical_by_id` 是一个对象身份索引。第一次看见 tensor 时，把当前名字定为 canonical；再次看见同一个对象时，只写 alias。
   - English: `canonical_by_id` indexes object identity. The first name becomes canonical; later names for the same object become aliases.
3. **第 93-94 行 / Lines 93-94**:
   - 中文：`remove_duplicate=False` 很关键。默认去重会让共享参数在遍历阶段就消失，之后再聪明的 alias 逻辑也无法恢复它。
   - English: `remove_duplicate=False` is essential. Default deduplication would erase the second name before alias reconstruction gets a chance.
4. **第 95-100 行 / Lines 95-100**:
   - 中文：buffer 也加入同一套规则，而且补上了 `state_dict` 可能漏掉的非持久 buffer，例如 rotary cache。
   - English: Buffers use the same rule, including non-persistent buffers such as rotary caches that a `state_dict` walk can miss.
5. **加载端的隐含契约 / The loader-side contract**:
   - 中文：客户端先创建 canonical tensor，再让 alias 名字指向同一个对象。alias 不是“再拷贝一份”的提示，而是“恢复共享身份”的指令。
   - English: The client creates the canonical tensor once, then points alias names at that same object. An alias means “restore shared identity,” not “copy this tensor again.”

## 类比 / The analogy

中文：像仓库里的货物和货架标签。货物本体只入库一次；`lm_head.weight` 和 `embed_tokens.weight` 是两个贴在同一箱货上的标签。缓存系统要记住两个标签，却不能把货物复制两次。

English: Think of inventory and shelf labels. The box is stored once, while `lm_head.weight` and `embed_tokens.weight` are two labels on the same box. The cache must remember both labels without duplicating the box.

## 自己跑一遍 / Try it yourself

```python
class Entry:
    def __init__(self, kind):
        self.kind = kind


def export(names):
    entries, aliases, canonical = {}, {}, {}
    for name, tensor, kind in names:
        key = id(tensor)
        if key in canonical:
            aliases[name] = canonical[key]
        else:
            canonical[key] = name
            entries[name] = Entry(kind)
    return entries, aliases


shared, separate = object(), object()
entries, aliases = export([
    ("embed_tokens.weight", shared, "param"),
    ("lm_head.weight", shared, "param"),
    ("layernorm.weight", separate, "param"),
])
print(sorted(entries), aliases)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
['embed_tokens.weight', 'layernorm.weight'] {'lm_head.weight': 'embed_tokens.weight'}
```

中文：`shared` 出现两次，但 `entries` 只有一个 canonical 项；第二个名字进入 `aliases`。真实 vLLM 只是把 `Entry` 换成带 shape、dtype、IPC 信息的 `TensorEntry`。

English: `shared` appears twice, but `entries` contains one canonical item and the second name goes into `aliases`. vLLM replaces the toy `Entry` with a `TensorEntry` carrying shape, dtype, and IPC information.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch tied embeddings** / **PyTorch tied embeddings**: 模型参数可以通过多个模块路径共享同一对象。 / Parameters can be reachable through multiple module paths while remaining one object.
- **state-dict deduplication** / **state-dict deduplication**: 序列化系统通常要在“名字完整”和“存储不重复”之间做取舍。 / Serialization systems must balance complete names against non-duplicated storage.
- **CUDA IPC handles** / **CUDA IPC handles**: 跨进程共享显存时，句柄描述的是底层 allocation，alias 表描述的是上层语义名字。 / CUDA IPC handles describe the allocation, while aliases describe semantic names.

## 注意事项 / Caveats / when it breaks

- **对象身份和 storage 身份不完全相同** / **Object identity is not identical to storage identity**: 当前实现按 `id(tensor)` 判定；若两个不同 tensor object 共享 storage，不能自动合并。
- **遍历顺序就是 canonical 选择规则** / **Traversal order chooses the canonical name**: 改变 module 注册顺序可能改变哪一个名字成为 canonical，但不应改变最终共享关系。
- **alias 需要在加载端兑现** / **The loader must honor aliases**: 只保存 alias map 却仍然独立分配 tensor，会把优化变成隐蔽的 correctness bug。
- **buffer 生命周期也要考虑** / **Buffer lifetime matters too**: 非持久 buffer 若被其他路径依赖，缓存协议必须保证它在进程间仍然可用。

## 延伸阅读 / Further reading

- [vLLM weight-cache daemon](https://github.com/vllm-project/vllm/blob/2671fedfc7ae604761990603fc736c0c4f21de57/vllm/model_executor/model_loader/weight_cache/daemon.py)
- [PyTorch named parameter traversal](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
