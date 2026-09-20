---
date: 2026-09-20
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/parallelism_config.py
permalink: https://github.com/huggingface/accelerate/blob/b795b4838eb33bde60eb398649c4ab006874e3e9/src/accelerate/parallelism_config.py#L211-L272
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, accelerate, device-mesh, distributed, parallelism]
---

# Accelerate device mesh：先按语义排序，再给 PyTorch 一张规范网格 / Accelerate Device Mesh: Sort by Meaning Before Giving PyTorch a Canonical Mesh

> **一句话 / In one line**: Accelerate 把 DP、TP、CP、SP 等并行语义整理成稳定的 mesh 维度，按需创建，并为联合 data-parallel 维度建立 flattened view。 / Accelerate turns DP, TP, CP, and SP semantics into a stable mesh, creates it lazily, and exposes flattened views for joint data-parallel dimensions.

## 为什么重要 / Why this matters

分布式训练配置往往不是一张简单的二维表：data parallel 可能拆成 replicate 和 shard，context parallel 可能和 shard 合并，tensor parallel 又需要放在确定的位置。若各组件自己拼 process group，维度顺序、空 mesh 和后端边界很容易不一致。

Distributed training configuration is rarely a simple 2D table: data parallel may split into replicate and shard, context parallel may join shard, and tensor parallel needs a deterministic position. If each component builds process groups independently, dimension order, empty meshes, and backend boundaries drift.

`build_device_mesh` 的策略是“先判断谁负责，再构造唯一 mesh”：DeepSpeed sequence parallel 已经由 DeepSpeed 管理时跳过；Torch 版本不够直接报错；没有激活维度就不创建；有维度则用 canonical order 初始化，并把联合维度 flatten 成稳定名称。

`build_device_mesh` follows a “decide ownership, then build one mesh” strategy: skip when DeepSpeed owns sequence parallel; fail clearly on old Torch; return `None` for no active dimension; otherwise initialize in canonical order and flatten joint dimensions into stable names.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/parallelism_config.py`](https://github.com/huggingface/accelerate/blob/b795b4838eb33bde60eb398649c4ab006874e3e9/src/accelerate/parallelism_config.py#L211-L272)

```python
def build_device_mesh(self, device_type: str):
    """Builds a device mesh for the given device type based on the parallelism configuration.
    This method will also create required joint meshes (e.g. `dp_shard_cp`, `dp_cp`, `dp`).
    """
    # Skip mesh creation for DeepSpeed SP - DeepSpeed handles its own SP groups
    # Only skip when SP is actually enabled (sp_size > 1), otherwise user might still want TP/CP/FSDP
    if self.sp_backend == "deepspeed" and self.sp_size > 1:
        return None

    if is_torch_version(">=", "2.2.0"):
        from torch.distributed.device_mesh import init_device_mesh
    else:
        raise RuntimeError("Building a device_mesh requires to have torch>=2.2.0")

    mesh = self._get_mesh()
    if len(mesh) == 0:
        return None
    mesh_dim_names, mesh_shape = mesh
    device_mesh = init_device_mesh(
        device_type,
        mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )
    if self.dp_dim_names:
        device_mesh[self.dp_dim_names]._flatten("dp")
    if self.dp_shard_cp_dim_names:
        device_mesh[self.dp_shard_cp_dim_names]._flatten("dp_shard_cp")
    if self.dp_cp_dim_names:
        device_mesh[self.dp_cp_dim_names]._flatten("dp_cp")

    return device_mesh

def _get_mesh(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Generate mesh shape and dimension names for torch.distributed.init_device_mesh()."""
    mesh_dims = {parallelism: self._sizes[parallelism] for parallelism in self.active_mesh_dims}
    mesh_order = ["dp_replicate", "dp_shard", "cp", "sp", "tp"]
    sorted_items = sorted(
        mesh_dims.items(),
        key=lambda x: (mesh_order.index(x[0])),
    )
    return tuple(zip(*sorted_items))
```

## 逐行讲解 / What's happening

1. **后端所有权 / Backend ownership**:
   - 中文: DeepSpeed SP 开启时直接返回，让两个系统不要同时创建同一组 sequence-parallel communicator。
   - English: When DeepSpeed SP is enabled, the method returns so two systems do not create the same sequence-parallel communicator.
2. **版本门槛 / Version gate**:
   - 中文: `init_device_mesh` 是明确依赖，旧 Torch 不走隐式 fallback，错误会在配置阶段暴露。
   - English: `init_device_mesh` is an explicit dependency; old Torch does not take an implicit fallback, so the error appears during configuration.
3. **空配置不造对象 / Do not create an empty object**:
   - 中文: 没有 active dimension 时返回 `None`，下游可以把“未启用分布式 mesh”和“有一维 mesh”区分开。
   - English: Returning `None` for no active dimension lets downstream code distinguish “no distributed mesh” from a one-dimensional mesh.
4. **canonical order / Canonical order**:
   - 中文: `_get_mesh` 使用 `dp_replicate → dp_shard → cp → sp → tp`，把配置字典的不稳定顺序变成稳定拓扑。
   - English: `_get_mesh` uses `dp_replicate → dp_shard → cp → sp → tp`, turning unstable dictionary/config order into a stable topology.
5. **联合维度 / Joint dimensions**:
   - 中文: `dp_shard_cp`、`dp_cp` 和 `dp` 是对多个物理维度的逻辑视图，flatten 后下游只需理解语义名。
   - English: `dp_shard_cp`, `dp_cp`, and `dp` are logical views over several physical dimensions; flattening lets downstream code consume semantic names.

## 类比 / The analogy

把 mesh 想成物流中心的货架地图。物理货架按固定顺序编号，但“所有可复制仓位”可以被另一个视图合并成一个配送区域；不同承运商已经负责的通道不能再重复铺一套路。

Think of the mesh as a warehouse map. Physical shelves have a fixed order, while a second view can merge every replicable slot into one delivery zone. A carrier-owned channel should not be rebuilt by another carrier.

## 自己跑一遍 / Try it yourself

```python
def canonical_mesh(active, sizes):
    order = ["dp_replicate", "dp_shard", "cp", "sp", "tp"]
    items = sorted(
        ((name, sizes[name]) for name in active),
        key=lambda pair: order.index(pair[0]),
    )
    return tuple(size for _, size in items), tuple(name for name, _ in items)

print(canonical_mesh(
    {"tp", "dp_shard", "dp_replicate"},
    {"tp": 2, "dp_shard": 4, "dp_replicate": 2},
))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
((2, 4, 2), ('dp_replicate', 'dp_shard', 'tp'))
```

中文: 输入集合无序，但输出拓扑稳定；这正是 process-group 构造需要的性质。
English: The input is an unordered set, but the output topology is stable, which is exactly what process-group construction needs.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch DeviceMesh** / **PyTorch DeviceMesh**: 中文: Accelerate 负责把用户配置翻译成 PyTorch 原生 mesh，随后 collective 和 DTensor 可以共享命名维度。 / English: Accelerate translates user configuration into a native PyTorch mesh so collectives and DTensor can share named dimensions.
- **FSDP + TP composition** / **FSDP + TP composition**: 中文: 联合维度让 shard、replicate 和 tensor parallel 可以在同一拓扑上表达。 / English: Joint dimensions express shard, replicate, and tensor parallelism on one topology.
- **DeepSpeed ownership boundaries** / **DeepSpeed ownership boundaries**: 中文: 多框架并行时，清楚声明谁创建 communicator 比“所有框架都初始化”更重要。 / English: In multi-framework parallelism, declaring who owns communicator creation matters more than having every framework initialize one.

## 注意事项 / Caveats / when it breaks

- **mesh 顺序是 API 契约** / **Mesh order is an API contract**: 中文: 改 canonical order 可能改变 rank 到坐标的映射，即使维度集合没变。 / English: Changing canonical order can change rank-to-coordinate mapping even when the dimension set is unchanged.
- **DeepSpeed skip 有条件** / **The DeepSpeed skip is conditional**: 中文: 只有 `sp_size > 1` 才跳过；否则 TP/CP/FSDP 仍然可能需要 Accelerate mesh。 / English: The skip applies only when `sp_size > 1`; TP/CP/FSDP may still need an Accelerate mesh otherwise.
- **flatten 不会重新分配设备** / **Flattening does not move devices**: 中文: `_flatten` 是逻辑视图，不能替代真实的 world-size、rank 或通信组校验。 / English: `_flatten` creates a logical view; it does not replace world-size, rank, or communicator validation.

## 延伸阅读 / Further reading

- [Accelerate parallelism configuration](https://github.com/huggingface/accelerate/blob/b795b4838eb33bde60eb398649c4ab006874e3e9/src/accelerate/parallelism_config.py#L211-L272)
- [PyTorch DeviceMesh](https://pytorch.org/docs/stable/distributed.html#devicemesh)
- [Accelerate distributed training guide](https://huggingface.co/docs/accelerate/main/en/concept_guides/parallelism)
