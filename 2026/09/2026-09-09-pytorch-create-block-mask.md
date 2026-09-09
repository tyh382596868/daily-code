---
date: 2026-09-09
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/flex_attention.py
permalink: https://github.com/pytorch/pytorch/blob/e327243b87ad143e228439048354f28116433020/torch/nn/attention/flex_attention.py#L1982-L2102
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, flex-attention, block-sparse, mask]
---

# PyTorch `create_block_mask`：把布尔规则压成稀疏块 / PyTorch `create_block_mask`: Compile a Boolean Rule into Sparse Blocks

> **一句话 / In one line**: `create_block_mask` 先生成元素级 mask，再把它压缩成 FlexAttention kernel 能跳过空块的 `BlockMask`。 / `create_block_mask` first materializes an elementwise mask, then compresses it into `BlockMask` metadata that lets FlexAttention skip empty blocks.

## 为什么重要 / Why this matters

中文：写 attention mask 时，人最容易理解的是 `q_idx >= kv_idx` 这种逐元素谓词；kernel 最喜欢的却是“第几个 query block 需要看哪些 key block”。这段代码承担了两种表示之间的转换，并额外处理完整块、编译路径和确定性反向传播所需的写入顺序。

English: Humans naturally describe attention sparsity with an elementwise predicate such as `q_idx >= kv_idx`. A kernel wants block-level metadata instead: which key blocks contribute to each query block. This function bridges those representations and optionally prepares full-block shortcuts and deterministic backward ordering.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/flex_attention.py`](https://github.com/pytorch/pytorch/blob/e327243b87ad143e228439048354f28116433020/torch/nn/attention/flex_attention.py#L1982-L2102)

```python
def create_block_mask(
    mask_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType | None = None,
    BLOCK_SIZE: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
    _compile=False,
    separate_full_blocks: bool = True,
    compute_dq_write_order: bool = False,
    dq_kv_order: bool = True,
) -> BlockMask:
    if device is None:
        device = torch.accelerator.current_accelerator() or "cpu"
    mod_type = _get_mod_type(mask_mod)
    if mod_type != _ModificationType.MASK_MOD:
        raise AssertionError(
            f"create-block_mask requires a mask_mod function! Got {mask_mod}"
        )
    if B is None:
        B = 1
    if H is None:
        H = 1
    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE = BLOCK_SIZE
        KV_BLOCK_SIZE = BLOCK_SIZE
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

    if _compile:
        warnings.warn(
            "_compile flag on create_block_mask was originally added to work around a torch.compile limitation. That limitation has since been addressed. So, to compile create_block_mask, we suggest doing torch.compile(create_block_mask). This still works for now, but will be removed in the future.",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.compile(create_block_mask)(
            mask_mod,
            B,
            H,
            Q_LEN,
            KV_LEN,
            device,
            BLOCK_SIZE,
            False,
            separate_full_blocks,
            compute_dq_write_order,
            dq_kv_order,
        )

    if not isinstance(dq_kv_order, bool):
        raise ValueError("dq_kv_order must be a bool when using create_block_mask")

    mask_tensor = create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
        mask_tensor,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=separate_full_blocks,
    )
    block_mask = _create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask),
        mask_mod,
        (Q_LEN, KV_LEN),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )

    if compute_dq_write_order:
        dq_wo, dq_wo_full = _compute_dq_write_order_from_block_mask(
            block_mask, dq_kv_order=dq_kv_order
        )
        block_mask.dq_write_order = dq_wo
        block_mask.dq_write_order_full = dq_wo_full
        block_mask.dq_kv_order = None
        block_mask.dq_kv_order_spt = dq_kv_order

    return block_mask
```

## 逐行讲解 / What's happening

1. **第 2038-2053 行 / Lines 2038-2053**:
   - 中文: 默认设备、batch/head 默认值和非对称 block size 都在入口统一解析。
   - English: Device selection, default batch/head values, and possibly asymmetric query/key block sizes are normalized at the boundary.
2. **第 2055-2073 行 / Lines 2055-2073**:
   - 中文: 旧的 `_compile` 开关仍被兼容，但现在建议直接 `torch.compile(create_block_mask)`，避免把编译策略藏在函数内部。
   - English: The legacy `_compile` flag remains supported, but the code points users toward compiling the function itself rather than hiding compilation inside the helper.
3. **第 2078-2091 行 / Lines 2078-2091**:
   - 中文: `create_mask` 先得到四维逐元素布尔 mask，再由 `_convert_mask_to_block_mask` 分成 partial/full block，最后包装成 `BlockMask`。
   - English: `create_mask` materializes the four-dimensional boolean mask; conversion identifies partial and full blocks; the sparse metadata is then wrapped as a `BlockMask`.
4. **第 2093-2101 行 / Lines 2093-2101**:
   - 中文: 可选的 `dq_write_order` 为确定性 dQ 累加准备 rank 表，让并行 backward 的写入顺序固定。
   - English: Optional `dq_write_order` metadata gives deterministic dQ accumulation a stable rank order during parallel backward.

## 类比 / The analogy

中文：像把一张座位表先按“每 16 排、每 16 列”切成方格。逐个座位的规则很直观，但剧场工作人员真正需要的是哪些方格有人、哪些方格整块空着。

English: Think of a theater seating chart divided into 16-by-16 tiles. A seat-by-seat rule is easy to write, but the staff needs to know which tiles contain any seats and which tiles are entirely empty.

## 自己跑一遍 / Try it yourself

```python
def block_summary(q_len, kv_len, block, allowed):
    rows = []
    for q0 in range(0, q_len, block):
        row = []
        for k0 in range(0, kv_len, block):
            cells = [
                allowed(q, k)
                for q in range(q0, min(q0 + block, q_len))
                for k in range(k0, min(k0 + block, kv_len))
            ]
            row.append(("full" if all(cells) else "partial") if any(cells) else "empty")
        rows.append(row)
    return rows

print(block_summary(6, 6, 2, lambda q, k: q >= k))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[['full', 'empty', 'empty'], ['partial', 'full', 'empty'], ['partial', 'partial', 'full']]
```

中文：同一条 causal 规则，在 block 视角下变成“整块跳过、整块放行、边界块细查”三种情况。

English: The same causal rule becomes three kernel-friendly cases at block granularity: skip an empty block, accept a full block, or inspect only a boundary block.

## 注意事项 / Caveats / when it breaks

- **元素级 mask 仍可能很贵** / **Materializing the elementwise mask can still be expensive**: `create_block_mask` 的入口先调用 `create_mask`，超长序列应考虑缓存或直接构造块元数据。
- **block size 会改变稀疏收益** / **Block size changes the sparsity payoff**: block 太大时 partial block 变多，太小时元数据和 kernel 调度成本会上升。
- **确定性 backward 不是默认免费得到的** / **Deterministic backward is not free by default**: 只有设置 `compute_dq_write_order=True` 才会计算额外的 dQ 写入顺序。

## 延伸阅读 / Further reading

- [FlexAttention documentation](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [PyTorch flex_attention.py](https://github.com/pytorch/pytorch/blob/e327243b87ad143e228439048354f28116433020/torch/nn/attention/flex_attention.py)
