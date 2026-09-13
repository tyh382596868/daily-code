---
date: 2026-09-13
topic: infrastructure
source: trending
repo: hao-ai-lab/FastVideo
file: fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py
permalink: https://github.com/hao-ai-lab/FastVideo/blob/bfc9c017977d1f428d43b6e580a0ff20503d442b/fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py#L23-L157
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, video-sparse-attention, block-sparse, metadata, cuda]
---

# FastVideo VSA 元数据：先把视频 token 排成块，再让稀疏 kernel 跳过 padding / FastVideo VSA Metadata: Tile Video Tokens Before Sparse Kernels Skip Padding

> **一句话 / In one line**: FastVideo 把 `(T, H, W)` 的视频 token 重排成 tile-contiguous 顺序，并提前计算边界 tile 的有效长度，让 VSA kernel 不必把 padding 当真实 token。 / FastVideo reorders `(T, H, W)` video tokens into tile-contiguous order and computes boundary lengths ahead of time so VSA kernels do not treat padding as real tokens.

## 为什么重要 / Why this matters

中文：稀疏 attention 的难点经常不在矩阵乘法，而在“哪些 token 属于哪个 block”这份 metadata。视频尺寸通常不能整除 tile size，边界块会比内部块短。`vsa_utils.py` 把布局重排、反向索引、variable block sizes 和 non-pad positions 一次准备好，CUDA kernel 只消费结果。

English: The hard part of sparse attention is often not the matrix multiplication but the metadata describing which tokens belong to which block. Video shapes rarely divide evenly by the tile size, so boundary blocks are shorter. `vsa_utils.py` prepares the layout permutation, inverse indices, variable block sizes, and non-padding positions before the CUDA kernel consumes them.

## 代码 / The code

`hao-ai-lab/FastVideo` — [`fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py`](https://github.com/hao-ai-lab/FastVideo/blob/bfc9c017977d1f428d43b6e580a0ff20503d442b/fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py#L23-L157)

```python
def _canonicalize_device(device: torch.device | str) -> torch.device:
    """Resolve an indexless CUDA device before it is used as a cache key."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Map raster-order token indices to tile-contiguous order."""
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(indices[
                    t * ts:min(t * ts + ts, T),
                    h * hs:min(h * hs + hs, H),
                    w * ws:min(w * ws + ws, W),
                ].flatten())
    return torch.cat(ls, dim=0)


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Inverse of get_tile_partition_indices: tile order back to raster."""
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    device: torch.device,
    tile_size: tuple[int, int, int] = VSA_TILE_SIZE,
) -> torch.LongTensor:
    """Compute the number of valid tokens in each tile."""
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n: int) -> torch.LongTensor:
        sizes = torch.full((n,), tile, dtype=torch.int, device=device)
        remainder = dim_len - (n - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)
    h_sizes = _sizes(h, ts_h, n_h)
    w_sizes = _sizes(w, ts_w, n_w)

    return (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)


def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
) -> torch.LongTensor:
    """Find positions of real tokens within a block-padded layout."""
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def build_vsa_metadata(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int] = VSA_TILE_SIZE,
    device: torch.device | str = "cuda",
) -> dict:
    """Build all VSA metadata from a video latent shape in one call."""
    device = _canonicalize_device(device)

    T, H, W = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    max_block_size = math.prod(tile_size)
    if max_block_size not in _SUPPORTED_VSA_BLOCK_VOLUMES:
        raise ValueError(
            f"Unsupported VSA tile volume {max_block_size} for tile_size={tile_size}; "
            f"supported volumes are {_SUPPORTED_VSA_BLOCK_VOLUMES}."
        )

    num_tiles = (
        math.ceil(T / ts_t),
        math.ceil(H / ts_h),
        math.ceil(W / ts_w),
    )

    tile_indices = get_tile_partition_indices(dit_seq_shape, tile_size, device)
    reverse_tile_indices = get_reverse_tile_partition_indices(dit_seq_shape, tile_size, device)
    vbs = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, tile_size)
    npi = get_non_pad_index(vbs, max_block_size)

    return {
        "tile_partition_indices": tile_indices,
        "reverse_tile_partition_indices": reverse_tile_indices,
        "variable_block_sizes": vbs,
        "non_pad_index": npi,
        "num_tiles": num_tiles,
        "max_block_size": max_block_size,
    }
```

## 逐行讲解 / What's happening

1. **第 23-28 行 / Lines 23-28 (device key)**:
   - 中文：`cuda` 没有显式 index 时，先解析成当前 device。否则同一个 GPU 可能以不同 cache key 存两份 metadata。
   - English: An indexless `cuda` device is resolved to the current device first. Otherwise the same GPU could receive multiple cache entries under different keys.
2. **第 31-54 行 / Lines 31-54 (tile partition)**:
   - 中文：先创建 raster-order 的 `[T, H, W]` 索引，再按时间、行、列 tile 遍历并 flatten。边界 tile 用 `min` 截断，所以不会读出范围。
   - English: The code creates raster-order `[T, H, W]` indices, then walks time, row, and column tiles and flattens each one. `min` clips boundary tiles safely.
3. **第 57-64 行 / Lines 57-64 (reverse permutation)**:
   - 中文：`argsort` 把 tile-order permutation 反过来，kernel 输出可以重新放回原始视频布局。
   - English: `argsort` inverts the tile-order permutation so kernel outputs can return to the original video layout.
4. **第 67-93 行 / Lines 67-93 (variable block sizes)**:
   - 中文：内部 tile 的每一维都是完整 tile；最后一块用 remainder。三个维度相乘后得到每个 3D block 的真实 token 数。
   - English: Interior tiles use the full tile length; the last tile uses the remainder. Multiplying the three dimensions yields each 3D block's real token count.
5. **第 96-110 行 / Lines 96-110 (non-pad indices)**:
   - 中文：kernel 可能要求所有 block 先 pad 到同一个 `max_block_size`。`get_non_pad_index` 给出其中真正有效 token 的 flat positions。
   - English: A kernel may require every block to be padded to `max_block_size`. `get_non_pad_index` returns the flat positions that contain real tokens.
6. **第 113-157 行 / Lines 113-157 (one metadata boundary)**:
   - 中文：入口先检查 tile volume 是否是 backend 支持的 64、128 或 256，再一次性返回所有索引和 shape metadata。
   - English: The entry point checks that tile volume is supported by the backend, then returns all permutations and shape metadata in one object.

## 类比 / The analogy

中文：像把一张不规则的城市地图切成快递分区。大多数分区正好装满 64 个包裹，边缘分区可能只有 40 个。先记录每个分区的真实大小，配送系统就不必把 24 个空位也当成包裹处理。

English: Imagine dividing an irregular city map into delivery zones. Most zones hold exactly 64 packages, while an edge zone may hold only 40. Recording the true size first lets the delivery system ignore the 24 empty slots.

## 自己跑一遍 / Try it yourself

```python
import math


def tile_order(shape, tile):
    T, H, W = shape
    tt, th, tw = tile
    order = []
    for t in range(math.ceil(T / tt)):
        for h in range(math.ceil(H / th)):
            for w in range(math.ceil(W / tw)):
                for dt in range(t * tt, min((t + 1) * tt, T)):
                    for dh in range(h * th, min((h + 1) * th, H)):
                        for dw in range(w * tw, min((w + 1) * tw, W)):
                            order.append((dt, dh, dw))
    return order


print(tile_order((1, 3, 5), (1, 2, 2)))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 0, 2), (0, 0, 3), (0, 1, 2), (0, 1, 3), (0, 0, 4), (0, 1, 4), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4)]
```

中文：输出顺序先保证每个 `2 x 2` 空间 tile 内的 token 连续，再处理右边和底部的边界 tile。真实实现把坐标换成 GPU tensor，并额外计算 block size 和反向索引。

English: The output keeps each `2 x 2` spatial tile contiguous before moving to right and bottom boundary tiles. The real implementation uses GPU tensors and also computes block sizes and the inverse permutation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FlashAttention varlen metadata** / **FlashAttention varlen metadata**: 用 cumulative lengths 描述每个样本的有效 token。 / Cumulative lengths describe each sample's valid tokens.
- **Open-Sora causal masks** / **Open-Sora causal masks**: kernel 前先把时空布局和可见性变成显式张量。 / Convert spacetime layout and visibility into explicit tensors before the kernel.
- **vLLM paged attention** / **vLLM paged attention**: block table 把逻辑序列映射到物理 KV block。 / A block table maps logical sequences to physical KV blocks.

## 注意事项 / Caveats / when it breaks

- **tile volume 是 backend contract** / **Tile volume is a backend contract**: 任意 tile size 不一定有对应 kernel，不能只看数学上是否整齐。
- **缓存 key 必须包含 device** / **Cache keys must include device**: 同一 shape 在不同 GPU 上的 index tensor 不能直接复用。
- **重排和反重排必须互逆** / **Partition and reverse partition must be inverses**: attention 结果若忘记恢复 raster order，视频会出现空间错位。
- **边界 block 不能假设满载** / **Boundary blocks are not full**: padding index 算错会把无效 token 送进 attention，表现可能是质量下降而不是立即崩溃。

## 延伸阅读 / Further reading

- [FastVideo VSA utilities](https://github.com/hao-ai-lab/FastVideo/blob/bfc9c017977d1f428d43b6e580a0ff20503d442b/fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
