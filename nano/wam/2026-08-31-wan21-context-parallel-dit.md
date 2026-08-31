---
date: 2026-08-31
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/distributed/xdit_context_parallel.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/distributed/xdit_context_parallel.py#L93-L180
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, dit-block, context-parallel, sequence-parallel]
build_role: dit-block advanced variant
---

# Wan2.1 context parallel：把视频 token 沿序列切开 / Wan2.1 Context Parallel: Split Video Tokens Along Sequence

> **一句话 / In one line**: 这段 forward 先把每个视频 patchify 并补齐，然后按 sequence-parallel rank 切 token，block 跑完后再 all-gather 回完整序列。 / This forward path patchifies and pads each video, splits tokens by sequence-parallel rank, runs the blocks, then all-gathers the full sequence.

## 为什么重要 / Why this matters

视频 DiT 的 token 数是 `T * H * W`，长视频或高分辨率时单卡注意力很快顶不住。Wan2.1 的 context parallel 把序列维切给多张卡，让每张卡只处理一段上下文，再用 long-context attention 和 all-gather 保持语义等价。

Video DiTs have `T * H * W` tokens, so long or high-resolution clips quickly overwhelm one GPU. Wan2.1's context parallelism splits the sequence dimension across devices, while long-context attention and all-gather preserve the full-sequence meaning.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/distributed/xdit_context_parallel.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/distributed/xdit_context_parallel.py#L93-L180)

```python
def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    vace_context=None,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
):
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if self.model_type != 'vace' and y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    for block in self.blocks:
        x = block(x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=None)

    x = self.head(x, e)
    x = get_sp_group().all_gather(x, dim=1)
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]
```

## 逐行讲解 / What's happening

1. **第 120-129 行 / Lines 120-129 (patchify and pad)**:
   - 中文: 每个视频先过 patch embedding，再展平成 token 序列，最后补齐到共同 `seq_len`。
   - English: Each video is patch-embedded, flattened into tokens, then padded to a shared `seq_len`.
2. **第 132-136 行 / Lines 132-136 (time conditioning)**:
   - 中文: timestep embedding 被投成 6 份，供 DiT block 的调制路径使用。
   - English: The timestep embedding is projected into six chunks for DiT block modulation.
3. **第 159-162 行 / Lines 159-162 (sequence split)**:
   - 中文: 真正的并行点在这里：沿 token 维切块，每个 rank 只拿自己的片段。
   - English: This is the core parallel step: split along the token axis and give each rank its slice.
4. **第 169-180 行 / Lines 169-180 (blocks, gather, unpatchify)**:
   - 中文: block 和 head 在局部序列上跑，最后 all-gather 合回完整序列并还原视频 latent。
   - English: Blocks and head run on local sequence slices; all-gather rebuilds the full sequence before unpatchifying.

## 类比 / The analogy

像把一卷长胶片剪成几段交给不同剪辑师。每个人只处理手上的片段，但交片前必须按原顺序拼回去，否则电影就乱了。

It is like cutting a long film strip into sections for several editors. Each editor works on one section, but the final reel must be stitched back in order.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `dit-block` 的 advanced variant。nanoWAM 最小版可以单卡跑完整序列；生产版遇到长视频、3D latent 或 action-conditioned rollout 时，需要这种 sequence/context parallel。上游是 VAE latent 和文本/动作条件，下游是输出 head 与 unpatchify。省掉它不会改变模型数学，但会限制可训练的视频长度和分辨率。

English: This is an advanced variant of the `dit-block` curriculum item. A minimal nanoWAM can run the whole sequence on one GPU; a production WAM needs sequence/context parallelism for long clips, 3D latents, or action-conditioned rollouts. Upstream are VAE latents and text/action conditions; downstream are the output head and unpatchify. Omitting it does not change the math, but it caps sequence length and resolution.

## 自己跑一遍 / Try it yourself

```python
import torch

x = torch.arange(2 * 8 * 3).view(2, 8, 3)
world = 4
chunks = torch.chunk(x, world, dim=1)
local = [chunk + rank * 100 for rank, chunk in enumerate(chunks)]
gathered = torch.cat(local, dim=1)
print(gathered.shape)
print(gathered[0, :, 0].tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
torch.Size([2, 8, 3])
[0, 3, 106, 109, 212, 215, 318, 321]
```

中文: 每段 token 可以在本地被处理，但输出必须沿原来的序列维拼回去。

English: Each token segment can be processed locally, but outputs must be gathered back along the original sequence axis.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Ring attention / Ulysses attention**: 中文: 长上下文 LLM 也常沿 sequence 维拆分 attention。 / English: Long-context LLMs also split attention along the sequence dimension.
- **vLLM tensor/context parallel serving**: 中文: 推理服务里同样会把一次请求拆给多卡再合并。 / English: Serving systems likewise split one request across devices and then merge results.

## 注意事项 / Caveats / when it breaks

- **padding mask 不能丢 / Padding masks cannot be ignored**: 中文: 补齐 token 不该影响有效视频 token 的注意力语义。 / English: Padded tokens must not affect valid video-token attention semantics.
- **通信成本是真成本 / Communication is real cost**: 中文: `all_gather` 太频繁会抵消切序列带来的显存收益。 / English: Frequent `all_gather` can erase the memory win from sequence splitting.

## 延伸阅读 / Further reading

- Wan2.1 context parallel source: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/distributed/xdit_context_parallel.py
- xDiT project: https://github.com/xdit-project/xDiT
