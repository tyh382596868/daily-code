---
date: 2026-08-03
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/utils/torch_utils.py
permalink: https://github.com/huggingface/diffusers/blob/cae82a76925be058790096068ea43dcc28075f02/src/diffusers/utils/torch_utils.py#L183-L234
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, random-noise]
---

# Diffusers randn_tensor：随机噪声也要管 device / Diffusers randn_tensor: Random Noise Has Device Rules Too

> **一句话 / In one line**: `randn_tensor` 统一处理 CPU/CUDA/MPS/Neuron 和 per-sample generator，保证 latent 噪声可复现、可搬运。 / `randn_tensor` centralizes CPU/CUDA/MPS/Neuron and per-sample generator handling so latent noise stays reproducible and movable.

## 为什么重要 / Why this matters

扩散模型第一步通常是一张随机 latent。这个随机数如果在错误设备上生成，轻则慢一点，重则 seed 不可复现或直接报错。Diffusers 把这件小事封装成公共工具。

A diffusion pipeline often starts from a random latent. If the tensor is generated on the wrong device, it may be slower, non-reproducible, or invalid. Diffusers wraps this small but important detail in one utility.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/utils/torch_utils.py`](https://github.com/huggingface/diffusers/blob/cae82a76925be058790096068ea43dcc28075f02/src/diffusers/utils/torch_utils.py#L183-L234)

```python
def randn_tensor(
    shape: tuple | list,
    generator: list["torch.Generator"] | "torch.Generator" | None = None,
    device: str | "torch.device" | None = None,
    dtype: "torch.dtype" | None = None,
    layout: "torch.layout" | None = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`."""
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if device.type == "neuron":
        rand_device = torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device.type not in ("mps", "neuron"):
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents
```

## 逐行讲解 / What's happening

1. **第 195-202 行 / Lines 195-202 (target vs random device)**:
   - 中文: `device` 是最终要放的位置，`rand_device` 是实际生成随机数的位置。
   - English: `device` is the final destination; `rand_device` is where random numbers are actually generated.
2. **第 207-219 行 / Lines 207-219 (generator device)**:
   - 中文: CPU generator 可以先在 CPU 采样再搬过去；CUDA generator 不能拿来生成别的设备 tensor。
   - English: A CPU generator can sample on CPU then move; a CUDA generator cannot generate tensors for a different device.
3. **第 224-233 行 / Lines 224-233 (per-sample seeds)**:
   - 中文: generator list 表示 batch 里每个样本单独 seed，于是逐样本生成再拼起来。
   - English: A generator list means one seed per batch item, so the function samples each item separately and concatenates.

## 类比 / The analogy

这像冲照片：底片可以最后放进不同相册，但冲洗设备和底片规格必须匹配。不同家庭成员还可以各用自己的底片编号。

It is like developing photos: the print can go into any album later, but the developing machine must match the film. Each family member can still use a separate film roll number.

## 自己跑一遍 / Try it yourself

```python
import random

def randn_batch(seeds, shape):
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        out.append([round(rng.gauss(0, 1), 3) for _ in range(shape[1])])
    return out

print(randn_batch([7, 8], (2, 3)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[-0.256, 0.511, -0.226], [0.373, 2.533, 1.095]]
```

中文: 每个样本用自己的 generator，就能单独控制 batch 中每张图的随机起点。  
English: Per-sample generators let each image in the batch own its random starting point.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Pipeline latent initialization** / **Pipeline latent initialization**: 文生图、图生图和视频 pipeline 都要创建初始 latent。 / Text-to-image, image-to-image, and video pipelines all need initial latents.
- **Distributed sampling** / **Distributed sampling**: 多进程推理会更依赖明确的 seed/device 语义。 / Multi-process inference depends even more on explicit seed and device semantics.

## 注意事项 / Caveats / when it breaks

- **generator list 长度** / **Generator list length**: list 应该和 batch size 对齐，否则逐样本索引会出问题。 / The list should match batch size, or per-sample indexing breaks.
- **设备搬运成本** / **Transfer cost**: CPU 采样再搬到 GPU 可复现但可能更慢。 / Sampling on CPU then moving to GPU can be reproducible but slower.

## 延伸阅读 / Further reading

- Diffusers `randn_tensor`: https://github.com/huggingface/diffusers/blob/cae82a76925be058790096068ea43dcc28075f02/src/diffusers/utils/torch_utils.py#L183-L234
