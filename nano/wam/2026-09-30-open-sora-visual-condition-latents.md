---
date: 2026-09-30
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/utils/train.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/train.py#L316-L407
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning]
build_role: action-conditioning advanced variant, visual condition latent packing
---

# Open-Sora visual condition：把参考帧编码成 latent，再和 mask 拼在一起 / Open-Sora Visual Condition: Encode Reference Frames into Latents, Then Concatenate the Mask

> **一句话 / In one line**: 条件视频生成不是只给模型一张图，而是把“哪些时间格已知”和“已知内容的 latent”做成同一个条件张量。 / Conditional video generation is not just giving the model an image; it packs “which latent slots are known” and “what those slots contain” into one tensor.

## 为什么重要 / Why this matters

WAM 要处理 text-to-video、image-to-video、video-to-video、loop 等不同任务。Open-Sora 这段代码把任务差异转成同一种条件格式：`masks` 指出哪些 latent slot 被钉住，`latent` 放入单帧或多帧参考内容，最后 concat 成模型可读的条件输入。

A WAM has to support text-to-video, image-to-video, video-to-video, and loop tasks. This Open-Sora code converts those tasks into one conditioning format: `masks` says which latent slots are pinned, `latent` stores the reference content, and their concatenation becomes model input.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/utils/train.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/train.py#L316-L407)

```python
def prepare_visual_condition_causal(x: torch.Tensor, condition_config: dict, model_ae: torch.nn.Module) -> torch.Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (torch.Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (torch.nn.Module): The video encoder module.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = torch.zeros(B, 1, T, H, W).to(
        x.device, x.dtype
    )  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    if T > 1:  # video
        # certain v2v conditions not are applicable for short videos
        if T <= (32 // model_ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head", None)  # given first 33 frames
            condition_config.pop("v2v_tail", None)  # given last 33 frames
            condition_config.pop("v2v_head_easy", None)  # given first 65 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 65 frames
        if T <= (64 // model_ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head_easy", None)  # given first 65 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 65 frames

        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities

        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor

            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                masks[i, :, 0, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))

            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))

            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the last image only
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))

            elif "v2v_head" in mask_cond:  # mask the first 33 video frames
                ref_t = 33 if not "easy" in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # encode the first ref_t frame video separately
                latent[i, :, :conditioned_t, :, :] = model_ae.encode(x[i, :, :ref_t, :, :].unsqueeze(0))

            elif "v2v_tail" in mask_cond:  # mask the last 32 video frames
                ref_t = 33 if not "easy" in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # encode the first ref_t frame video separately
                latent[i, :, -conditioned_t:, :, :] = model_ae.encode(x[i, :, -ref_t:, :, :].unsqueeze(0))
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    else:  # image
        x_0 = model_ae.encode(x)  # latent video

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = torch.cat((masks, latent), dim=1)
    return x_0, cond
```

## 逐行讲解 / What's happening

1. **第 328-339 行 / Lines 328-339**:
   - 中文: 函数先根据 VAE latent size 建 `masks`、`latent` 和完整视频 latent `x_0`，三者空间时间尺寸一致。
   - English: The function first creates `masks`, `latent`, and full-video latent `x_0` with matching latent time and spatial dimensions.
1. **第 340-357 行 / Lines 340-357**:
   - 中文: 太短的视频不能用长头尾条件，代码会删掉不适用选项，再按权重随机选择一种 condition。
   - English: Short videos cannot use long head/tail conditions, so the code removes invalid options and samples a condition by weight.
1. **第 359-378 行 / Lines 359-378**:
   - 中文: I2V head、loop、tail 都是同一个模式：设置 mask，编码完整视频作 target，再单独编码参考帧填进条件 latent。
   - English: I2V head, loop, and tail share one pattern: set mask slots, encode the full video as target, then separately encode reference frames into condition latents.
1. **第 380-407 行 / Lines 380-407**:
   - 中文: V2V 头尾条件把多帧参考压进 latent；最后 `masks * latent` 清掉未激活位置，再把 mask 通道和 latent 通道拼接。
   - English: V2V head/tail conditions encode multi-frame references. Finally `masks * latent` clears inactive slots and concatenates mask channels with latent channels.

## 类比 / The analogy

像给拼图机器人一张半成品图：你不仅告诉它哪些格子已经固定，还把那些格子的图案贴上去；空格才需要它补全。

It is like giving a puzzle robot a partially filled board. You mark which cells are fixed and paste the known pieces there; only the empty cells need generation.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，这属于 `action-conditioning` 的视觉条件分支。输入是原始视频、任务条件配置和 VAE；输出是训练 target latent `x_0` 以及可拼到 DiT 输入通道上的 `cond`。没有这个组件，I2V/V2V/loop 都会退化成普通 T2V，模型不知道哪些帧必须保留。

In a nanoWAM, this is the visual branch of `action-conditioning`. It takes raw video, a task-condition config, and the VAE, then returns the training target latent `x_0` plus a `cond` tensor that can be concatenated into DiT inputs. Without it, I2V, V2V, and loop generation collapse into plain T2V because the model cannot know which frames are fixed.

## 自己跑一遍 / Try it yourself

```python
def condition(kind, T):
    mask = [0] * T
    latent = ['.'] * T
    if kind == 'i2v_head':
        mask[0], latent[0] = 1, 'first'
    elif kind == 'i2v_loop':
        mask[0], mask[-1] = 1, 1
        latent[0], latent[-1] = 'first', 'last'
    elif kind == 't2v':
        pass
    return list(zip(mask, latent))

print(condition('i2v_head', 4))
print(condition('i2v_loop', 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[(1, 'first'), (0, '.'), (0, '.'), (0, '.')]
[(1, 'first'), (0, '.'), (0, '.'), (1, 'last')]
```

这个玩具例子只保留时间维，但核心相同：mask 和内容必须对齐到同一 latent 网格。

The toy keeps only the time axis, but the core is the same: mask and content must align on the same latent grid.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 VACE** / **Wan2.1 VACE**: 也把可控区域和参考内容编码到额外 latent 通道里。 / It also encodes controllable regions and reference content into extra latent channels.
- **Inpainting diffusion** / **Inpainting diffusion**: mask 和 masked image/latent 一起输入，模型只补未固定区域。 / Mask and masked image/latent are fed together so the model fills only unfixed areas.

## 注意事项 / Caveats / when it breaks

- **不要泄漏未来** / **Avoid future leakage**: 条件帧必须单独编码并放到指定 latent slot，不能让完整视频 latent 混进条件。 / Reference frames must be encoded separately into specific slots; the full-video latent must not leak into conditioning.
- **时间压缩要整除** / **Temporal compression must line up**: `ref_t` 和 VAE `time_compression_ratio` 不匹配时，条件帧会落不到整数 latent 帧。 / If `ref_t` and the VAE compression ratio do not align, reference frames will not map to integer latent frames.

## 延伸阅读 / Further reading

- [Open-Sora repository](https://github.com/hpcaitech/Open-Sora)
- [Latent diffusion models](https://arxiv.org/abs/2112.10752)
