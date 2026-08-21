---
date: 2026-08-21
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models_pytorch/pi0_pytorch.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py#L377-L420
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, openpi, inference-loop, flow-matching]
build_role: inference-loop advanced variant
---

# openpi PyTorch sample_actions：前缀只算一次，动作一路积分 / openpi PyTorch sample_actions: Cache the Prefix, Integrate the Actions

> **一句话 / In one line**: `sample_actions` 先把图像和语言前缀编码成 KV cache，再从噪声动作出发，用 Euler step 反复调用 action expert 得到最终动作 chunk。 / `sample_actions` encodes image and language prefix tokens into a KV cache, then starts from noisy actions and repeatedly applies Euler steps through the action expert.

## 为什么重要 / Why this matters

VLA 推理里，图像和语言条件通常在一个 action chunk 内不变。每个去噪步都重新跑视觉语言前缀会浪费大量算力。openpi 的 PyTorch 路径先缓存 prefix KV，只在每个时间步更新动作后缀，让推理循环更接近生产系统需要的形态。

In VLA inference, image and language conditioning usually stays fixed within an action chunk. Recomputing that prefix at every denoising step wastes compute. openpi's PyTorch path caches prefix KV once and updates only the action suffix at each step.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models_pytorch/pi0_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py#L377-L420)

```python
def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
    """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
    bsize = observation.state.shape[0]
    if noise is None:
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = self.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # Compute image and language key value cache
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    dt = -1.0 / num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        time += dt
    return x_t
```

## 逐行讲解 / What's happening

1. **第 379-384 行 / Lines 379-384 (noise and preprocessing)**:
   - 中文: 如果外部没有给 noise，就按 batch、action horizon、action dim 生成一块动作噪声。
   - English: If no noise is provided, the method creates a noisy action tensor shaped by batch, horizon, and action dimension.
2. **第 386-400 行 / Lines 386-400 (prefix cache)**:
   - 中文: 图像和语言被 embed 成 prefix，并通过一次 forward 得到 `past_key_values`。
   - English: Image and language are embedded as a prefix, then one forward pass produces `past_key_values`.
3. **第 402-407 行 / Lines 402-407 (reverse-time setup)**:
   - 中文: `dt` 是负数，`time` 从 1.0 往 0 走，表示从噪声端积分到数据端。
   - English: `dt` is negative and `time` moves from 1.0 toward 0, integrating from noise to data.
4. **第 408-420 行 / Lines 408-420 (Euler denoise loop)**:
   - 中文: 每步用 action expert 预测速度 `v_t`，再做 `x_t = x_t + dt * v_t`。
   - English: Each step predicts velocity `v_t` and applies the Euler update `x_t = x_t + dt * v_t`.

## 类比 / The analogy

像导航前先把地图和目的地装进缓存。之后每一小步只根据当前位置修正方向，不需要重新读取整张地图。

It is like loading the map and destination once before navigation. Each later step updates direction from the current position without rereading the whole map.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoVLA 里，这是 `inference-loop` 的高级版本。上游是 observation preprocessing、视觉/语言 prefix embedding 和 action noise；下游是机器人执行的 action chunk。最小实现可以先不做 KV cache，直接每步全量 forward；生产实现应保留这里的 prefix cache、mask/position ids、固定步数和可注入 noise，方便 deterministic eval。

In a nanoVLA, this is an advanced `inference-loop` component. Upstream is observation preprocessing, vision/language prefix embedding, and action noise; downstream is the action chunk sent to the robot. A minimal version can recompute the full model every step; a production version should keep prefix cache, masks, position ids, fixed step count, and injectable noise for deterministic evaluation.

## 自己跑一遍 / Try it yourself

```python
prefix_cache = {"scene": "cup left of plate"}
x = 1.0
num_steps = 4
dt = -1.0 / num_steps

for step in range(num_steps):
    velocity = x - 0.2
    x = x + dt * velocity
    print(step, round(x, 4), prefix_cache["scene"])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 0.8 cup left of plate
1 0.65 cup left of plate
2 0.5375 cup left of plate
3 0.4531 cup left of plate
```

中文: prefix cache 不变，动作变量 `x` 每步向目标端积分。

English: The prefix cache stays fixed while the action variable `x` is integrated step by step.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LLM decoding KV cache** / **LLM decoding KV cache**: prompt KV 只算一次，新 token 逐步追加。
- **Diffusion policy samplers** / **Diffusion policy samplers**: 条件观测固定，动作样本在每个 denoising step 更新。

## 注意事项 / Caveats / when it breaks

- **prefix 变化时必须重算** / **Recompute when the prefix changes**: 新图像、新语言或新 mask 都会让旧 cache 失效。
- **步数改变会改行为** / **Changing step count changes behavior**: Euler 不是无关步数的黑盒，`num_steps` 会影响动作质量和延迟。

## 延伸阅读 / Further reading

- [openpi PyTorch `sample_actions`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py#L377-L420)
- [openpi repository](https://github.com/Physical-Intelligence/openpi)
