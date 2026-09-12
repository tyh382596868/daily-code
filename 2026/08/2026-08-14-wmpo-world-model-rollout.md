---
date: 2026-08-14
topic: diffusion
source: trending
repo: WM-PO/WMPO
file: verl/workers/rollout/robwm_rollout.py
permalink: https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/workers/rollout/robwm_rollout.py#L338-L413
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model-rollout]
---

# WMPO rollout：VLA 动作驱动世界模型想象未来 / WMPO Rollout: VLA Actions Drive a World Model's Imagination

> **一句话 / In one line**: `run_wm_inference` 让 VLA 根据当前帧生成动作，再把动作作为条件交给 world model 采样下一段 latent video。 / `run_wm_inference` asks the VLA for actions from the current frame, then conditions a world model on those actions to sample the next latent-video chunk.

## 为什么重要 / Why this matters

WMPO 的核心卖点是不用真实机器人反复交互，也能做 on-policy 风格的策略优化。这段 rollout 代码展示了闭环：当前图像进 VLA，动作进世界模型，世界模型生成未来帧，再把最后一帧喂回 VLA。

WMPO's core idea is to do on-policy-style policy optimization without repeatedly interacting with the real robot. This rollout code shows the loop: current image into the VLA, actions into the world model, future frames out, and the last generated frame fed back into the VLA.

## 代码 / The code

`WM-PO/WMPO` — [`verl/workers/rollout/robwm_rollout.py`](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/workers/rollout/robwm_rollout.py#L338-L413)

```python
@torch.no_grad()
def run_wm_inference(self, image_paths, max_steps, repeat=1):
    """
    使用小批量（mini-batch）运行视频生成推理，并返回生成的视频数据。

    Returns:
        list: 一个字典列表。每个字典包含两个键:
              'name' (str): 视频的名称。
              'video' (np.ndarray): 视频的Numpy数组，形状为 (T, H, W, C)。
    """
    self.world_model.eval()
    vla_history = []
    latent_chunk = self.config.action_chunks_len
    init_frames_tensors, init_frames_numpys, task_descriptions, video_names = self._prepare_data(image_paths, repeat)

    current_batch_size = init_frames_tensors.shape[0]
    init_frames_for_vae = init_frames_tensors.unsqueeze(2)
    with torch.no_grad():
        latents = self.vae.encode(init_frames_for_vae)
    image_history_tensor = latents.repeat(1, 1, self.queue_len, 1, 1)
    predicted_videos = [[np.expand_dims(frame, axis=0)] for frame in init_frames_numpys]
    current_frames_np = init_frames_numpys
    frame_num = 1
    while frame_num <= max_steps:
        current_inputs = [{'full_image': resize_image(frame, (224, 224))} for frame in current_frames_np]

        vla_input = self.process_input(current_inputs, task_descriptions)
        vla_output = self._generate_one_step(vla_input)
        actions = vla_output["action"]
        step_data = {
            "responses": vla_output["responses"],
            "input_ids": vla_output["input_ids"],
            "attention_mask": vla_output["attention_mask"],
            "pixel_values": vla_output["pixel_values"],
            "action": actions,
            "step": frame_num-1
        }
        vla_history.append(step_data)

        actions = torch.from_numpy(vla_output['normalized_actions'])
        y = actions.to(self.device).to(self.dtype).reshape(current_batch_size, latent_chunk, -1)

        latent_size = self.latent_size

        z = torch.randn(current_batch_size, self.vae.out_channels, latent_chunk, *latent_size[1:], device=self.device, dtype=self.dtype)

        z_combined = torch.concat([image_history_tensor, z], dim=2)

        masks = torch.zeros(current_batch_size, image_history_tensor.shape[2] + latent_chunk, device=self.device, dtype=torch.long)
        masks[:, -latent_chunk:] = 1
        samples = self.scheduler.sample(self.world_model, z=z_combined, y=y, device=self.device, additional_args=self.model_args, progress=False, mask=masks)

        pred_latents = samples[:, :, -latent_chunk:].to(self.dtype)

        image_history_tensor = pred_latents.clone()[:, :, -self.queue_len:]

        decoded_images = self.vae.decode(pred_latents)

        pred_imgs_np = ((decoded_images.to(torch.float32).cpu().permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

        new_current_frames = []
        for i in range(current_batch_size):
            predicted_videos[i].append(pred_imgs_np[i])
            new_current_frames.append(pred_imgs_np[i, -1])
        current_frames_np = new_current_frames
        frame_num += latent_chunk
        print(f"Batch processing frame_num: {frame_num}")
```

## 逐行讲解 / What's happening

1. **第 348-358 行 / Lines 348-358**: 中文: 初始帧先过 VAE 变 latent，并复制成 `queue_len` 长度的历史窗口。 / English: The initial frame is encoded by the VAE and repeated into a `queue_len` latent-history window.
2. **第 361-376 行 / Lines 361-376**: 中文: 每轮把当前帧处理成 VLA 输入，生成动作，同时把 token、mask、像素和动作都记录下来。 / English: Each loop turns current frames into VLA inputs, generates actions, and records tokens, masks, pixels, and actions.
3. **第 378-390 行 / Lines 378-390**: 中文: normalized actions reshape 成条件 `y`；历史 latent 和新噪声拼接，mask 标出需要生成的未来 chunk。 / English: Normalized actions become condition `y`; history latents concatenate with fresh noise, and a mask marks the future chunk to generate.
4. **第 392-413 行 / Lines 392-413**: 中文: 只取采样结果的未来 chunk，decode 成图像，并把最后一帧作为下一轮 VLA 的观测。 / English: Only the future chunk is decoded, and its last frame becomes the next observation for the VLA.

## 类比 / The analogy

像飞行模拟器训练飞行员：飞行员根据当前窗外画面打方向，模拟器根据操作生成下一段窗外画面，再继续让飞行员操作。

It is like training a pilot in a simulator: the pilot acts from the current view, the simulator generates the next view from that action, and the pilot keeps acting inside the imagined loop.

## 自己跑一遍 / Try it yourself

```python
frame = "F0"
history = [frame] * 2
video = [frame]
for step in range(3):
    action = f"a{step}_from_{frame}"
    future = [f"wm({history[-1]}+{action})_0", f"wm({history[-1]}+{action})_1"]
    history = future[-2:]
    frame = future[-1]
    video.extend(future)
print(video)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['F0', 'wm(F0+a0_from_F0)_0', 'wm(F0+a0_from_F0)_1', 'wm(wm(F0+a0_from_F0)_1+a1_from_wm(F0+a0_from_F0)_1)_0', 'wm(wm(F0+a0_from_F0)_1+a1_from_wm(F0+a0_from_F0)_1)_1', 'wm(wm(wm(F0+a0_from_F0)_1+a1_from_wm(F0+a0_from_F0)_1)_1+a2_from_wm(wm(F0+a0_from_F0)_1+a1_from_wm(F0+a0_from_F0)_1)_1)_0', 'wm(wm(wm(F0+a0_from_F0)_1+a1_from_wm(F0+a0_from_F0)_1)_1+a2_from_wm(wm(F0+a0_from_F0)_1+a1_from_wm(F0+a0_from_F0)_1)_1)_1']
```

这个玩具例子故意把字符串越滚越长，直观显示“策略动作 -> 世界模型未来帧 -> 新观测”的闭环。

The toy strings grow on purpose, making the loop visible: policy action -> world-model future frame -> new observation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Ctrl-World / VLAW** / **Ctrl-World / VLAW**: 机器人动作作为视频生成条件，用想象轨迹补充真实交互。 / Robot actions condition video generation to supplement real interaction.
- **DreamZero / FastWAM** / **DreamZero / FastWAM**: 未来视频或 action-conditioned latent rollout 用来给策略提供训练信号。 / Future-video or action-conditioned latent rollout provides training signal to the policy.

## 注意事项 / Caveats / when it breaks

- **误差会闭环放大** / **Errors compound in the loop**: 生成的最后一帧会喂回 VLA，世界模型偏差可能越滚越大。 / The generated last frame feeds back into the VLA, so world-model bias can compound.
- **action 和 latent chunk 要对齐** / **Actions and latent chunks must align**: `action_chunks_len`、mask 和 VAE temporal stride 不一致会让条件错位。 / `action_chunks_len`, masks, and VAE temporal stride must agree or conditioning shifts out of place.

## 延伸阅读 / Further reading

- [WM-PO/WMPO source](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/workers/rollout/robwm_rollout.py#L338-L413)
