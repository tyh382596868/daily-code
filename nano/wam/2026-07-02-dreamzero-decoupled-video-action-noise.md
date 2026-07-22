---
date: 2026-07-02
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py#L674-L766
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, flow-matching, action-conditioning]
build_role: decoupled video/action noise schedule for a WAM action head
---

# DreamZero 解耦噪声：视频和动作不必同一个 timestep / DreamZero Decoupled Noise: Video and Action Need Not Share a Timestep

> **一句话 / In one line**: 训练 WAM 时，视频 latent 可以偏向高噪声采样，动作则独立均匀采样，从而贴近“视频还乱、动作要准”的推理场景。 / In WAM training, video latents can sample high-noise timesteps while actions sample independent uniform timesteps, matching inference where video may stay noisy but actions must become precise.

## 为什么重要 / Why this matters

世界动作模型同时预测未来画面和未来控制。画面生成可以容忍还原得慢一些，但机器人动作必须收敛到可执行命令。DreamZero 这里把 video timestep 和 action timestep 拆开：同一个 DiT 输入里，两种目标可以拥有不同噪声日程。

A World Action Model predicts both future video and future control. Video generation can tolerate gradual refinement, but robot actions must converge to executable commands. DreamZero splits video and action timesteps: within one DiT input, the two targets can follow different noise schedules.

## 代码 / The code

`dreamzero0/dreamzero` — [`wan_flow_matching_action_tf.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py#L674-L766)

```python
        if self.config.decouple_video_action_noise:
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
            noise_mode = "STANDARD"

        timestep_id_block = timestep_id[:, 1:].reshape(timestep_id.shape[0], -1, self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block)

            if self.config.decouple_video_action_noise:
                timestep_action_id = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (actions.shape[0], actions.shape[1])
                )
                action_mode = "INDEPENDENT"
            else:
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
                action_mode = "COUPLED"

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)

        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)
```

## 逐行讲解 / What's happening

1. **第 674-687 行 / Lines 674-687**: 中文: video timestep 有三种模式：解耦 Beta、高噪声 Beta、标准 uniform。 / English: Video timesteps have three modes: decoupled Beta, high-noise Beta, and standard uniform.
2. **第 690-691 行 / Lines 690-691**: 中文: 多帧 block 内强制共享同一 video timestep，保证局部时间块噪声一致。 / English: Frames inside a block share a video timestep, keeping local temporal blocks noise-consistent.
3. **第 697-711 行 / Lines 697-711**: 中文: 解耦模式下动作 timestep 独立均匀采样；非解耦模式则从 video block 重复得到。 / English: In decoupled mode, action timesteps are sampled independently; otherwise they repeat the video block timesteps.
4. **第 720-766 行 / Lines 720-766**: 中文: video 和 action 分别 add noise、分别构造 flow-matching target。 / English: Video and action separately add noise and build their own flow-matching targets.

## 类比 / The analogy

像电影拍摄和机械臂执行共用一个导演：画面可以先拍粗剪版再精修，但机械臂的每一步走位必须马上准确。两个任务在同一个片场，却不该共享同一张时间表。

It is like filming a movie while controlling a robot arm. The video can start as a rough cut and refine later, but each robot motion cue must be accurate immediately. They share one set, but not one schedule.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文: 这是 `action-conditioning` 与 `training-loop` 之间的噪声调度层。输入是 clean video latents、clean actions 和 scheduler，输出是 noisy latents/actions 以及两个 training targets。省掉它时，你的 nanoWAM 会被迫假设“视频和动作一样难去噪”，这通常不符合机器人推理。

English: This is the noise scheduling layer between `action-conditioning` and the `training-loop`. Clean video latents, clean actions, and the scheduler go in; noisy latents/actions and two training targets come out. Without it, a nanoWAM assumes video and actions are equally hard to denoise, which often mismatches robot inference.

## 自己跑一遍 / Try it yourself

```python
import torch
T = 1000
B, frames, actions = 2, 4, 6
video_ratio = torch.distributions.Beta(3.0, 1.0).sample((B, frames))
video_t = ((1 - video_ratio) * T).long().clamp(0, T - 1)
action_t = torch.randint(0, T, (B, actions))
print(video_t.shape, action_t.shape)
print("video mean", video_t.float().mean().round().item())
print("action mean", action_t.float().mean().round().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
torch.Size([2, 4]) torch.Size([2, 6])
video mean <usually lower than 500>
action mean <around 500>
```

中文: Beta(3,1) 让 `video_ratio` 偏大，所以 `1-ratio` 对应的 timestep 偏小；动作仍覆盖全范围。

English: Beta(3,1) makes `video_ratio` large, so `1-ratio` yields smaller timesteps; actions still cover the full range.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GR00T flow action heads / GR00T flow action heads**: 中文: 动作流匹配通常关心 action horizon 的独立噪声。 / English: Flow action heads often care about independent noise over the action horizon.
- **Video diffusion schedulers / Video diffusion schedulers**: 中文: 视频帧 block 内共享 timestep 可以减少时间闪烁。 / English: Sharing timesteps within frame blocks can reduce temporal flicker.

## 注意事项 / Caveats / when it breaks

- **shape 约束 / Shape contracts**: 中文: action 数量必须能按 video block 对齐，否则 assert 会失败。 / English: The action count must align with video blocks, or the assertions fail.
- **训练推理一致性 / Train-inference alignment**: 中文: 解耦训练要配套解耦推理，否则学到的噪声分布会错位。 / English: Decoupled training should pair with decoupled inference, or the learned noise distribution misaligns.

## 延伸阅读 / Further reading

- Source permalink above.
- DreamZero `FlowMatchScheduler` in the same repository.
