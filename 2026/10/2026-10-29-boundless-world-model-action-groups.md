---
date: 2026-10-29
topic: robotics
source: trending
repo: boundless-large-model/boundless-world-model
file: wan_video_action/models/wan_video_action_encoder.py
permalink: https://github.com/boundless-large-model/boundless-world-model/blob/cc0bc74ec24867975f1dd26939f3864343368367/wan_video_action/models/wan_video_action_encoder.py#L6-L51
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, action-conditioning]
---

# Boundless World Model 动作分组 / Boundless World Model Action Groups

> **一句话 / In one line**: 把四拍动作拼成一组，再投影成视频模型的调制向量。 / Pack four action steps and project them into a video-model modulation vector.

## 为什么重要 / Why this matters

动作条件视频不能只“看到”机器人动作；它还要把控制频率和视频 latent 的时间压缩对齐。

Action-conditioned video must do more than see robot actions: it must align control rate with the video latent’s temporal compression.

## 代码 / The code

`boundless-large-model/boundless-world-model` — [`wan_video_action_encoder.py`](https://github.com/boundless-large-model/boundless-world-model/blob/cc0bc74ec24867975f1dd26939f3864343368367/wan_video_action/models/wan_video_action_encoder.py#L42-L51)

```python
action_context_emb = self.action_mlp1(action)
grouped_action = torch.cat([action[:, 0:1].repeat(1, 3, 1), action], dim=1)
grouped_action = grouped_action.reshape(action.shape[0], (action.shape[1] + 3) // 4, action.shape[2] * 4)
action_mod_emb = self.action_mlp2(grouped_action)
return action_context_emb, action_mod_emb
```

## 逐行讲解 / What's happening

1. 中文：第一条 MLP 保留逐帧 action context。 English: The first MLP preserves per-frame action context.
2. 中文：首帧复制三次，使长度能整除四且开头有明确边界行为。 English: The first frame is repeated three times so length divides by four with an explicit boundary behavior.
3. 中文：四帧一组匹配压缩时间轴，第二条 MLP 产生调制 token。 English: Four-frame groups match the compressed timeline, and the second MLP produces modulation tokens.

## 类比 / The analogy

像把每四次踏板踩踏压成一小节乐谱，再交给鼓手控制整段节奏。

It is like compressing every four pedal strokes into one bar of sheet music for the drummer.

## 自己跑一遍 / Try it yourself

```python
actions = [1, 2, 3, 4, 5]
padded = [actions[0]] * 3 + actions
groups = [padded[i:i+4] for i in range(0, len(padded), 4)]
print(groups)
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `[[1, 1, 1, 1], [2, 3, 4, 5]]`

中文：前补不是偶然；它定义了第一组的因果边界。 English: The leading padding is intentional; it defines the first group’s causal boundary.

## 注意事项 / Caveats / when it breaks

- 中文：四帧是该模型的时间约定，改 VAE 压缩率时必须同步更新。 / English: Four frames is this model’s temporal contract and must change with the VAE compression ratio.

## 延伸阅读 / Further reading

- [Boundless World Model repository](https://github.com/boundless-large-model/boundless-world-model)
