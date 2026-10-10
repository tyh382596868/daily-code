---
date: 2026-10-10
topic: infrastructure
source: trending
repo: chandar-lab/semantic-wm
file: src/models/world_model.py
permalink: https://github.com/chandar-lab/semantic-wm/blob/2fd7618fb5fefa3045f137387a17868688329016/src/models/world_model.py#L119-L180
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, world-model]
---
# semantic-wm rollout：动作先写进时间线，再逐步洗出未来帧 / semantic-wm Rollout: Write Actions onto the Timeline, Then Denoise Future Frames

> **一句话 / In one line**: `generate_chunk` 先扩展 action 和 latent 时间线，再用 pyramid schedule 对未来 chunk 做 DDIM 更新。 / `generate_chunk` first extends the action and latent timelines, then uses a pyramid schedule to DDIM-update the future chunk.

## 为什么重要 / Why this matters

机器人 world model 推理是在历史后面接一段动作，再把噪声 latent 逐步洗成未来视频。这个函数把时间线维护、context 截断、action conditioning 和逐步 decode 串起来。

A robotic world model appends a planned action chunk after history, then denoises latent future frames into video. This function ties together timeline state, context trimming, action conditioning, and incremental decode.

## 代码 / The code

`chandar-lab/semantic-wm` — [`src/models/world_model.py`](https://github.com/chandar-lab/semantic-wm/blob/2fd7618fb5fefa3045f137387a17868688329016/src/models/world_model.py#L119-L180)

```python
    def generate_chunk(self, action_vec):
        """See Diffusion.generate"""
        action_chunk = torch.zeros(
            (1, self.chunk_size, self.model.action_dim), device=self.device
        )
        assert self.actions.shape[1] == self.curr_frame
        self.actions = torch.cat([self.actions, action_chunk], dim=1)
        self.actions[:, self.curr_frame : self.curr_frame + self.chunk_size, :] = (
            action_vec
        )

        scheduling_matrix = self.diffusion.generate_pyramid_scheduling_matrix(
            self.chunk_size
        )
        chunk = torch.randn(
            (1, self.chunk_size, *self.xs.shape[-3:]), device=self.device
        )
        self.xs = torch.cat([self.xs, chunk], dim=1)

        # Adjust context length
        start_frame = max(0, self.curr_frame + self.chunk_size - self.model.max_frames)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for m in range(scheduling_matrix.shape[0] - 1):
                t, t_next = scheduling_matrix[m], scheduling_matrix[m + 1]
                t, t_next = map(
                    lambda x: einops.repeat(x, "t -> b t", b=1), (t, t_next)
                )
                t, t_next = map(
                    lambda x: torch.cat(
                        (torch.zeros((1, self.curr_frame), dtype=torch.long), x), dim=1
                    ),
                    (t, t_next),
                )

                self.xs[:, start_frame:] = self.diffusion.ddim_sample_step(
                    self.model,
                    self.xs[:, start_frame:],
                    self.actions[:, start_frame : self.curr_frame + self.chunk_size],
                    t[:, start_frame:],
                    t_next[:, start_frame:],
                    cfg=self.cfg,
                )

                latest_clean_idx = (t_next == 0).nonzero()[-1][1]
                if latest_clean_idx >= self.curr_frame:
                    z = self.xs[:, latest_clean_idx : latest_clean_idx + 1]
                    if self.num_views > 1:
                        # (1, 1, h, V*w, c) -> (V, 1, h, w, c) for decoding
                        z = einops.rearrange(
                            z, "b t h (v w) c -> (b v) t h w c", v=self.num_views
                        )
                    if not self._is_identity_adapter:
                        z = self.adapter.decode(z)
                    xs = self.autoencoder.decode(z)
                    if self.num_views > 1:
                        # (V, 1, H, W, C) -> (1, 1, H, V*W, C)
                        xs = einops.rearrange(
                            xs, "(b v) t h w c -> b t h (v w) c", v=self.num_views
                        )
                    yield latest_clean_idx, xs
        self.curr_frame += self.chunk_size
```

## 逐行讲解 / What's happening

1. **入口 / Entry**:
   - 中文: 先看函数签名和输入，它定义了这个模块承担的边界职责。
   - English: Start from the signature and inputs; they define the module boundary.
2. **核心状态 / Core state**:
   - 中文: 中间变量保存的是工程约束，例如 cache、rank、bin 或时间线。
   - English: The intermediate variables encode engineering constraints such as cache, rank, bins, or timeline state.
3. **返回值 / Return value**:
   - 中文: 输出不是孤立结果，而是给下游模块继续消费的契约。
   - English: The output is not an isolated result; it is the contract consumed downstream.

## 类比 / The analogy

像安排路线：先把未来几步动作写进日程表，再把每个未知地点从模糊草图逐步描清。

It is like planning a route: write future actions into the schedule, then refine each unknown stop from a blurry sketch.

## 自己跑一遍 / Try it yourself

```python
def rollout(history, actions):
    xs = list(history) + ["noise" for _ in actions]
    for step in range(3, 0, -1):
        for i, x in enumerate(xs):
            if x == "noise":
                xs[i] = f"frame_{i}@t{step-1}"
        clean = [x for x in xs if x.endswith("@t0")]
        if clean: yield clean[-1]
print(list(rollout(["f0"], ["left", "grip"])))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
['frame_2@t0']
```

这个小例子保留了源码里最重要的控制结构，方便你先写最小版再回到工程实现。

The small example keeps the most important control structure from the source, so you can write the minimal version before returning to the production implementation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **训练/推理边界** / **Training/inference boundaries**: 这类代码常把数学步骤转换成工程契约。 / This kind of code turns a mathematical step into an engineering contract.
- **nano 系统实现** / **Nano-system implementation**: 从这些片段抽象出的接口可以直接变成你自己的最小模块。 / The interface abstracted from these snippets can become your own minimal module.

## 注意事项 / Caveats / when it breaks

- **边界条件** / **Boundary cases**: 空输入、短 batch、短视频或越界动作通常最容易出 bug。 / Empty inputs, short batches, short videos, or out-of-range actions are the easiest places to break.
- **契约要写测试** / **Test the contract**: 这些函数依赖调用方遵守 shape、顺序和状态生命周期。 / These functions rely on callers respecting shapes, ordering, and state lifetime.

## 延伸阅读 / Further reading

- [chandar-lab/semantic-wm source](https://github.com/chandar-lab/semantic-wm/blob/2fd7618fb5fefa3045f137387a17868688329016/src/models/world_model.py#L119-L180)
