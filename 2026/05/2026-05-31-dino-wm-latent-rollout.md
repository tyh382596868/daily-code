---
date: 2026-05-31
topic: diffusion
source: tracked
repo: gaoyuezhou/dino_wm
file: models/visual_world_model.py
permalink: https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py#L284-L309
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, latent-dynamics, autoregressive]
---

# DINO 世界模型的 26 行自回归 rollout / DINO-WM's 26-line autoregressive rollout

> **一句话 / In one line**: 一个 latent 世界模型只需要预测视觉 token —— 每一步把 transformer 吐出来的 action token *换成* 用户给的下一个动作,然后再接着预测。 / A latent world model only needs to predict *visual* tokens; at every step you **replace** the action token in its output with the next user-supplied action, then keep predicting.

## 为什么重要 / Why this matters

很多人第一次看 latent 世界模型,最迷的问题是:"动作到底是模型预测的、还是用户给的?" 因为训练时整个序列(observation + action)都被同一个 transformer 看到,推理时如果你也让它继续吐 action,就变成 policy 了;如果你让它老老实实只滚视觉,那 action 从哪里进来?dino_wm 用一个 26 行的 `rollout` 把答案讲得明明白白:transformer 预测的是 `(visual, proprio, action)` 这一整团 latent,但每一步推理时,我们只信它的 visual 预测,action 那一格直接被 `replace_actions_from_z` 覆盖成你想问"接下来如果我执行 a 会发生什么"的那个 a。这就是 "world model 服从规划者" 的具体接线方式。

People meeting latent world models for the first time always trip on the same question: "Does the model predict actions, or do I provide them?" During training the transformer sees the full `(observation, action)` sequence; if you let it keep generating actions at inference time, it becomes a policy. dino_wm's 26-line `rollout` is the cleanest answer I've found: the transformer predicts the **full** `(visual, proprio, action)` latent at every step, but during rollout we *throw away* its action prediction and overwrite that slot with the user-supplied next action via `replace_actions_from_z`. That's the exact wiring that makes a world model "obey" a planner.

## 代码 / The code

`gaoyuezhou/dino_wm` — [`models/visual_world_model.py`](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py#L284-L309)

```python
def rollout(self, obs_0, act):
    """
    input:  obs_0 (dict): (b, n, 3, img_size, img_size)
              act: (b, t+n, action_dim)
    output: embeddings of rollout obs
            visuals: (b, t+n+1, 3, img_size, img_size)
            z: (b, t+n+1, num_patches, emb_dim)
    """
    num_obs_init = obs_0['visual'].shape[1]
    act_0 = act[:, :num_obs_init]
    action = act[:, num_obs_init:]
    z = self.encode(obs_0, act_0)
    t = 0
    inc = 1
    while t < action.shape[1]:
        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -inc:, ...]
        z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
        z = torch.cat([z, z_new], dim=1)
        t += inc

    z_pred = self.predict(z[:, -self.num_hist :])
    z_new = z_pred[:, -1 :, ...]  # take only the next pred
    z = torch.cat([z, z_new], dim=1)
    z_obses, z_acts = self.separate_emb(z)
    return z_obses, z
```

## 逐行讲解 / What's happening

1. **拆分输入 / Split inputs (lines 292-294)**:
   - 中文: `act` 的总长度是 `t + n`,其中前 `n` 个是 *已经发生过* 的动作(配合初始观察 `obs_0` 一起编码进上下文),后 `t` 个是 *要问的未来动作*。所以才有 `act_0 = act[:, :n]` 和 `action = act[:, n:]`。
   - English: `act` has length `t + n`. The first `n` actions are the ones that *actually happened* alongside the initial observations `obs_0` (they form the context); the remaining `t` are the future actions whose consequences you want to predict. Hence `act_0 = act[:, :n]` and `action = act[:, n:]`.

2. **编码初始上下文 / Encode the initial context (line 295)**:
   - 中文: `self.encode(obs_0, act_0)` 把 `n` 帧观察 + `n` 个动作压成 latent `z`,形状 `(B, n, num_patches+2, emb_dim)`(每一帧里有视觉 patch token + 1 个 proprio token + 1 个 action token,见 `concat_dim=0` 的分支)。
   - English: `encode` collapses the `n` observations and their `n` actions into a latent stack `z` of shape `(B, n, num_patches+2, emb_dim)` — each timestep has visual patch tokens, one proprio token, and one action token (the `concat_dim=0` branch).

3. **循环预测下一步 / Loop predicting the next step (lines 298-303)**:
   - 中文: 每次循环只滚动 `inc=1` 步。`self.predict` 是一个 transformer,输入是 `z` 最后 `num_hist` 帧组成的滑动窗口(也就是模型训练时见过的固定上下文长度),输出是 *同样长度* 的预测 latent。我们只取最后 1 帧 `z_pred[:, -inc:]` 当作"下一帧"。
   - English: each iteration unrolls by `inc=1` step. `self.predict` is the transformer; it consumes the last `num_hist` frames of `z` (the same context length seen during training) and outputs a same-length predicted latent. We grab only the last frame `z_pred[:, -inc:]` as the next step.

4. **关键替换 / The key substitution (line 301)**:
   - 中文: 这一行是整个 rollout 的灵魂。`z_new` 是 transformer 自己吐的下一帧 latent(里面也有它"自以为正确"的 action token),`replace_actions_from_z(z_new, action[:, t:t+inc])` 把那一格 action token 强制覆盖成调用者给的真实下一动作。视觉/proprio 不动 —— 那是模型的预测;动作动 —— 那是你的剧本。
   - English: this line is the soul of the rollout. `z_new` is the next-frame latent the transformer just emitted (including its own guess at the action token). `replace_actions_from_z` overwrites that action slot with the action the caller actually wants to take. The visual/proprio tokens stay (those are the model's prediction); the action token is rewritten (that's your script).

5. **拼接、推进时间 / Append and advance (lines 302-303)**:
   - 中文: `torch.cat([z, z_new], dim=1)` 把新一帧粘到上下文末尾,下一轮 `predict` 自然就会看到它。`t += inc` 推进时间游标。
   - English: `cat` appends the new frame to the context; the next `predict` call automatically sees it via the sliding window. `t += inc` advances the time cursor.

6. **多滚一步 / One bonus step at the end (lines 305-307)**:
   - 中文: 循环结束后再 `predict` 一次,取最后一帧加进去 —— 因为最后一个真实 action 喂进去之后,我们还想看它执行之后的下一帧观察会是什么样。所以输出长度是 `t+n+1` 而不是 `t+n`。
   - English: after the loop, one extra `predict` produces the observation *after* the final user action has been executed. That's why the output sequence is length `t+n+1`, not `t+n`.

## 类比 / The analogy

把 `predict` 想成一个棋类的预测引擎:你告诉它"棋盘当前状态(visual+proprio)+ 我下一步走什么(action)",它告诉你"下一手棋盘会变成什么样"。但这个引擎是个话痨,它会同时告诉你它觉得 *你* 下一步会怎么走(action 预测)。`replace_actions_from_z` 就是你伸手把它嘴里的那句话堵掉、塞进你 *自己* 想下的那一步 —— 这样引擎才老老实实当世界模拟器,不会偷偷变成自动下棋的 AI。

Picture `predict` as a chess prediction engine: tell it "current board + my next move" and it tells you "the resulting board." But this engine is chatty — it also tries to predict *your* next move (the action token). `replace_actions_from_z` is you reaching in, gagging the engine, and stuffing in the move *you* want to play. Without that gag the world model would silently turn into an autonomous player; with it, it stays a faithful simulator that obeys your plan.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch

class ToyWM(torch.nn.Module):
    """Minimal stand-in for dino_wm: predict next frame as 'visual + 0.1*action'."""
    def __init__(self, num_patches=4, dim=8, action_dim=2, num_hist=2):
        super().__init__()
        self.num_hist, self.action_dim = num_hist, action_dim
        # one weight matrix; predicts ALL tokens including action
        self.proj = torch.nn.Linear(dim, dim)

    def predict(self, z):
        # z: (B, T, P+1, D)  -- last token is the action; we still predict it
        return self.proj(z)

    def replace_action(self, z_new, action):
        # rewrite the last token (the action slot) to encode the user's action
        z_new[:, :, -1, :self.action_dim] = action
        return z_new

def rollout(wm, z_init, actions):
    z = z_init.clone()
    for t in range(actions.shape[1]):
        z_pred = wm.predict(z[:, -wm.num_hist:])  # use sliding window
        z_new  = z_pred[:, -1:].clone()           # next-frame guess
        z_new  = wm.replace_action(z_new, actions[:, t:t+1])
        z = torch.cat([z, z_new], dim=1)
    return z

torch.manual_seed(0)
B, T0, P, D = 1, 2, 4, 8                       # batch, init frames, patches+action, dim
wm = ToyWM(num_patches=P-1, dim=D)
z_init  = torch.randn(B, T0, P, D)             # 2 initial frames
actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])  # 3 future actions
out = rollout(wm, z_init, actions)
print("init frames:", T0, "future actions:", actions.shape[1], "out frames:", out.shape[1])
print("action slots over time:", out[0, :, -1, :2].tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
init frames: 2 future actions: 3 out frames: 5
action slots over time: [[...random...], [...random...], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
```

中文: 注意最后三个 action 槽 *精确* 等于你传进去的 `actions` —— 模型自己的预测被完全覆盖了。前两个槽是初始上下文里的随机值,没动。

English: notice the last three action slots match the input `actions` exactly — the model's own guesses were overwritten. The first two slots are the random initial context, untouched.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Dreamer V3 (`danijar/dreamerv3`)** / **Dreamer V3**: 中文: 用 RSSM 在 latent 里 rollout,actor 在 latent 上选动作 —— 不是替换 token,而是 *额外* 用一个 actor 网络生成 action,但同样保留了"动作从外部注入"的边界。 / English: rolls out an RSSM in latent space and uses a separate actor network to pick actions instead of overwriting tokens — same outer interface ("actions injected from outside"), different mechanism.
- **CEM / MPPI 规划器** / **CEM / MPPI planners**: 中文: 5-25/5-27 教过的 CEM 和 MPPI 调的就是这种 `rollout(world_model, actions) -> future_states` 接口 —— 规划器枚举几千条动作序列,每条都让世界模型模拟出未来,然后选 reward 最高的。dino_wm 的 `rollout` 就是它们调用的那个 `f`。 / English: the CEM/MPPI planners covered on 2026-05-25 / 2026-05-27 call exactly this `rollout(model, actions) -> future_states` signature. Planners enumerate thousands of action sequences and ask the world model to simulate each.
- **GAIA-1 / Wayve LingoQA** / **GAIA-1 / Wayve LingoQA**: 中文: 自动驾驶的视频世界模型也是把"控制信号"作为额外 token 注入到生成流里,逻辑跟 dino_wm 完全同构。 / English: driving-video world models inject control signals as extra tokens into the generation stream — same wiring, different domain.

## 注意事项 / Caveats / when it breaks

- **`num_hist` 是训练 / 推理共享的滑动窗口长度** / **`num_hist` is the train-time / infer-time sliding window**: 中文: 训练时预测器看的是 `num_hist` 帧的窗口,推理时 `z[:, -self.num_hist:]` 也只喂这么多。如果你推理时给的初始帧少于 `num_hist`,预测会读到训练时没见过的短序列,质量直接崩。 / English: the transformer was trained on `num_hist`-length windows. If you start a rollout with fewer initial frames than that, you feed it a distribution it has never seen — quality collapses.
- **行内修改 `z_new`** / **In-place mutation of `z_new`**: 中文: `replace_actions_from_z` 是 in-place 改 tensor,如果 `z_pred` 后面还要复用,记得 `.clone()`(我们的 try.py 里就 clone 了)。 / English: `replace_actions_from_z` mutates the tensor in place. If you reuse `z_pred` later, `.clone()` first (our try-it example does this).
- **不会自动停止** / **No termination signal**: 中文: 这个 rollout 永远滚 `len(action)` 步,模型不会自己说"环境结束了"。如果你需要 episodic 任务,得在外面单独维护 done 标志。 / English: this rollout always unrolls for `len(action)` steps. There's no learned "episode end" — if you need termination, handle it externally.

## 延伸阅读 / Further reading

- DINO-WM paper: <https://arxiv.org/abs/2411.04983>
- Dreamer V3: <https://github.com/danijar/dreamerv3>
- Past entry on CEM-in-world-models: `2026/05/2026-05-25-nano-world-model-cem-planner.md`
