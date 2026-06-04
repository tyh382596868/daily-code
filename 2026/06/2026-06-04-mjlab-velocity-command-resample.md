---
date: 2026-06-04
topic: robotics
source: trending
repo: mujocolab/mjlab
file: src/mjlab/tasks/velocity/mdp/velocity_command.py
permalink: https://github.com/mujocolab/mjlab/blob/8574f9cd796bd96aa332c98799a4233deca873ef/src/mjlab/tasks/velocity/mdp/velocity_command.py#L75-L138
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, robotics, mjlab, parallel-rl, command-randomization, humanoid]
---

# mjlab 用四个 mask 同时训四种行走模式 / mjlab uses four boolean masks to train four locomotion modes at once

> **一句话 / In one line**: `UniformVelocityCommand._resample_command` 给每个 env 采一个 (vx, vy, wz),然后用四个 Bernoulli mask 把同一批 env 分流成"heading / standing / world-frame / forward-only"四种模式,一次 reset 同时训四种 policy 行为。 / `UniformVelocityCommand._resample_command` samples (vx, vy, wz) for each env, then uses four Bernoulli masks to multiplex the same env batch into "heading / standing / world-frame / forward-only" command modes — training four behaviors per reset, in one shot.

## 为什么重要 / Why this matters

mjlab 是 2026 年 6 月最热的 robotics 框架之一(2.4k+ stars,几周内崛起):它把 Isaac Lab 那套 manager-based API 直接架在 MuJoCo Warp(GPU 加速的 MuJoCo)之上,意味着你可以用 IsaacLab 习惯的方式跑 4096 并行环境,但底层是开源 MuJoCo + Warp,不依赖 Isaac Sim。这段 `_resample_command` 展示了**怎么用 GPU-native 的 boolean mask 同时训多种行为**——核心 idea 不限于 mjlab,所有大规模并行 robot RL 都能用:你不是在 4096 个 env 里跑同一个 task,而是在每个 reset 里给每个 env 抽一个"角色"(heading-tracking、standing、world-frame、forward-only),用 mask 把策略训到通用。这就是为什么 Unitree G1 训出来既能站桩、又能任意方向走、又能盯航向 —— 不是切换 task,而是 mask 多任务。

mjlab is one of June 2026's hottest robotics frameworks (2.4k+ stars in a few weeks): it sits Isaac Lab's manager-based API directly on top of MuJoCo Warp (GPU-accelerated MuJoCo), giving you Isaac Lab-style 4096 parallel envs without the Isaac Sim dependency. This `_resample_command` snippet demonstrates **how to train multiple behaviors simultaneously using GPU-native boolean masks** — a pattern that applies to any massively-parallel robot RL stack, not just mjlab. Instead of running 4096 envs of the same task, every reset assigns each env a *role* (heading-tracking, standing, world-frame, forward-only) and the masks multiplex training into a single policy. That's why a single Unitree G1 controller from this codebase can stand still, walk in arbitrary directions, *and* hold a heading — not by switching task, but by training all of them at once under different masks.

## 代码 / The code

`mujocolab/mjlab` — [`src/mjlab/tasks/velocity/mdp/velocity_command.py`](https://github.com/mujocolab/mjlab/blob/8574f9cd796bd96aa332c98799a4233deca873ef/src/mjlab/tasks/velocity/mdp/velocity_command.py#L75-L138)

```python
def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
        assert self.cfg.ranges.heading is not None
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
        self.vel_command_b[fwd_ids, 0] = (
            self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
        )
        self.vel_command_b[fwd_ids, 1] = 0.0
        self.vel_command_b[fwd_ids, 2] = 0.0

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
        root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
        root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
        lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
        lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
        root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
        root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
        root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
        root_state = torch.cat(
            [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
        )
        self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

def _update_command(self) -> None:
    if self.cfg.heading_command:
        self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[env_ids, 2] = torch.clip(
            self.cfg.heading_control_stiffness * self.heading_error[env_ids],
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
        w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
        heading = self.robot.data.heading_w[w_ids]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_w = self.vel_command_w[w_ids, 0]
        vy_w = self.vel_command_w[w_ids, 1]
        self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
        self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0
```

## 逐行讲解 / What's happening

1. **三行均匀采样**:
   - 中文: 第一段三行 `r.uniform_(*self.cfg.ranges.lin_vel_x)` 给每个 env 抽一个 `(vx, vy, wz)` 速度命令,范围由 config 给定(常见 `vx ∈ [-1, 1.5] m/s`、`wz ∈ [-1, 1] rad/s`)。`r` 是一次性复用的临时 buffer,避免每次重新分配。
   - English: The first three lines `r.uniform_(*self.cfg.ranges.lin_vel_x)` draw a `(vx, vy, wz)` velocity command per env from config-specified ranges (typically `vx ∈ [-1, 1.5] m/s`, `wz ∈ [-1, 1] rad/s`). `r` is a reused scratch buffer to avoid per-call allocation.
2. **`is_heading_env`、`is_standing_env`、`is_world_env`、`is_forward_env` —— 四个 Bernoulli 掩码**:
   - 中文: 这是整段代码的灵魂。每个 env 在 reset 时各自抽一次 [0,1] 均匀,小于阈值就被"标记"为某种模式。同一个 env 可以同时被多个模式标记(比如又是 heading 又是 forward),后续 `_update_command` 按优先级处理覆盖。
   - English: The crux of the design. Each env independently draws a uniform [0, 1] at reset; below threshold means "this env is in mode X". A single env can carry multiple flags (e.g. heading AND forward); `_update_command` resolves overlap by priority later.
3. **`fwd_ids = env_ids[self.is_forward_env[env_ids]]`**:
   - 中文: 取出那些被标记为 "forward-only" 的 env,然后强制把它们的 `vx ≥ 0.3, vy = 0, wz = 0` —— 模拟人类"直行模式"。这是 high-level domain knowledge 直接嵌进采样里。
   - English: Pull out the envs flagged as forward-only and clamp `vx ≥ 0.3, vy = 0, wz = 0` — simulating human "walking straight ahead" mode. Domain knowledge encoded directly into the sampler.
4. **`init_vel_mask` —— 初始速度注入**:
   - 中文: 一小部分 env 在 reset 时直接把当前线速度/角速度写成 sample 的命令值。意思是"机器人启动时就已经在以这个速度走"。这种 trick 让 policy 也学到稳态行走,不全靠"从站立开始加速"。
   - English: A small fraction of envs have their *current* linear/angular velocity written to match the sampled command. Meaning "the robot starts off already moving at the commanded speed." This trick teaches the policy steady-state walking, not just acceleration from a stand.
5. **`_update_command()` 的航向闭环**:
   - 中文: 对 heading envs,把 `vel_command_b[:, 2]`(yaw 速度)从"采样值"覆盖成"航向误差 × 刚度",形成一个简单 P 控制器。意思是 policy 不需要自己"想清楚为了对准航向应该转多快"——上层已经把误差转成了速度命令。
   - English: For heading envs, overwrite `vel_command_b[:, 2]` (yaw rate) with "heading error × stiffness" — a simple P controller. The policy is freed from reasoning "what yaw rate do I need to face that direction"; the higher-level command term has already converted the error into a velocity command.
6. **`_update_command()` 的 world→body 旋转**:
   - 中文: 对 world-frame envs,每帧把世界系的速度命令旋转到机器人当前 body frame,这样不管 robot 朝哪个方向走,policy 看到的"前进"始终是 body x 方向。
   - English: For world-frame envs, rotate the world-frame command into the robot's current body frame every tick. No matter which way the robot is facing, "forward" in the observation is always body-x.

## 类比 / The analogy

想象你在驾校,有 4096 辆教练车同时上路,每辆车的教练给学员发指令。普通做法是每辆车都练同一项目(比如"沿着直线开"),浪费——大家学的都一样。mjlab 的做法是每辆车在出发前抽签:有的练"按航向走"(heading)、有的练"原地停车"(standing)、有的练"按世界坐标系开"(world)、有的练"只能往前"(forward)。每辆车的教练用相同的口令格式(`vel_command_b`),但内容根据签条变化。一节课下来,policy 就同时学会了所有 4 种驾驶模式。

Picture a driving school with 4096 cars on the road, each instructor giving the student a command. The naïve approach has every car do the same drill (e.g. "drive in a straight line") — wasteful, everyone learns the same thing. mjlab instead has each car draw a card at the start: some practice "follow a heading," some "stop and stand still," some "drive in world coordinates," some "forward-only." Every instructor uses the same command format (`vel_command_b`), but the contents come from the card. After one lesson, the policy has been trained on all four driving modes in parallel.

## 自己跑一遍 / Try it yourself

```python
import torch

class MaskedCommandSampler:
    def __init__(self, num_envs, p_heading=0.2, p_standing=0.1, p_forward=0.15, device="cpu"):
        self.N = num_envs
        self.device = device
        self.vel_cmd = torch.zeros(num_envs, 3, device=device)
        self.is_heading = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.is_standing = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.is_forward = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.p = (p_heading, p_standing, p_forward)

    def resample(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_cmd[env_ids, 0] = r.uniform_(-1.0, 1.5)
        self.vel_cmd[env_ids, 1] = r.uniform_(-1.0, 1.0)
        self.vel_cmd[env_ids, 2] = r.uniform_(-1.0, 1.0)
        ph, ps, pf = self.p
        self.is_heading[env_ids]  = r.uniform_(0., 1.) <= ph
        self.is_standing[env_ids] = r.uniform_(0., 1.) <= ps
        self.is_forward[env_ids]  = r.uniform_(0., 1.) <= pf
        fwd = env_ids[self.is_forward[env_ids]]
        self.vel_cmd[fwd, 0] = self.vel_cmd[fwd, 0].abs().clamp(min=0.3)
        self.vel_cmd[fwd, 1:] = 0.0
        self.vel_cmd[self.is_standing] = 0.0  # standing wins over forward

torch.manual_seed(0)
sampler = MaskedCommandSampler(num_envs=16)
sampler.resample(torch.arange(16))
for i in range(16):
    flags = "".join(c for c, b in zip("HSF", [sampler.is_heading[i], sampler.is_standing[i], sampler.is_forward[i]]) if b)
    print(f"env {i:2d} [{flags:3s}] cmd={sampler.vel_cmd[i].tolist()}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
env  0 [   ] cmd=[..., ..., ...]
env  1 [HF ] cmd=[+x_abs_clamped, 0.0, ...]    <- forward dominates: vy zeroed
env  2 [ S ] cmd=[0.0, 0.0, 0.0]              <- standing zeroes everything
env  3 [   ] cmd=[..., ..., ...]
...
```

中文重点:同一行 reset 出现"HF"组合(heading + forward 都抽中),但 forward 的强制 vy=0 会立即覆盖采样值——这种"行为优先级"是 mask 多任务训练的核心机制。

The key thing to notice: a single env may have both H and F flags set, but the forward mode's `vy = 0` clamp immediately overrides the sampled value. This "behavior priority" is the core mechanism of mask-based multi-behavior training.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Isaac Lab `UniformVelocityCommand`** / **Isaac Lab `UniformVelocityCommand`**: mjlab 直接借鉴的版本 —— mjlab 把它移植到 MuJoCo Warp 上,API 几乎一致。 / The version mjlab inherits from — same API, just ported onto MuJoCo Warp.
- **Cassie / Atlas locomotion (DeepMind)** / **Cassie / Atlas locomotion**: 早期论文里就用过"per-env Bernoulli flag"来同时训 standing + walking + turning。 / Early papers already used per-env Bernoulli flags to co-train standing, walking, and turning.
- **VLA dataset balancing** / **VLA dataset balancing**: 类似套路在数据采样阶段也常见 —— 给 batch 里每条数据抽个 "augmentation flag",决定它走哪条 pipeline。 / The same trick appears in dataset balancing: draw an augmentation flag per sample to decide which pipeline it takes.
- **LeRobot AsyncInference batching** / **LeRobot AsyncInference batching**: 不同环境 reset 频率不同时,同样用 boolean mask 标记 "这个 env 现在该跑 high-level policy 还是 low-level"。 / When envs reset at different cadences, boolean masks similarly flag whether an env should run the high-level policy or the low-level one this tick.

## 注意事项 / Caveats / when it breaks

- **行为优先级要明确写下来** / **behavior priority must be explicit**: 一个 env 同时是 standing + forward 时,`_update_command` 必须知道哪个覆盖哪个。这段代码里 standing 最后被设零 → standing wins。如果你加新模式,priority 顺序写错会得到一种没人预料的杂交行为。 / If an env is flagged both standing and forward, `_update_command` must know which wins. Here standing zeros the command at the end → standing wins. Add a new mode in the wrong order and you'll get an unintended hybrid.
- **Bernoulli 概率别加起来超过 1** / **don't let Bernoulli probabilities sum above 1**: 每个标志各自独立抽签,理论上四个 0.4 概率会让 60% 的 env 至少多模式叠加。一般用低概率(0.1-0.2)避免太多冲突。 / Each flag is independent. Four flags at 0.4 each would put ~60% of envs into multi-mode combinations. Low probabilities (0.1-0.2) keep conflicts manageable.
- **`r` 复用 buffer 的副作用** / **reused `r` buffer side effect**: `r.uniform_` 是 in-place 操作,每次调用前一次的结果会被覆盖。代码读起来像"两个 uniform_ 共享同一组随机数",但其实不是 —— 每次调用都生成新数据。 / `r.uniform_` is in-place; each call overwrites the previous draw. The code *reads* like two `uniform_` calls share random numbers, but each call freshly samples.
- **MuJoCo Warp 才能用** / **only on MuJoCo Warp**: mjlab 整个 pipeline 假设底层是 mjwarp.Model/Data,这段命令系统才能跟 sim 同步。直接在原生 MuJoCo 上跑要自己写 sim-side 数据接口。 / The whole mjlab pipeline assumes mjwarp.Model/Data underneath. Running on stock MuJoCo means implementing the sim-side data interface yourself.

## 延伸阅读 / Further reading

- mjlab repo: <https://github.com/mujocolab/mjlab>
- MuJoCo Warp: <https://github.com/google-deepmind/mujoco_warp>
- Isaac Lab (the API mjlab inherits): <https://github.com/isaac-sim/IsaacLab>
- "Massively Parallel Deep RL for Humanoid Locomotion" — the canonical reference for parallel-env command randomization
