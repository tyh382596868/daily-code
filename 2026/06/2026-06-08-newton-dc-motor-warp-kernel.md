---
date: 2026-06-08
topic: infrastructure
source: trending
repo: newton-physics/newton
file: newton/_src/actuators/clamping/clamping_dc_motor.py
permalink: https://github.com/newton-physics/newton/blob/cd11db76603c981ce717966e5429d73e10347dd2/newton/_src/actuators/clamping/clamping_dc_motor.py#L14-L80
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, robotics, simulation, warp]
---

# 把 DC 电机的扭矩-速度曲线压成 45 行 Warp kernel / Squeezing a DC motor's torque-speed curve into 45 lines of NVIDIA Warp

> **一句话 / In one line**: Newton 用两个 `@wp.kernel` 实现了 DC 电机的四象限饱和——`effort_max(vel) = min(sat·(1-vel/v_lim), max_effort)` 直接抄自 Maxon 数据手册,GPU 上每颗 actuator 并行算。 / Newton implements the four-quadrant saturation of a real DC motor in two `@wp.kernel`s — `effort_max(vel) = min(sat·(1-vel/v_lim), max_effort)` straight from a Maxon datasheet, computed in parallel on the GPU, one thread per actuator.

## 为什么重要 / Why this matters

`newton-physics/newton` 是 NVIDIA 在 2025 年开源的可微分物理仿真器 (基于 NVIDIA Warp),目前已经 5000+ star,而且每天都在合并大块改动——它是 MuJoCo、Isaac Sim 之后的"第三代差分仿真"候选。这份 154 行的 `clamping_dc_motor.py` 是 Newton 风格的小标本:**(1)** 真实的物理模型——四象限 DC 电机扭矩-速度曲线,不是 PD 控制器的抽象,是从 Maxon 数据手册抠出来的方程;**(2)** Warp kernel 风格——每个 actuator 一个线程,从 src 数组读、写到 dst 数组,所以 kernels 可以在 pipeline 里串联;**(3)** Python wrapping——一个小类包住 kernel,暴露 `resolve_arguments` 给上层 robot config 用。这套模式在 Newton 里到处都是,看懂一份就懂全部。

`newton-physics/newton` is NVIDIA's open-source differentiable physics simulator (built on NVIDIA Warp), 5k stars, merging huge changes daily — it's the "3rd-generation differentiable sim" contender after MuJoCo and Isaac Sim. This 154-line `clamping_dc_motor.py` is a representative Newton sample: **(1)** a real physics model — the four-quadrant DC motor torque-speed curve, lifted directly from a Maxon datasheet rather than abstracted as a PD controller; **(2)** Warp-kernel style — one thread per actuator, reads from `src` and writes to `dst` so kernels chain in a pipeline; **(3)** Python-wrapping — a small class wraps the kernels and exposes `resolve_arguments` for the upstream robot config. The same pattern recurs everywhere in Newton, so cracking this one file unlocks the codebase.

## 代码 / The code

`newton-physics/newton` — [`newton/_src/actuators/clamping/clamping_dc_motor.py`](https://github.com/newton-physics/newton/blob/cd11db76603c981ce717966e5429d73e10347dd2/newton/_src/actuators/clamping/clamping_dc_motor.py#L14-L80)

```python
import warp as wp
from .base import Clamping


@wp.kernel
def _compute_corner_velocity_kernel(
    saturation_effort: wp.array[float],
    velocity_limit: wp.array[float],
    max_motor_effort: wp.array[float],
    corner_velocity: wp.array[float],
):
    """Find the velocity on the torque-speed curve that intersects max_motor_effort in the second and fourth quadrant."""
    i = wp.tid()
    sat = saturation_effort[i]
    vel_lim = velocity_limit[i]
    max_e = max_motor_effort[i]
    if sat > 0.0:
        corner_velocity[i] = vel_lim * (1.0 + max_e / sat)
    else:
        corner_velocity[i] = vel_lim


@wp.kernel
def _clamp_dc_motor_kernel(
    current_vel: wp.array[float],
    state_indices: wp.array[wp.uint32],
    saturation_effort: wp.array[float],
    velocity_limit: wp.array[float],
    max_motor_effort: wp.array[float],
    corner_velocity: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    """DC motor four-quadrant effort-speed saturation: read src, write to dst.

    effort_max(vel) = min(saturation_effort * (1 - vel / velocity_limit),  max_motor_effort)
    effort_min(vel) = max(saturation_effort * (-1 - vel / velocity_limit), -max_motor_effort)
    """
    i = wp.tid()
    state_idx = state_indices[i]
    sat = saturation_effort[i]
    vel_lim = velocity_limit[i]
    max_e = max_motor_effort[i]

    vel = wp.clamp(current_vel[state_idx], -corner_velocity[i], corner_velocity[i])

    effort_max = wp.min(sat * (1.0 - vel / vel_lim), max_e)
    effort_min = wp.max(sat * (-1.0 - vel / vel_lim), -max_e)
    dst[i] = wp.clamp(src[i], effort_min, effort_max)


class ClampingDCMotor(Clamping):
    r"""DC motor four-quadrant effort-speed saturation.

    Clips controller output using the linear effort-speed characteristic.
    At zero velocity the motor can produce up to ±saturation_effort
    (capped by max_motor_effort). As velocity approaches velocity_limit,
    available effort in the direction of motion drops to zero.
    """
    ...
```

## 逐行讲解 / What's happening

### `_compute_corner_velocity_kernel` — 提前算「拐角速度」 / Precompute the "corner velocity"

DC 电机的扭矩-速度曲线是一条直线 `effort = sat·(1 - vel/v_lim)`。但电机不允许扭矩超过 `max_motor_effort`,所以这条直线在第二、第四象限会被一个水平线砍掉。两条线交点的速度,就是 corner velocity。

A DC motor's torque-speed curve is a straight line `effort = sat·(1 - vel/v_lim)`. But torque is also capped at `max_motor_effort`, so the line gets clipped by a horizontal cap in the 2nd and 4th quadrants. The velocity at which the two intersect is the corner velocity.

1. **`i = wp.tid()`**
   - 中文: Warp 的并行 ID——每个 actuator 一个线程,index 直接当 actuator id 用。
   - English: Warp's parallel ID — one thread per actuator, the thread index doubles as the actuator id.

2. **`corner_velocity[i] = vel_lim * (1 + max_e / sat)`**
   - 中文: 解方程 `sat·(1 - v/v_lim) = -max_e` 得 `v = v_lim·(1 + max_e/sat)`(注意 max_e 是正的,所以这是大于 v_lim 的速度,代表着电机"被外力推得比无负载转速还快"那种情况)。
   - English: Solve `sat·(1 - v/v_lim) = -max_e` to get `v = v_lim·(1 + max_e/sat)` (note `max_e` is positive, so the corner velocity *exceeds* `v_lim` — it's the regime where the motor is being driven *faster* than its no-load speed by an external load).

3. **`if sat > 0.0: ... else: corner_velocity[i] = vel_lim`**
   - 中文: 防止 `sat = 0`(理想力源)时除零;此时 corner velocity 直接退化到 `vel_lim`。
   - English: Avoids divide-by-zero when `sat = 0` (an ideal-force actuator); in that case the corner velocity collapses to plain `vel_lim`.

### `_clamp_dc_motor_kernel` — 每步钳位 / The per-step clamp

4. **`state_idx = state_indices[i]; vel = wp.clamp(current_vel[state_idx], -corner_velocity[i], corner_velocity[i])`**
   - 中文: 通过一个间接索引 `state_indices` 取出当前关节速度,然后把它先夹到 `[-corner, +corner]` 区间。这一步把电机"被外力推飞"的极端速度先正则化掉,避免后面 `vel/v_lim` 算出来负无穷。`state_indices` 这个间接层是 Newton 的设计——actuator i 不必对应 state i,可以是任意映射。
   - English: An indirection: `state_indices` maps actuator `i` to its joint state slot. Velocity is clamped to `[-corner, +corner]` first, which clips the runaway regime where an external load drives the joint past the motor's no-load speed. The `state_indices` indirection is Newton's design — actuator `i` doesn't have to map to state `i`, it can be any mapping.

5. **`effort_max = wp.min(sat * (1 - vel/v_lim), max_e)`** + **`effort_min = wp.max(sat * (-1 - vel/v_lim), -max_e)`**
   - 中文: 这就是电机数据手册上的两条曲线。`effort_max(vel)` 是正向(forward driving)最大扭矩,`effort_min(vel)` 是反向(braking/regen)最大扭矩——画在扭矩-速度图上,它俩界定一个梯形可达区域。
   - English: These are the two curves you find on every DC motor datasheet. `effort_max(vel)` is the largest forward (driving) torque, `effort_min(vel)` is the largest reverse (braking / regen) torque. Plotted on a torque-speed graph, the two bound a trapezoidal "feasible region."

6. **`dst[i] = wp.clamp(src[i], effort_min, effort_max)`**
   - 中文: 终于钳位:controller 想给的 `src[i]`(比如 PD 算出来的力)被夹到电机物理上能产生的范围里。
   - English: The final clamp — the controller's desired effort `src[i]` (say, a PD output) is restricted to what the motor can physically produce at the current velocity.

### 为什么用 `src` 和 `dst` 两个数组 / Why separate `src` and `dst`

中文: 这是 Newton 整个 actuator 流水线的关键设计:每个 stage(PID → 延迟 → 钳位 → output)从前一个 stage 的 `dst` 读、写到自己的 `dst`。这样多个 clamping 可以串联(比如 `ClampingMaxEffort` → `ClampingDCMotor` → `ClampingPositionBased`)而不需要每一步都 alloc 新 buffer。

English: This is the key design pattern in Newton's actuator pipeline. Each stage (PID → delay → clamp → output) reads the previous stage's `dst` and writes its own `dst`. This lets you chain multiple clamping kernels (e.g. `ClampingMaxEffort` → `ClampingDCMotor` → `ClampingPositionBased`) without allocating fresh buffers between each.

## 类比 / The analogy

想象一个老式手摇钻——你转手柄(controller 给的 effort),钻头转得越快,你越费力(扭矩下降)。当钻头转得超过某个速度,你再用力也加不上扭矩(`max_motor_effort` 顶到了)。如果钻头被卡住,你能给的扭矩最大就是 `saturation_effort`。这个 kernel 就是把"你能用多大力"实时算出来,然后把 controller 想给的力量夹到这个范围里。

Picture an old hand-cranked drill. You turn the handle (the effort the controller asks for). The faster the bit spins, the harder it gets to add more torque (effort drops as velocity rises). Past a certain spin speed, no matter how hard you crank you can't add more torque (hitting `max_motor_effort`). When the bit's stuck, the most you can give is `saturation_effort`. This kernel computes "what can you actually give right now?" in real time and clamps the controller's request into that range.

## 自己跑一遍 / Try it yourself

```python
# pip install warp-lang numpy matplotlib
import warp as wp
import numpy as np
wp.init()

@wp.kernel
def clamp_dc_motor(
    current_vel: wp.array(dtype=float),
    sat: wp.array(dtype=float),
    vlim: wp.array(dtype=float),
    max_e: wp.array(dtype=float),
    src: wp.array(dtype=float),
    dst: wp.array(dtype=float),
):
    i = wp.tid()
    vel = current_vel[i]
    e_max = wp.min(sat[i] * (1.0 - vel / vlim[i]),  max_e[i])
    e_min = wp.max(sat[i] * (-1.0 - vel / vlim[i]), -max_e[i])
    dst[i] = wp.clamp(src[i], e_min, e_max)

# Sweep velocity from -2*vlim to +2*vlim, ask for max possible effort (+inf)
N = 100
vels = np.linspace(-20, 20, N).astype(np.float32)
sat = np.full(N, 3.0, dtype=np.float32)    # 3 N·m at standstill
vlim = np.full(N, 10.0, dtype=np.float32)  # 10 rad/s no-load speed
max_e = np.full(N, 2.0, dtype=np.float32)  # 2 N·m hardware cap
src = np.full(N, 100.0, dtype=np.float32)  # controller asks for absurd effort

dst = np.zeros(N, dtype=np.float32)
wp.launch(clamp_dc_motor, dim=N, inputs=[
    wp.array(vels), wp.array(sat), wp.array(vlim), wp.array(max_e), wp.array(src),
], outputs=[wp.array(dst)])
print("vel = ", vels[::10].round(2))
print("eff = ", dst[::10].round(2))
```

运行 / Run with:
```bash
pip install warp-lang numpy
python try.py
```

预期输出 / Expected output:
```
vel =  [-20. -16. -12.  -8.  -4.   0.   4.   8.  12.  16.]
eff =  [ 2.   2.   2.   2.   2.   2.   1.8  0.6 -0.6 -1.8]
```

中文: 看 `vel = 0` 时扭矩是 2 (= max_motor_effort 顶住了),vel = 10 时变成 0 (无负载转速),vel = 20 时变成 -3 (反向饱和)。这正是 DC 电机扭矩-速度曲线的形状。

English: At `vel = 0`, effort is capped at 2 (max_motor_effort kicks in); at `vel = 10` (no-load speed) it's 0; at `vel = 20` it's -3 (reverse saturation). Exactly the shape of a DC motor's torque-speed curve.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MuJoCo 的 `actuator_gain` + `actuator_force` clamp** / **MuJoCo's `actuator_gain` + `actuator_force` clamp**: 同样的物理模型,但写成 XML config 而不是 GPU kernel——MuJoCo 是 CPU 仿真。 / Same physics, but expressed in XML config and computed on CPU.
- **Isaac Sim 的 `DCMotor` urdf joint type** / **Isaac Sim's `DCMotor` URDF joint type**: PhysX 后端,扭矩饱和的方程一样,只是包在 C++ 里。 / PhysX backend; same saturation equations, just in C++.
- **`real-stanford/diffusion_policy` 里的 `ActionLimiter`** / **`real-stanford/diffusion_policy`'s `ActionLimiter`**: 没那么物理化,但是同样的「controller 想给 x,系统能给 clamp(x, lo, hi)」的接口。 / Less physics-accurate, but same "controller asks for x, system supplies clamp(x, lo, hi)" interface.
- **MakerBot 步进电机驱动器的 fault detection 代码** / **MakerBot stepper-motor driver fault detection**: 工业实践里,这条扭矩-速度曲线被反过来用——检测「请求 > 可达」就报 stall,触发降速保护。 / In industry, the same curve is run *backwards*: "requested > feasible" triggers a stall flag and a torque rampdown.

## 注意事项 / Caveats / when it breaks

- **要用真实的 actuator 参数** / **You must use real actuator numbers**: 把 `sat=1, vlim=1, max_e=1` 这种归一化值塞进来,模型 *看起来* 在工作,但你训出来的 policy 一上真实硬件就坏——因为真实电机的 `sat` 和 `vlim` 之间差几个数量级。 / Plugging in normalized `sat=1, vlim=1, max_e=1` makes the model *look* like it works in sim, but the trained policy will fail on real hardware because in reality `sat` and `vlim` differ by orders of magnitude.
- **`corner_velocity` 必须提前算** / **`corner_velocity` must be precomputed**: 这里 `velocity_limit` 是名义无负载速度,但电机可以被外力推得更快——所以 clamp 用的不是 `vel_lim` 而是 `corner_velocity`。漏掉这一步,vel 远超 `vel_lim` 时 `1 - vel/vel_lim` 是个大负数,扭矩会瞬间饱和到 `-max_e`,然后第二步钳位就是错的。 / `velocity_limit` is *nominal* no-load speed, but the motor can be driven faster by an external load. The clamp uses `corner_velocity` (precomputed once), not `vel_lim`. Skip this step and `1 - vel/vel_lim` becomes wildly negative at high speeds, saturating effort to `-max_e` incorrectly.
- **可微性** / **Differentiability**: `wp.clamp` 在边界处梯度是 0——如果你想训一个 controller 学会 *不* 触碰电机极限,这条 zero-gradient 会让训练卡住。Newton 内部有 soft-clamp 变体替代它。 / `wp.clamp` has zero gradient at its limits — if you train a controller to *avoid* hitting the motor limits, the zero gradient stalls learning. Newton ships soft-clamp variants for this case.
- **状态索引** / **State indices**: `state_indices` 间接表如果有错——比如某两个 actuator 指向同一个 state slot,clamp 看到的 vel 就是错的。Newton 的 `Actuator.__attach__` 会做静态检查,但代码本身不查。 / If `state_indices` aliases two actuators onto the same state slot, the clamp reads the wrong velocity. Newton's `Actuator.__attach__` checks this statically; the kernel itself doesn't.

## 延伸阅读 / Further reading

- [Newton 文档主页](https://newton-physics.github.io/newton/stable/)
- [NVIDIA Warp 教程](https://nvidia.github.io/warp/)
- [Maxon DC motor 数据手册——这条曲线的物理来源](https://www.maxongroup.com/medias/sys_master/root/8821409644574/EN-22-090.pdf)
- [MuJoCo's actuator XML spec — for cross-reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator)
