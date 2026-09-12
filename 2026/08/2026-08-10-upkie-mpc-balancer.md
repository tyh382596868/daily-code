---
date: 2026-08-10
topic: robotics
source: trending
repo: upkie/upkie
file: upkie/controllers/mpc_balancer.py
permalink: https://github.com/upkie/upkie/blob/831f31aa8c7d5e99d509d59a660e82449abb4ea1/upkie/controllers/mpc_balancer.py#L237-L312
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, robotics, mpc]
---

# Upkie MPCBalancer：每一拍都重规划平衡速度 / Upkie MPCBalancer: Replan Balance Velocity at Every Tick

> **一句话 / In one line**: `MPCBalancer.step()` 把最新姿态和轮速变成 MPC 状态，解一个 QP，然后输出下一拍的地面速度。 / `MPCBalancer.step()` converts fresh attitude and wheel odometry into an MPC state, solves a QP, and emits the next ground velocity.

## 为什么重要 / Why this matters

轮腿倒立摆机器人不能只跟踪目标速度；它还必须不断平衡机身 pitch。Upkie 的 controller 每个周期都用当前观测刷新 MPC 问题，在“向前走”和“别摔倒”之间求一个约束优化解。

A wheeled biped cannot merely track target velocity; it must keep its body pitch balanced. Upkie refreshes the MPC problem every cycle and solves a constrained optimization between “move forward” and “do not fall.”

## 代码 / The code

`upkie/upkie` — [`upkie/controllers/mpc_balancer.py`](https://github.com/upkie/upkie/blob/831f31aa8c7d5e99d509d59a660e82449abb4ea1/upkie/controllers/mpc_balancer.py#L237-L312)

```python
def step(
    self,
    target_ground_velocity: float,
    spine_observation: dict,
    dt: float,
    qp_solver: str = "proxqp",
) -> float:
    floor_contact = spine_observation["floor_contact"]["contact"]
    base_orientation = spine_observation["base_orientation"]
    base_pitch = base_orientation["pitch"]
    base_angular_velocity = base_orientation["angular_velocity"][1]
    ground_position = spine_observation["wheel_odometry"]["position"]
    ground_velocity = spine_observation["wheel_odometry"]["velocity"]

    fallen = abs(base_pitch) > self.fall_pitch
    if fallen and not self.fallen:
        logger.warning(f"Base angle {base_pitch=:.3} rad denotes a fall")
    self.fallen = fallen

    cur_state = np.array(
        [ground_position, base_pitch, ground_velocity, base_angular_velocity]
    )

    nx = self.pendulum.STATE_DIM
    target_states = get_target_states(self.pendulum, cur_state, target_ground_velocity)
    self.mpc_problem.update_initial_state(cur_state)
    self.mpc_problem.update_goal_state(target_states[-nx:])
    self.mpc_problem.update_target_states(target_states[:-nx])

    self.mpc_qp.update_cost_vector(self.mpc_problem)
    if self.warm_start:
        qpsol = self.proxqp.solve_with_warm_start(self.mpc_qp)
    else:
        qpsol = qpsolvers.solve_problem(self.mpc_qp.problem, solver=qp_solver)
    if not qpsol.found:
        logger.warning("No solution found to the MPC problem")
    plan = Plan(self.mpc_problem, qpsol)

    if fallen or not floor_contact:
        self.commanded_velocity = low_pass_filter(
            prev_output=self.commanded_velocity,
            cutoff_period=0.1,
            new_input=0.0,
            dt=dt,
        )
    elif plan.is_empty:
        logger.error("Solver found no solution to the MPC problem")
        logger.info("Re-sending previous ground velocity")
    else:
        self.pendulum.state = cur_state
        commanded_accel = plan.first_input[0]
        self.commanded_velocity = clamp_abs(
            self.commanded_velocity + commanded_accel * dt / 2.0,
            self.max_ground_velocity,
        )
    return self.commanded_velocity
```

## 逐行讲解 / What's happening

1. **第 253-258 行 / Lines 253-258 (observation unpacking)**:
   - 中文: controller 只抽取平衡需要的量：接触、pitch、pitch rate、轮位置和轮速度。
   - English: The controller extracts only what balance needs: contact, pitch, pitch rate, wheel position, and wheel velocity.
2. **第 260-273 行 / Lines 260-273 (state vector)**:
   - 中文: `cur_state` 按倒立摆模型约定排列，注释提醒结构来自 `WheeledInvertedPendulum`。
   - English: `cur_state` follows the `WheeledInvertedPendulum` state convention.
3. **第 275-283 行 / Lines 275-283 (moving horizon)**:
   - 中文: 目标轨迹每拍重算，MPC 的 initial/goal/target states 都随观测刷新。
   - English: The target trajectory is recomputed every tick, refreshing initial, goal, and target states.
4. **第 284-293 行 / Lines 284-293 (solve QP)**:
   - 中文: warm-start 时复用 ProxQP workspace，否则走通用 `qpsolvers`。
   - English: With warm-starting, it reuses a ProxQP workspace; otherwise it calls generic `qpsolvers`.
5. **第 295-312 行 / Lines 295-312 (safe output)**:
   - 中文: 摔倒或离地时低通降到 0；求解成功时只取第一拍加速度，更新并夹紧速度。
   - English: On a fall or lost contact it filters toward zero; on success it applies only the first acceleration and clamps velocity.

## 类比 / The analogy

像骑独轮车的人每一步都重新看身体倾角：目标是往前走，但如果身体要倒了，下一脚必须先救平衡。

It is like a unicyclist checking body lean at every pedal stroke: moving forward matters, but if the body is falling, the next move must recover balance first.

## 自己跑一遍 / Try it yourself

```python
def clamp_abs(x, limit):
    return max(-limit, min(limit, x))

commanded_velocity = 0.4
commanded_accel = 3.0
dt = 0.02
max_ground_velocity = 0.42
commanded_velocity = clamp_abs(commanded_velocity + commanded_accel * dt / 2.0, max_ground_velocity)
print(round(commanded_velocity, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.42
```

这个 toy 只保留最后一步：MPC 产出加速度后，controller 积分半步并用速度上限保护执行器。

This toy keeps only the final step: after MPC proposes acceleration, the controller integrates a half step and clamps the velocity.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Legged locomotion MPC** / **Legged locomotion MPC**: 每个控制周期只执行 receding horizon 的第一步。 / Each control cycle executes only the first step of a receding horizon.
- **Autonomous driving controllers** / **Autonomous driving controllers**: 轨迹规划器反复用最新状态重解优化问题。 / Planners repeatedly resolve the optimization with the newest state.

## 注意事项 / Caveats / when it breaks

- **观测延迟会伤害稳定性** / **Observation delay hurts stability**: pitch 和轮速滞后时，MPC 解的是过期状态。 / If pitch and wheel velocity lag, MPC solves for stale state.
- **QP 失败要有 fallback** / **QP failure needs fallback**: 这里失败时重发上一速度；真实机器人还需要更完整的安全策略。 / This code resends the previous velocity; real robots need broader safety handling.

## 延伸阅读 / Further reading

- [Upkie `MPCBalancer.step`](https://github.com/upkie/upkie/blob/831f31aa8c7d5e99d509d59a660e82449abb4ea1/upkie/controllers/mpc_balancer.py#L237-L312)
