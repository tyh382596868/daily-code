---
date: 2026-10-08
topic: robotics
source: trending
repo: StanfordASL/frax
file: frax/core/manipulator.py
permalink: https://github.com/StanfordASL/frax/blob/cc0a066b5b539c729838308496ae1b8a1d6aeada/frax/core/manipulator.py#L142-L210
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, operational-space-control]
---

# frax control matrices：一次 kinematics，打包控制器要的矩阵 / frax Control Matrices: One Kinematics Pass, Package What the Controller Needs

> **一句话 / In one line**: 这段代码把 operational-space controller 常用的质量矩阵、重力、科氏项、Jacobian 和末端位姿一次算齐。 / This code computes the mass matrix, gravity, Coriolis terms, Jacobian, and end-effector pose needed by operational-space controllers in one pass.

## 为什么重要 / Why this matters

机器人控制循环里，重复算 forward kinematics 很浪费，也容易让矩阵来自不同时间点。frax 的做法是先算一次 `joint_transforms`，然后所有控制矩阵都从这份同一时刻的 kinematic state 派生出来。

In a robot control loop, recomputing forward kinematics repeatedly is wasteful and can make matrices come from slightly different states. frax computes `joint_transforms` once, then derives every control matrix from the same kinematic snapshot.

## 代码 / The code

`StanfordASL/frax` — [`frax/core/manipulator.py`](https://github.com/StanfordASL/frax/blob/cc0a066b5b539c729838308496ae1b8a1d6aeada/frax/core/manipulator.py#L142-L210)

```python
def torque_control_matrices(
    self, q: Array, v: Array
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Compute the matrices required for operational space torque control
    with just a single evaluation of the kinematics

    Args:
        q (Array): Configuration vector, shape (nq,)
        v (Array): Generalized velocities, shape (nv,)

    Returns:
        Tuple[Array, Array, Array, Array, Array, Array]:
            M: Mass matrix, shape (nv, nv)
            M_inv: Inverse of the mass matrix, shape (nv, nv)
            G: Gravity vector, shape (nv,)
            C: Centrifugal/coriolis vector, shape (nv,)
            J: End effector basic Jacobian, shape (6, nv)
            T: End effector transformation matrix, shape (4, 4)
    """
    joint_transforms = self.joint_to_world_transforms(q)
    M = self._mass_matrix(joint_transforms)
    M_inv = self.mass_matrix_inverse(M)
    G = self._gravity_vector(joint_transforms)
    C = self._centrifugal_coriolis_vector(v, joint_transforms)
    J = self._ee_jacobian(joint_transforms)
    T = self._ee_transform(joint_transforms)
    return M, M_inv, G, C, J, T

def velocity_control_matrices(self, q: Array) -> Tuple[Array, Array]:
    """Compute the matrices required for operational space velocity control
    with just a single evaluation of the kinematics

    Args:
        q (Array): Configuration vector, shape (nq,)

    Returns:
        Tuple[Array, Array]:
            J: End effector basic Jacobian, shape (6, nv)
            T: End effector transformation matrix, shape (4, 4)
    """
    joint_transforms = self.joint_to_world_transforms(q)
    J = self._ee_jacobian(joint_transforms)
    T = self._ee_transform(joint_transforms)
    return J, T

def dynamically_consistent_velocity_control_matrices(
    self, q: Array
) -> Tuple[Array, Array, Array]:
    """Compute the matrices required for operational space velocity control
    with just a single evaluation of the kinematics.

    This version also returns the inverse of the mass matrix, which is required
    to construct the dynamically-consistent generalized Jacobian inverse

    Args:
        q (Array): Configuration vector, shape (nq,)

    Returns:
        Tuple[Array, Array, Array]:
            M_inv: Inverse of the mass matrix, shape (nv, nv)
            J: End effector basic Jacobian, shape (6, nv)
            T: End effector transformation matrix, shape (4, 4)
    """
    joint_transforms = self.joint_to_world_transforms(q)
    J = self._ee_jacobian(joint_transforms)
    T = self._ee_transform(joint_transforms)
    M = self._mass_matrix(joint_transforms)
    M_inv = self.mass_matrix_inverse(M)
    return M_inv, J, T
```

## 逐行讲解 / What's happening

1. **第 161 行 / Line 161**:
   - 中文: 所有 torque-control 量共享同一次 `joint_to_world_transforms(q)`，这是性能和一致性的核心。
   - English: All torque-control quantities share one `joint_to_world_transforms(q)` call, which is the core performance and consistency choice.
1. **第 162-168 行 / Lines 162-168**:
   - 中文: torque controller 需要动力学项 `M/M_inv/G/C`，也需要任务空间项 `J/T`，所以一次返回六个对象。
   - English: A torque controller needs dynamics terms `M/M_inv/G/C` and task-space terms `J/T`, so the function returns six objects.
1. **第 170-185 行 / Lines 170-185**:
   - 中文: 速度控制只需要 Jacobian 和末端 transform，省掉质量矩阵和速度相关项。
   - English: Velocity control only needs the Jacobian and end-effector transform, so it skips mass and velocity-dependent terms.
1. **第 187-210 行 / Lines 187-210**:
   - 中文: 动力学一致的速度控制额外返回 `M_inv`，用于构造 mass-weighted Jacobian inverse。
   - English: Dynamically consistent velocity control additionally returns `M_inv` for building a mass-weighted Jacobian inverse.

## 类比 / The analogy

像手术前的器械托盘：不同手术步骤要的工具不同，但护士先按同一份病人信息把本轮可能用到的工具一次摆好。

It is like a surgical instrument tray. Different steps need different tools, but the nurse prepares each tray from the same patient state before the procedure begins.

## 自己跑一遍 / Try it yourself

```python
class Arm:
    def transforms(self, q): return {"q": q, "sum": sum(q)}
    def mass(self, tr): return [[1 + tr["sum"], 0], [0, 1]]
    def jacobian(self, tr): return [1, tr["sum"]]
    def pose(self, tr): return [tr["sum"], 0, 0]
    def velocity_pack(self, q):
        tr = self.transforms(q)
        return self.jacobian(tr), self.pose(tr)
    def torque_pack(self, q):
        tr = self.transforms(q)
        m = self.mass(tr)
        return m, self.jacobian(tr), self.pose(tr)

arm = Arm()
print(arm.velocity_pack([0.2, 0.3]))
print(arm.torque_pack([0.2, 0.3]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
([1, 0.5], [0.5, 0, 0])
([[1.5, 0], [0, 1]], [1, 0.5], [0.5, 0, 0])
```

两个 pack 都先建立同一份 transforms，再按控制器需要取不同矩阵。

Both packs build the same transforms first, then select the matrices each controller needs.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Pinocchio operational space control** / **Pinocchio operational space control**: 任务空间控制也会围绕 `M`、`J`、`Jdot`、重力项组织缓存。 / Task-space control is similarly organized around cached `M`, `J`, `Jdot`, and gravity terms.
- **MuJoCo controllers** / **MuJoCo controllers**: 仿真器通常先更新整机 state，再让多个 controller 读取同一帧数据。 / Simulators usually update robot state once, then let multiple controllers read the same frame.

## 注意事项 / Caveats / when it breaks

- **缓存只对同一个 q/v 有效** / **The cache is only valid for the same q/v**: 控制循环里状态更新后，上一帧矩阵不能继续用。 / Once the state changes in the control loop, matrices from the previous frame are stale.
- **矩阵逆要注意数值稳定** / **Matrix inverses need numerical care**: 接近奇异构型时，`M_inv` 和 Jacobian inverse 都可能放大噪声。 / Near singular configurations, `M_inv` and Jacobian inverses can amplify noise.

## 延伸阅读 / Further reading

- [frax repository](https://github.com/StanfordASL/frax)
- [Operational space control overview](https://en.wikipedia.org/wiki/Robot_kinematics)
