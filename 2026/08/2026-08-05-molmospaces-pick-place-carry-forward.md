---
date: 2026-08-05
topic: robotics
source: trending
repo: allenai/molmospaces
file: molmo_spaces/tasks/pick_and_place_task.py
permalink: https://github.com/allenai/molmospaces/blob/7db0aaf8ba7134a54c973fffaca01347a2fa3412/molmo_spaces/tasks/pick_and_place_task.py#L140-L179
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, molmospaces, pick-and-place, success-metric]
---

# MolmoSpaces pick-place：成功状态需要 carry-forward / MolmoSpaces Pick-Place: Success State Needs Carry-Forward

> **一句话 / In one line**: 当接触力短暂消失时，MolmoSpaces 用已确认支撑时的相对位姿缓存，判断物体是否仍停在容器上。 / When contact force briefly disappears, MolmoSpaces uses cached relative poses from confirmed support moments to decide whether the object is still on the receptacle.

## 为什么重要 / Why this matters

机器人 benchmark 的 success metric 不能只看单帧接触。轻物体、静止摩擦、仿真接触求解器都会让“明明放好了”的物体在某一帧没有足够接触力。如果 success 立刻变 false，数据生成会误杀好轨迹。这里用相对位姿 carry-forward：只要物体相对容器没有漂移，就承认之前的支撑状态仍然成立。

A robot benchmark success metric cannot rely only on single-frame contact. Light objects, static friction, and simulator contact solvers can produce a frame with too little contact force even when the object is correctly placed. If success flips to false immediately, data generation rejects good trajectories. This code carries support forward through relative pose: if the object has not drifted relative to the receptacle, the previously confirmed support still counts.

## 代码 / The code

`allenai/molmospaces` — [`molmo_spaces/tasks/pick_and_place_task.py`](https://github.com/allenai/molmospaces/blob/7db0aaf8ba7134a54c973fffaca01347a2fa3412/molmo_spaces/tasks/pick_and_place_task.py#L140-L179)

```python
# Relative-pose carry-forward: track the object's pose in the
# receptacle's frame. When contact-based checks detect support,
# snapshot this relative pose. When they fail (e.g. lightweight
# objects at equilibrium losing contact forces), compare against
# the snapshot — if the relative pose hasn't drifted, the object
# is still where it was when support was last confirmed.
rel_pose = np.linalg.solve(place_receptacle.pose, pickup_obj.pose)

carried_forward = False
carry_forward_pos_diff = float("inf")
carry_forward_rot_diff = float("inf")
if supported_by_receptacle:
    self._supported_rel_poses.setdefault(i, []).append(rel_pose.copy())
    carry_forward_pos_diff = 0.0
    carry_forward_rot_diff = 0.0
if i in self._supported_rel_poses:
    # Track nearest-by-position for diagnostics, and separately
    # check if any cached pose satisfies both thresholds jointly
    # (greedy nearest-by-position can miss a slightly farther pose
    # that satisfies rotation while the nearest one doesn't).
    best_pos_diff = float("inf")
    best_rot_diff = float("inf")
    match_found = False
    for stored in self._supported_rel_poses[i]:
        pd = float(np.linalg.norm(rel_pose[:3, 3] - stored[:3, 3]))
        rd = float(R.from_matrix(rel_pose[:3, :3] @ stored[:3, :3].T).magnitude())
        if pd < best_pos_diff:
            best_pos_diff = pd
            best_rot_diff = rd
        if (
            pd <= task_config.carry_forward_rel_pos_threshold
            and rd <= task_config.carry_forward_rel_rot_threshold
        ):
            match_found = True
    carry_forward_pos_diff = best_pos_diff
    carry_forward_rot_diff = best_rot_diff
    if not supported_by_receptacle and match_found:
        supported_by_receptacle = True
        carried_forward = True
```

## 逐行讲解 / What's happening

1. **第 140-146 行 / Lines 140-146 (relative frame)**:
   - 中文: `np.linalg.solve(place_receptacle.pose, pickup_obj.pose)` 把物体位姿表达在容器坐标系里，而不是世界坐标里。
   - English: `np.linalg.solve(place_receptacle.pose, pickup_obj.pose)` expresses the object pose in the receptacle frame rather than the world frame.
2. **第 151-154 行 / Lines 151-154 (snapshot confirmed support)**:
   - 中文: 当接触/启发式已经确认支撑时，保存这一帧的相对位姿，并把诊断差值置零。
   - English: When contact or heuristics already confirm support, the relative pose is saved and diagnostic differences are set to zero.
3. **第 163-175 行 / Lines 163-175 (search cached poses)**:
   - 中文: 当前相对位姿会和所有历史支撑快照比较；位置和旋转同时过阈值才算 match。
   - English: The current relative pose is compared against all cached support snapshots; both position and rotation thresholds must pass.
4. **第 176-179 行 / Lines 176-179 (carry forward)**:
   - 中文: 如果当前帧没检测到支撑，但位姿和历史支撑状态足够接近，就把 `supported_by_receptacle` 改回 true，并记录这是 carry-forward。
   - English: If current support is not detected but the pose matches a past supported state, `supported_by_receptacle` is restored to true and marked as carry-forward.

## 类比 / The analogy

这像看杯子是否还在桌上。摄像头某一帧被手挡住了，你不会马上判定杯子掉了；如果遮挡前后杯子相对桌角的位置没变，就承认它还在桌上。

It is like checking whether a cup is still on a table. If a hand blocks the camera for one frame, you do not immediately declare that the cup fell. If its position relative to the table corner is unchanged before and after, it still counts as being on the table.

## 自己跑一遍 / Try it yourself

```python
stored = [(0.10, 0.02)]
current = (0.11, 0.025)
pos_threshold = 0.02
rot_threshold = 0.02
supported_now = False

match = any(abs(current[0] - p) <= pos_threshold and abs(current[1] - r) <= rot_threshold for p, r in stored)
if not supported_now and match:
    supported_now = True
    carried_forward = True

print(supported_now, carried_forward)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
True True
```

中文: 当前帧没有直接支撑信号，但相对位姿足够接近历史成功状态，所以 success 不会闪断。

English: The current frame has no direct support signal, but the relative pose is close enough to a previously supported state, so success does not flicker off.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **contact debouncing** / **contact debouncing**: 机器人控制里常用时间窗口过滤接触信号抖动。 / Robot control often filters contact flicker with short temporal windows.
- **object permanence in tracking** / **object permanence in tracking**: 视觉跟踪也会用上一帧状态穿过短暂遮挡。 / Visual tracking also carries state across brief occlusions.

## 注意事项 / Caveats / when it breaks

- **阈值太松会误判 / Loose thresholds cause false positives**: 物体已经滑走时仍可能被当成成功。 / An object that has slid away may still count as successful.
- **缓存要按 episode 清空 / Cache must reset per episode**: 否则上一个 episode 的支撑位姿会污染当前任务。 / Otherwise a previous episode's support pose can contaminate the current task.

## 延伸阅读 / Further reading

- MolmoSpaces pick-and-place success metric: https://github.com/allenai/molmospaces/blob/7db0aaf8ba7134a54c973fffaca01347a2fa3412/molmo_spaces/tasks/pick_and_place_task.py#L140-L179
