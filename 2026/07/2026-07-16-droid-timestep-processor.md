---
date: 2026-07-16
topic: robotics
source: tracked
repo: droid-dataset/droid
file: droid/data_processing/timestep_processing.py
permalink: https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/data_processing/timestep_processing.py#L10-L113
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, dataset, preprocessing, multimodal]
---

# DROID TimestepProcesser：把一帧机器人数据整理成稳定契约 / DROID TimestepProcesser: Turn One Robot Step into a Stable Contract

> **一句话 / In one line**: DROID 先按确定顺序拼接 state、相机标定和动作，再把图像交给统一 transformer，避免同一条轨迹在不同机器上生成不同 schema。 / DROID concatenates state, camera calibration, and action in deterministic order, then sends images through one transformer so the same trajectory does not produce different schemas on different machines.

## 为什么重要 / Why this matters

机器人数据不是一张表，而是一包相机、机器人状态、标定、动作和元数据。训练策略前，最关键的不是模型，而是把这包异构数据变成稳定的输入/输出契约。DROID 的 `TimestepProcesser` 做的就是这件事：排序、过滤、拼接、转换图像、最后附上 action。

Robot data is not a table; it is a bundle of cameras, robot state, calibration, action, and metadata. Before policy training, the hard part is converting that bundle into a stable input/output contract. DROID's `TimestepProcesser` does the unglamorous core work: sort, filter, concatenate, transform images, then attach the action.

## 代码 / The code

`droid-dataset/droid` — [`droid/data_processing/timestep_processing.py`](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/data_processing/timestep_processing.py#L10-L113)

```python
class TimestepProcesser:
    def __init__(
        self,
        ignore_action=False,
        action_space="cartesian_velocity",
        gripper_action_space=None,
        robot_state_keys=["cartesian_position", "gripper_position", "joint_positions", "joint_velocities"],
        camera_extrinsics=["hand_camera", "varied_camera", "fixed_camera"],
        state_dtype=np.float32,
        action_dtype=np.float32,
        image_transform_kwargs={},
    ):
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        self.action_space = action_space
        self.gripper_key = "gripper_velocity" if "velocity" in gripper_action_space else "gripper_position"
        self.image_transformer = ImageTransformer(**image_transform_kwargs)

    def forward(self, timestep):
        timestep = deepcopy(timestep)
        camera_type_dict = {k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()}
        sorted_camera_ids = sorted(camera_type_dict.keys())

        sorted_state_keys = sorted(self.robot_state_keys)
        full_robot_state = timestep["observation"]["robot_state"]
        robot_state = [np.array(full_robot_state[key]).flatten() for key in sorted_state_keys]
        if len(robot_state):
            robot_state = np.concatenate(robot_state)

        calibration_dict = timestep["observation"]["camera_extrinsics"]
        sorted_calibrated_ids = sorted(calibration_dict.keys())
        extrinsics_dict = defaultdict(list)
        for serial_number in sorted_camera_ids:
            cam_type = camera_type_dict[serial_number]
            if cam_type not in self.camera_extrinsics:
                continue
            for full_cam_id in sorted_calibrated_ids:
                if serial_number in full_cam_id:
                    extrinsics_dict[cam_type].append(calibration_dict[full_cam_id])

        low_level_state = np.concatenate([robot_state, extrinsics_state, intrinsics_state], dtype=self.state_dtype)
        processed_timestep = {"observation": {"state": low_level_state, "camera": high_dim_state_dict}}
        self.image_transformer.forward(processed_timestep)

        if not self.ignore_action:
            arm_action = timestep["action"][self.action_space]
            gripper_action = timestep["action"][self.gripper_key]
            action = np.concatenate([arm_action, [gripper_action]], dtype=self.action_dtype)
            processed_timestep["action"] = action
        return processed_timestep
```

## 逐行讲解 / What's happening

1. **动作空间先固定 / The action space is fixed first**: 中文: arm action 只能来自四种受支持空间，gripper action 根据 velocity/position 自动切 key。 English: The arm action must come from one supported space, while the gripper key follows velocity vs position mode.
2. **相机和 state 都排序 / Cameras and state are sorted**: 中文: `sorted(...)` 把字典遍历变成确定顺序，否则同样数据可能拼出不同维度排列。 English: `sorted(...)` turns dictionary traversal into a deterministic schema.
3. **低维 state 是拼接合同 / Low-dimensional state is a concatenation contract**: 中文: robot state、extrinsics、intrinsics 被拼成一个 `float32` 向量。 English: robot state, extrinsics, and intrinsics become one `float32` vector.
4. **图像单独走 transformer / Images go through a separate transformer**: 中文: 高维相机数据留在 nested camera dict 里，交给 `ImageTransformer` 做 resize/augment。 English: high-dimensional camera data stays nested and is normalized by `ImageTransformer`.
5. **action 最后附上 / Action is attached last**: 中文: 训练用 timestep 有 action，纯观测处理可以用 `ignore_action=True` 跳过。 English: training timesteps include action; observation-only processing can skip it.

## 类比 / The analogy

这像把一次实验记录装订成标准表格：先按固定目录放传感器读数，再附相机参数，最后放答案栏。顺序不稳定时，模型学到的是“第 17 列”而不是“夹爪速度”。

It is like binding a lab record into a standard form: sensor values first, camera calibration next, answer field last. If the order is unstable, the model learns "column 17" instead of "gripper velocity."

## 自己跑一遍 / Try it yourself

```python
import numpy as np

state_keys = ["joint", "cart"]
robot_state = {"cart": [1, 2], "joint": [3, 4, 5]}
state = np.concatenate([np.array(robot_state[k]).flatten() for k in sorted(state_keys)])

cam_ids = {"b": "hand", "a": "fixed"}
ordered = [cam_ids[k] for k in sorted(cam_ids)]
arm = np.array([0.1, 0.2, 0.3])
gripper = 0.7
action = np.concatenate([arm, [gripper]]).astype(np.float32)

print(state.tolist())
print(ordered)
print(action.tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 2, 3, 4, 5]
['fixed', 'hand']
[0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.699999988079071]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot dataset transforms** / **LeRobot dataset transforms**: 先把 episode schema 映射到 policy schema。 / They map episode schemas into policy schemas before training.
- **openpi `RepackTransform`** / **openpi `RepackTransform`**: 用显式路径重排嵌套字典。 / It repacks nested dictionaries by explicit paths.

## 注意事项 / Caveats / when it breaks

- **默认参数是可变对象 / Mutable defaults**: `image_transform_kwargs={}` 是老式写法，调用方不要在内部修改它。 / `image_transform_kwargs={}` is old-style Python; callers should not mutate it.
- **相机匹配依赖 serial substring / Camera matching uses serial substrings**: 标定 key 命名变了会漏配。 / Calibration key naming changes can drop matches.
- **state 维度来自配置 / State dimension comes from config**: 改 `robot_state_keys` 会改下游模型输入维度。 / Changing `robot_state_keys` changes the downstream input dimension.

## 延伸阅读 / Further reading

- [DROID `timestep_processing.py`](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/data_processing/timestep_processing.py)
- [DROID dataset](https://github.com/droid-dataset/droid)
