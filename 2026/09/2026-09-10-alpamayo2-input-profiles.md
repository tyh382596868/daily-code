---
date: 2026-09-10
topic: infrastructure
source: trending
repo: NVlabs/alpamayo2
file: src/alpamayo2_super/input_profiles.py
permalink: https://github.com/NVlabs/alpamayo2/blob/main/src/alpamayo2_super/input_profiles.py#L21-L189
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, autonomous-driving, input-contracts, vla]
---

# Alpamayo2 input profiles：任务先选相机和帧 / Alpamayo2 Input Profiles: Choose Cameras and Frames Per Task

> **一句话 / In one line**: Alpamayo2 用 frozen `InputProfile` 明确每个 public task 的相机顺序和帧索引，并在切片后再次校验。 / Alpamayo2 uses frozen `InputProfile` records to define each public task's camera order and frame indices, then validates the selected payload after slicing.

## 为什么重要 / Why this matters

中文：自动驾驶 VLA/世界模型的输入不是“一堆图像”。前后左右相机顺序、每个相机取几帧、时间戳如何对齐，都会影响模型对轨迹的理解。Alpamayo2 把这些规则写成 task profile，再通过 `select_task_input` 统一切片和校验，避免 notebook、CLI、可视化各自理解一套输入格式。

English: An autonomous-driving VLA/world model does not consume "some images." Camera order, frame count, and timestamp alignment all affect trajectory reasoning. Alpamayo2 encodes those rules as task profiles, then routes selection and validation through `select_task_input`, keeping notebooks, CLI, and visualization on one input contract.

## 代码 / The code

`NVlabs/alpamayo2` — [`src/alpamayo2_super/input_profiles.py`](https://github.com/NVlabs/alpamayo2/blob/main/src/alpamayo2_super/input_profiles.py#L21-L189)

```python
@dataclass(frozen=True)
class InputProfile:
    """Camera IDs and source-frame indices consumed by one inference task."""

    camera_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]

VQA_SIX_CAMERA_FOUR_FRAME = InputProfile(
    camera_ids=(0, 1, 2, 3, 4, 5),
    frame_indices=(0, 1, 2, 3),
)
DRIVING_SIX_CAMERA_FOUR_FRAME = InputProfile(
    camera_ids=(0, 1, 2, 3, 5, 6),
    frame_indices=(0, 1, 2, 3),
)
TASK_INPUT_PROFILES = {
    "trajectory": DRIVING_SIX_CAMERA_FOUR_FRAME,
    "meta_action": DRIVING_SIX_CAMERA_FOUR_FRAME,
    "auto_labeling": DRIVING_SIX_CAMERA_FOUR_FRAME,
    "vqa": VQA_SIX_CAMERA_FOUR_FRAME,
    "grounding": DRIVING_SIX_CAMERA_FOUR_FRAME,
}
```

Selection and validation happen in the same module:

```python
camera_positions = [source_camera_ids.index(camera_id) for camera_id in profile.camera_ids]
camera_selector = torch.tensor(camera_positions, dtype=torch.long, device=camera_indices.device)
frame_selector = torch.tensor(frame_indices, dtype=torch.long, device=image_frames.device)
selected = dict(data)
selected["camera_indices"] = torch.tensor(profile.camera_ids, dtype=camera_indices.dtype, device=camera_indices.device)
selected["camera_names"] = list(profile.camera_names)
...
assert_task_input(selected, task)
```

## 逐行讲解 / What's happening

1. **第 21-30 行 / Lines 21-30**:
   - 中文: `InputProfile` 是 frozen dataclass，任务输入契约创建后不应被运行时代码改写。
   - English: `InputProfile` is a frozen dataclass, so the task input contract is not meant to be mutated at runtime.
2. **第 33-47 行 / Lines 33-47**:
   - 中文: VQA 和驾驶任务都用六相机四帧，但相机集合不同；trajectory/meta/auto-labeling/grounding 共享 driving profile。
   - English: VQA and driving tasks both use six cameras and four frames, but the camera set differs; trajectory, meta-action, auto-labeling, and grounding share the driving profile.
3. **第 110-163 行 / Lines 110-163**:
   - 中文: 选择逻辑先验证七相机源环，再用 `index_select` 同步切 camera、frame、timestamp。
   - English: Selection first validates the seven-camera source ring, then uses `index_select` to slice cameras, frames, and timestamps together.
4. **第 165-189 行 / Lines 165-189**:
   - 中文: `select_task_input` 最后调用 `assert_task_input`，防止切片结果和 task profile 漂移。
   - English: `select_task_input` finishes with `assert_task_input`, preventing selected payloads from drifting away from the task profile.

## 类比 / The analogy

中文：像拍电影前固定分镜表。哪个机位、哪几帧、按什么顺序进剪辑台，先写死；剪完再核对一次镜头清单。

English: It is like locking a shot list before editing a film. The camera positions, frames, and ordering are fixed first; after selection, the clip list is checked again.

## 自己跑一遍 / Try it yourself

```python
profiles = {
    "trajectory": {"cameras": (0, 1, 2, 3, 5, 6), "frames": (0, 1, 2, 3)},
    "vqa": {"cameras": (0, 1, 2, 3, 4, 5), "frames": (0, 1, 2, 3)},
}
source_cameras = [0, 1, 2, 3, 4, 5, 6]

def select(task):
    profile = profiles[task]
    positions = [source_cameras.index(camera) for camera in profile["cameras"]]
    return {"camera_positions": positions, "frames": list(profile["frames"])}

print(select("trajectory"))
print(select("vqa"))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'camera_positions': [0, 1, 2, 3, 5, 6], 'frames': [0, 1, 2, 3]}
{'camera_positions': [0, 1, 2, 3, 4, 5], 'frames': [0, 1, 2, 3]}
```

中文：任务输入 profile 是可复现实验的第一层基础设施。

English: A task input profile is the first infrastructure layer for reproducible evaluation.

## 注意事项 / Caveats / when it breaks

- **源相机顺序必须先验证** / **Validate source camera order first**: 源数据顺序错了，后面的 index_select 会稳定地产生错误输入。
- **时间戳要和图像一起切** / **Slice timestamps with images**: 只切图像不切时间会破坏运动对齐。
- **任务 profile 应版本化** / **Task profiles should be versioned**: 改相机或帧数会改变指标可比性。

## 延伸阅读 / Further reading

- [Alpamayo2 input_profiles.py](https://github.com/NVlabs/alpamayo2/blob/main/src/alpamayo2_super/input_profiles.py)
- [Alpamayo2 repository](https://github.com/NVlabs/alpamayo2)
