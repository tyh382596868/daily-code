---
date: 2026-10-14
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/policies/droid_policy.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py#L21-L74
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, observation-adapter, policy-contract]
build_role: vision-encoder advanced variant, observation schema adapter before the model boundary
---

# openpi DROID adapter：把机器人观测翻译成 policy 契约 / openpi DROID Adapter: Translate Robot Observations into the Policy Contract

> **一句话 / In one line**: openpi 在 policy 边界把 DROID 图像、关节和 gripper 状态整理成模型统一输入。 / openpi reshapes DROID images, joints, and gripper state into the model's shared policy input format.

## 为什么重要 / Why this matters

VLA 的主干模型喜欢稳定字段名、稳定图像布局和稳定 state 向量；真实机器人数据则常常来自不同相机、不同 gripper 表达和不同张量 layout。

A VLA backbone wants stable field names, image layouts, and state vectors. Real robot data often arrives with different cameras, gripper conventions, and tensor layouts.

## 代码 / The code

`Physical-Intelligence/openpi` - [`src/openpi/policies/droid_policy.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py#L21-L74)

```python
def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
```

## 逐行讲解 / What's happening

1. **第 21-27 行 / Lines 21-27 (`_parse_image`)**:
   - 中文: 浮点图像转回 `uint8`，`C,H,W` 转成模型期望的 `H,W,C`。
   - English: Floating-point images are converted back to `uint8`, and `C,H,W` becomes the model's expected `H,W,C`.
2. **第 35-45 行 / Lines 35-45 (state and cameras)**:
   - 中文: gripper 标量先变成一维，再和关节拼成统一 state；两路相机也在这里取出。
   - English: A scalar gripper value is made one-dimensional, concatenated with joints, and the two camera streams are extracted.
3. **第 47-58 行 / Lines 47-58 (model-specific slots)**:
   - 中文: `PI0` / `PI05` 和 `PI0_FAST` 的相机槽位不同，所以 adapter 在模型边界前补零图像和 mask。
   - English: `PI0` / `PI05` and `PI0_FAST` use different camera slots, so the adapter inserts zero images and masks before the model boundary.
4. **第 60-74 行 / Lines 60-74 (policy dict)**:
   - 中文: 输出固定成 `state`、`image`、`image_mask`，并可选携带动作和 prompt。
   - English: The output is normalized into `state`, `image`, and `image_mask`, with optional actions and prompt.

## 类比 / The analogy

这像给国际插头配转换头：墙里电压和设备接口都真实存在，但中间少一个小转换头，机器就插不上。

It is like using a travel plug adapter: the wall power and the device are both real, but without the small adapter in between, they cannot connect.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `vision-encoder` 前面的 observation adapter。它不负责提特征，而是把真实机器人的相机、状态和文本任务整理成 encoder 能稳定消费的契约；省掉它，后面每个模型模块都要理解 DROID 的原始字段名和 layout。

English: This is the observation adapter just before the `vision-encoder`. It does not extract features; it normalizes real robot cameras, state, and text into a contract the encoder can consume reliably. Without it, every downstream model module has to understand DROID's raw field names and layouts.

## 自己跑一遍 / Try it yourself

```python
def parse_image(x):
    if len(x) == 3:
        h, w = len(x[0]), len(x[0][0])
        return [[[x[c][i][j] for c in range(3)] for j in range(w)] for i in range(h)]
    return x

joint = list(range(7))
gripper = 0.5
state = joint + [gripper]
img = parse_image([[[0, 0], [0, 0]]] * 3)
print(len(state), len(img), len(img[0]), len(img[0][0]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
8 2 2 3
```

中文: 这个例子最值得看的是“模型输入契约”比原始数据格式更重要。

English: The important point is that the model input contract matters more than the raw data format.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot processors** / **LeRobot processors**: LeRobot 也把数据流拆成可组合 transforms。 / LeRobot also expresses the data path as composable transforms.
- **OpenVLA collators** / **OpenVLA collators**: OpenVLA 在 batch 边界整理语言、图像和动作标签。 / OpenVLA normalizes language, images, and action labels at the batch boundary.

## 注意事项 / Caveats / when it breaks

- **相机槽位是模型契约** / **Camera slots are model contracts**: 换模型时不能只换权重，还要换 adapter 映射。 / Changing models may require changing the adapter mapping, not only the weights.
- **float 图像范围要确认** / **Float image ranges must be checked**: 这里假设 float 在 `[0, 1]`，其他范围会被错误缩放。 / This assumes floats are in `[0, 1]`; other ranges will be scaled incorrectly.

## 延伸阅读 / Further reading

- [openpi DROID policy adapter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py#L21-L74)
