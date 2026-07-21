---
date: 2026-07-05
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, data-transform, pipeline]
build_role: training-step advanced variant: observation/action transform pipeline
---

# openpi Transform Group：把输入和输出变换分成两段流水线 / openpi Transform Group: Split Input and Output Transforms into Two Pipelines

> **一句话 / In one line**: VLA policy 不是直接吃 raw dict，而是先跑输入变换、推理后再跑输出变换。 / A VLA policy does not consume raw dictionaries directly; it runs input transforms first and output transforms after inference.

## 为什么重要 / Why this matters

真实机器人数据里，图像、语言、状态、动作经常来自不同日志格式。openpi 用 `Group` 把一组输入 transform 和一组输出 transform 固定成同一个可调用对象，让训练、评估、部署都走同一套数据边界。

Real robot data mixes images, language, state, and actions from different log formats. openpi uses `Group` to freeze input transforms and output transforms into one callable object, so training, evaluation, and deployment share the same data boundary.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)

```python
@dataclasses.dataclass(frozen=True)
class Group(DataTransformFn):
    inputs: Sequence[DataTransformFn] = ()
    outputs: Sequence[DataTransformFn] = ()

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.inputs:
            data = transform(data)
        for transform in self.outputs:
            data = transform(data)
        return data
```

## 逐行讲解 / What's happening

1. **冻结 dataclass / Frozen dataclass**: 中文: transform group 一旦构造就不应被训练循环悄悄改掉，否则数据语义会漂。 / English: Once built, the transform group should not be silently mutated by the training loop, or data semantics drift.
2. **`inputs` 在前 / `inputs` first**: 中文: 图像 resize、prompt tokenize、状态归一化这类“喂给模型前”的处理排在这里。 / English: Image resizing, prompt tokenization, and state normalization happen here before the model sees data.
3. **`outputs` 在后 / `outputs` second**: 中文: action unnormalize、坐标系还原、chunk 后处理这类“模型输出后”的处理排在这里。 / English: Action unnormalization, frame conversion, and chunk postprocessing happen after model output.
4. **每一步都返回 dict / Every step returns a dict**: 中文: transform 之间只共享 `DataDict` 协议，组件可以自由组合。 / English: Transforms share only the `DataDict` protocol, so components remain composable.

## 类比 / The analogy

像机场安检和入境：上飞机前要安检，下飞机后要过海关。两个流程都处理同一个旅客，但职责和顺序完全不同。

It is like airport security and immigration. Before boarding you pass security; after landing you pass immigration. Both process the same traveler, but the responsibilities and order are different.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `training-step` 的外壳组件：上游是 dataset reader，下游是 policy forward 和 action head。你的 nanoVLA 至少需要 `obs -> model_inputs` 和 `model_actions -> robot_actions` 两段 transform；否则训练时的归一化和部署时的反归一化很容易不一致。

English: This is the shell around the `training-step` component: upstream is the dataset reader, downstream is policy forward and the action head. A nanoVLA needs at least `obs -> model_inputs` and `model_actions -> robot_actions`; otherwise training-time normalization and deployment-time denormalization diverge.

## 自己跑一遍 / Try it yourself

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Group:
    inputs: tuple
    outputs: tuple
    def __call__(self, data):
        for fn in self.inputs + self.outputs:
            data = fn(data)
        return data

g = Group(inputs=(lambda d: {**d, "x": d["x"] * 2},),
          outputs=(lambda d: {**d, "action": d["x"] + 1},))
print(g({"x": 3}))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'x': 6, 'action': 7}
```

中文: 这个例子把输入缩放和输出生成拆成两个阶段，形状虽小，控制流和 openpi 一样。

English: The example splits input scaling and output generation into two stages. The shape is tiny, but the control flow matches openpi.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot policies** / **LeRobot policies**: 中文: policy 通常把 observation normalization 和 action unnormalization 放在模型边界。 / English: Policies commonly put observation normalization and action unnormalization at the model boundary.
- **OpenVLA processors** / **OpenVLA processors**: 中文: 图像和语言 prompt 先被 processor 规范化，再进入 VLM。 / English: Images and language prompts are normalized by processors before entering the VLM.

## 注意事项 / Caveats / when it breaks

- **顺序就是语义 / Order is semantics**: 中文: resize 在 tokenize 前后无所谓，但 state normalize 和 delta-action 还原的顺序不能乱。 / English: Resize may be independent from tokenization, but state normalization and delta-action recovery are order-sensitive.
- **部署必须复用同一组 transform / Deployment must reuse the same transforms**: 中文: 训练和机器人端各写一套会制造隐蔽偏差。 / English: Separate training and robot-side transform code creates subtle drift.

## 延伸阅读 / Further reading

- openpi transform source linked above.
- openpi policy configuration examples.
