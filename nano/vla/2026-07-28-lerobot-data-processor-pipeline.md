---
date: 2026-07-28
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/processor/pipeline.py
permalink: https://github.com/huggingface/lerobot/blob/95211b98f1cd6b638bda84a8d28f9e41323229dd/src/lerobot/processor/pipeline.py#L253-L343
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, processor-pipeline, training-step]
build_role: training-step advanced variant, serializable observation/action processor chain
---

# LeRobot ProcessorPipeline：把 VLA 数据流做成链 / LeRobot ProcessorPipeline: Make the VLA Data Flow a Chain

> **一句话 / In one line**: `DataProcessorPipeline` 把 raw batch 先转成统一 transition，再按步骤变换，最后转回策略需要的输出。 / `DataProcessorPipeline` converts raw batches into a normalized transition, applies steps sequentially, and converts the result back to the policy-facing output.

## 为什么重要 / Why this matters

生产 VLA 的训练和推理不是一个 `forward()` 就结束：图像要改 key，动作要归一化，遥操作字段要桥接，调试时还要看中间状态。LeRobot 把这些处理变成显式 step 列表，让数据契约可以保存、加载、调试和复用。

A production VLA does not stop at one `forward()`: image keys are renamed, actions are normalized, teleop fields are bridged, and debugging needs intermediate states. LeRobot turns these transforms into an explicit step list, making the data contract saveable, loadable, debuggable, and reusable.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/processor/pipeline.py`](https://github.com/huggingface/lerobot/blob/95211b98f1cd6b638bda84a8d28f9e41323229dd/src/lerobot/processor/pipeline.py#L253-L343)

```python
@dataclass
class DataProcessorPipeline[TInput, TOutput](HubMixin):
    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "DataProcessorPipeline"

    to_transition: Callable[[TInput], EnvTransition] = field(
        default_factory=lambda: cast(Callable[[TInput], EnvTransition], batch_to_transition), repr=False
    )
    to_output: Callable[[EnvTransition], TOutput] = field(
        default_factory=lambda: cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        repr=False,
    )

    before_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)
    after_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)

    def __call__(self, data: TInput) -> TOutput:
        transition = self.to_transition(data)
        transformed_transition = self._forward(transition)
        return self.to_output(transformed_transition)

    def _forward(self, transition: EnvTransition) -> EnvTransition:
        for idx, processor_step in enumerate(self.steps):
            # Execute pre-hooks
            for hook in self.before_step_hooks:
                hook(idx, transition)

            transition = processor_step(transition)

            # Execute post-hooks
            for hook in self.after_step_hooks:
                hook(idx, transition)
        return transition

    def step_through(self, data: TInput) -> Iterable[EnvTransition]:
        transition = self.to_transition(data)

        # Yield the initial state before any processing.
        yield transition

        for processor_step in self.steps:
            transition = processor_step(transition)
            yield transition
```

## 逐行讲解 / What's happening

1. **第 270-279 行 / Lines 270-279 (`boundary converters`)**:
   - 中文: raw input 和 final output 不固定，中间统一成 `EnvTransition`。
   - English: Raw input and final output are flexible, while the middle is normalized as `EnvTransition`.
2. **第 281-282 行 / Lines 281-282 (`hooks`)**:
   - 中文: 每步前后都有 hook，适合记录 shape、断言 schema 或调试异常 batch。
   - English: Every step has before/after hooks, useful for logging shapes, asserting schema, or debugging bad batches.
3. **第 289-321 行 / Lines 289-321 (`forward chain`)**:
   - 中文: pipeline 的主逻辑只是转换、顺序执行、再转换回输出。
   - English: The main pipeline logic is only convert, run steps in order, and convert back to output.
4. **第 323-343 行 / Lines 323-343 (`step_through`)**:
   - 中文: 调试接口逐步 yield 中间 transition，不必给每个 processor 手写打印。
   - English: The debug API yields intermediate transitions step by step, avoiding ad hoc prints in every processor.

## 类比 / The analogy

这像工厂流水线：原料先放进统一料筐，每个工位只处理自己负责的一小步，质检员可以站在任意工位前后看状态。

It is like a factory line: raw material first enters a standard bin, each station performs one small transform, and inspectors can observe before or after any station.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

在 nanoVLA 里，这属于 `training-step` 和 `inference-loop` 之间的 processor 层。输入可以是 dataset batch、Gym observation 或 robot client message；输出是策略模型吃的标准字段，或机器人控制器吃的动作字段。省掉它会让数据清洗逻辑散落在训练脚本、policy 和 env wrapper 里，生产级还要补版本化配置、状态保存和迁移。

In nanoVLA this sits between `training-step` and `inference-loop` as the processor layer. Inputs may be dataset batches, Gym observations, or robot-client messages; outputs are standard policy fields or robot-controller action fields. Without it, data cleanup leaks into training scripts, policies, and env wrappers. A production version also needs versioned configs, state saving, and migration.

## 自己跑一遍 / Try it yourself

```python
class Pipeline:
    def __init__(self, steps, to_transition=dict, to_output=dict):
        self.steps = steps
        self.to_transition = to_transition
        self.to_output = to_output

    def __call__(self, data):
        x = self.to_transition(data)
        for step in self.steps:
            x = step(x)
        return self.to_output(x)

def rename_pixels(t):
    t["observation.image"] = t.pop("pixels")
    return t

def add_batch(t):
    t["batched"] = True
    return t

pipe = Pipeline([rename_pixels, add_batch])
print(pipe({"pixels": "frame0", "action": [0, 1]}))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'action': [0, 1], 'observation.image': 'frame0', 'batched': True}
```

这个最小版展示了关键思想：模型前处理是显式链，而不是散在各处的隐式副作用。

This minimal version shows the key idea: model preprocessing is an explicit chain, not implicit side effects scattered through the codebase.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi transform groups** / **openpi transform groups**: 中文: openpi 也把输入/输出变换组合成独立对象。 / English: openpi also composes input/output transforms as standalone objects.
- **OpenVLA processors** / **OpenVLA processors**: 中文: OpenVLA 通过 processor 把图像、prompt 和动作 token 组织进模型格式。 / English: OpenVLA uses processors to organize images, prompts, and action tokens into model format.

## 注意事项 / Caveats / when it breaks

- **顺序敏感** / **Order sensitive**: 中文: rename、normalize、batching 的顺序错了，schema 可能仍合法但语义错。 / English: If rename, normalize, and batching run in the wrong order, the schema may look valid while semantics are wrong.
- **状态 step** / **Stateful steps**: 中文: 有统计量的 processor 需要跟模型 checkpoint 一起保存。 / English: Processors with statistics must be saved together with the model checkpoint.

## 延伸阅读 / Further reading

- [huggingface/lerobot source](https://github.com/huggingface/lerobot/blob/95211b98f1cd6b638bda84a8d28f9e41323229dd/src/lerobot/processor/pipeline.py#L253-L343)
