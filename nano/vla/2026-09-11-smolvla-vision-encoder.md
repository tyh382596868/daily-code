---
date: 2026-09-11
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/smolvla/smolvlm_with_expert.py
permalink: https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L152-L215
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, vision-encoder, image-embedding, frozen-backbone]
build_role: vision-encoder advanced variant
---

# SmolVLA 视觉入口：冻结塔、统一 dtype，再接 connector / SmolVLA Vision Boundary: Freeze, Cast, Then Connect

> **一句话 / In one line**: SmolVLA 把 vision encoder 的训练所有权、eval 行为、输入 dtype 和 modality connector 都集中在一个清晰边界里。 / SmolVLA makes gradient ownership, eval behavior, input dtype, and modality projection explicit at the vision boundary.

## 为什么重要 / Why this matters

中文：VLA 的视觉入口不只是“把图片喂给 backbone”。你还要决定视觉塔是否训练、它是否保持 eval、输入是否匹配权重 dtype，以及视觉 token 怎样进入语言或动作主干。SmolVLA 把这些决定分成 `set_requires_grad`、`train` 和 `embed_image` 三个小接口，便于微调和分布式训练时检查。

English: A VLA vision entrance is more than passing pixels into a backbone. You must define whether the vision tower trains, whether it stays in eval mode, which dtype the pixels use, and how visual tokens enter the language/action backbone. SmolVLA separates those concerns across `set_requires_grad`, `train`, and `embed_image`.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/smolvla/smolvlm_with_expert.py`](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L152-L215)

```python
    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.layers.{layer}.")

            unmatched_patterns = set(frozen_layers)
            for name, params in self.vlm.named_parameters():
                matched_patterns = [k for k in frozen_layers if k in name]
                if matched_patterns:
                    params.requires_grad = False
                    unmatched_patterns.difference_update(matched_patterns)
            if unmatched_patterns:
                raise RuntimeError(
                    "Some frozen layer patterns matched no VLM parameters, so the corresponding layers "
                    "would silently remain trainable (parameter naming may have changed in transformers): "
                    f"{sorted(unmatched_patterns)}"
                )
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states
```

## 逐行讲解 / What's happening

1. **第 153-156 行 / Lines 153-156**:
   - 中文: 冻结视觉塔时同时做 `eval()` 和 `requires_grad=False`；前者冻结行为，后者冻结参数更新。
   - English: Freezing the vision tower does both `eval()` and `requires_grad=False`: the first freezes behavior, the second freezes parameter updates.
2. **第 157-160 行 / Lines 157-160**:
   - 中文: `train_expert_only` 更强，它把整个 VLM 设为 eval 并关闭梯度，只让 expert 学习。
   - English: `train_expert_only` is stronger: the whole VLM goes to eval with gradients disabled, leaving the expert trainable.
3. **第 163-187 行 / Lines 163-187**:
   - 中文: 部分微调时用参数名 pattern 冻结指定层，并用 `unmatched_patterns` 防止 transformers 改名后静默失效。
   - English: For partial fine-tuning, parameter-name patterns freeze selected layers, while `unmatched_patterns` prevents a silent failure after a Transformers rename.
4. **第 193-200 行 / Lines 193-200**:
   - 中文: 重载 `train()` 是防御性措施，因为训练框架可能每个 step 都把整棵模型切回 train mode。
   - English: Overriding `train()` is defensive because the training framework may switch the whole model back to training mode every step.
5. **第 202-215 行 / Lines 202-215**:
   - 中文: 图片先进入 vision model 得到 `[B, L, D_vision]`，再经 connector/resampler 变成下游 backbone 能接收的视觉 token。
   - English: Images enter the vision model as pixels and produce `[B, L, D_vision]`; the connector/resampler turns them into tokens compatible with the downstream backbone.

## 类比 / The analogy

中文：像一条装配线上的相机工位。相机可以锁定曝光设置，采集出的照片先变成标准化零件，再交给后面的总装线；不能因为总装线开始训练，就让相机的规则也跟着乱变。

English: Think of a camera station on an assembly line. Its exposure settings may be locked, its images are converted into standardized parts, and only then do they enter the main assembly line. The camera should not change behavior just because the downstream line is training.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文：这是 `vision-encoder` 的 advanced variant，位于图像预处理之后、VLM/action expert 之前；它不依赖其他 curriculum 节点。输入是 `[B, C, H, W]` 像素，输出是连接器投影后的 `[B, L, D_model]` 视觉 token。省掉 connector 会造成维度不匹配，省掉 freeze contract 则会让微调时视觉特征漂移。生产版还要补上多相机拼接、patch mask、checkpoint compatibility、混合精度策略和视觉 token 数预算。

English: This is an advanced `vision-encoder` variant after image preprocessing and before the VLM/action expert; it has no curriculum dependency. It consumes `[B, C, H, W]` pixels and emits connector-projected `[B, L, D_model]` visual tokens. Without the connector, dimensions do not line up; without the freeze contract, visual features can drift during fine-tuning. A production implementation also needs multi-camera fusion, patch masks, checkpoint compatibility, mixed-precision policy, and a visual-token budget.

## 自己跑一遍 / Try it yourself

```python
class VisionBoundary:
    def __init__(self, width):
        self.width = width
        self.training = True
        self.frozen = False

    def freeze(self):
        self.training = False
        self.frozen = True

    def embed(self, pixels):
        tokens = [sum(pixel) / len(pixel) for pixel in pixels]
        return [[value] * self.width for value in tokens]

vision = VisionBoundary(width=3)
vision.freeze()
print(vision.training, vision.frozen, vision.embed([[1, 2], [3, 5]]))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
False True [[1.5, 1.5, 1.5], [4.0, 4.0, 4.0]]
```

中文：这个小例子把“冻结行为”和“输出宽度契约”放在同一个边界里；真实模型只是把平均值换成 patch encoder 和 connector。

English: The miniature combines frozen behavior with an output-width contract; the real model replaces the average with a patch encoder and connector.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **nanoVLM modality projector** / **nanoVLM modality projector**: 先压缩视觉 token 数，再扩展到语言模型通道宽度。 / Reduce visual token count first, then expand to the language-model width.
- **openpi image preprocessing** / **openpi image preprocessing**: 在视觉塔入口先固定 resize/pad 的几何契约。 / Preserve a geometric resize/pad contract before the vision tower.
- **GR00T frozen modules** / **GR00T frozen modules**: 训练框架把整模切到 train 后，再把冻结模块恢复为 eval。 / Restore frozen modules to eval after the trainer switches the whole model to train.

## 注意事项 / Caveats / when it breaks

- **`eval()` 不等于不求梯度** / **`eval()` is not the same as no gradients**: 两个开关要按训练目标分别设置。
- **dtype 必须跟权重一致** / **Dtype must match the weights**: 视觉塔用 bfloat16 时，输入 cast 不能遗漏。
- **名称匹配会随版本变化** / **Name matching can change across versions**: 任何 frozen pattern 都应该有 unmatched 校验和测试。

## 延伸阅读 / Further reading

- [LeRobot SmolVLMWithExpertModel](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
- [nanoVLM modality projector](https://github.com/huggingface/nanoVLM)
