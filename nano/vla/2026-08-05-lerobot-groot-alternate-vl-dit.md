---
date: 2026-08-05
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/action_head/cross_attention_dit.py
permalink: https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/groot/action_head/cross_attention_dit.py#L325-L387
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-head, cross-attention, dit]
build_role: action-head-continuous advanced variant, alternating image/text conditioning masks in a DiT action head
---

# LeRobot GROOT AlternateVLDiT：动作 head 轮流看文字和图像 / LeRobot GROOT AlternateVLDiT: Let the Action Head Alternate Text and Image Attention

> **一句话 / In one line**: `AlternateVLDiT` 用 `image_mask` 把 backbone token 分成图像和非图像 token，并让 cross-attention block 按层轮流看不同上下文。 / `AlternateVLDiT` splits backbone tokens into image and non-image tokens with `image_mask`, then lets cross-attention blocks alternate which context they attend to.

## 为什么重要 / Why this matters

VLA 的动作 head 不能只问一个混在一起的上下文：“我该怎么动？” 图像 token 提供几何和物体位置，文本 token 提供任务意图。GROOT 这个变体把两者显式分开，再按层交替注入，让动作 denoiser 有机会在不同深度分别对齐“看见什么”和“要做什么”。

A VLA action head should not ask one undifferentiated context, "what should I do?" Image tokens carry geometry and object locations; text tokens carry task intent. This GROOT variant separates them explicitly and alternates the conditioning by layer, giving the action denoiser separate chances to align what it sees with what it should do.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/action_head/cross_attention_dit.py`](https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/groot/action_head/cross_attention_dit.py#L325-L387)

```python
class AlternateVLDiT(DiT):
    """N1.7 DiT variant that alternates cross-attention over image and text tokens."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
    ):
        if image_mask is None:
            raise ValueError("image_mask is required for AlternateVLDiT.")
        if backbone_attention_mask is None:
            raise ValueError("backbone_attention_mask is required for AlternateVLDiT.")

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        all_hidden_states = [hidden_states]
        if not self.config.interleave_self_attention:
            raise ValueError("AlternateVLDiT requires interleave_self_attention=True.")

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                curr_encoder_attention_mask = (
                    non_image_attention_mask
                    if idx % (2 * self.attend_text_every_n_blocks) == 0
                    else image_attention_mask
                )
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)
```

## 逐行讲解 / What's happening

1. **第 342-345 行 / Lines 342-345 (required masks)**:
   - 中文: 这个模块拒绝猜测 token 类型；没有 `image_mask` 和 backbone mask 就直接报错。
   - English: The module refuses to infer token types implicitly; missing `image_mask` or backbone mask is an error.
2. **第 347-352 行 / Lines 347-352 (split context)**:
   - 中文: timestep embedding 给 denoiser 当前扩散步条件；`image_mask & backbone_attention_mask` 得到图像上下文，取反后得到文本/非图像上下文。
   - English: The timestep embedding conditions the denoiser on the diffusion step; `image_mask & backbone_attention_mask` selects image context, while the inverse selects text or non-image context.
3. **第 358-379 行 / Lines 358-379 (alternate blocks)**:
   - 中文: 奇数层只做 action token self-attention；偶数层做 cross-attention，但按 `attend_text_every_n_blocks` 在文字和图像上下文之间切换。
   - English: Odd layers run self-attention over action tokens; even layers run cross-attention, switching between text and image context according to `attend_text_every_n_blocks`.
4. **第 382-387 行 / Lines 382-387 (conditioned output)**:
   - 中文: 最后用 timestep embedding 生成 shift/scale 调制输出，再投影成动作维度。
   - English: The final timestep embedding produces shift/scale modulation before projecting back to action dimensions.

## 类比 / The analogy

这像装家具时交替看两张纸：一张是房间照片，告诉你桌子在哪里；另一张是说明书，告诉你要拧哪颗螺丝。一直只看其中一张都不够。

It is like assembling furniture while alternating between two sheets: a room photo tells you where the table is, and the manual tells you which screw to tighten. Looking at only one sheet is not enough.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-head-continuous` 的 advanced variant。上游是 VLM backbone 输出的混合 image/text token、action latent 和 diffusion timestep；中间层是一个 DiT action denoiser；下游是连续动作预测。如果省掉这种显式分流，最小版 nanoVLA 仍能工作，但图像几何和语言意图会在同一张 attention mask 里混成一团。生产级实现还要处理 padding、多相机 token、语言 token 类型、batch 内不同 prompt 长度。

This is an advanced variant of `action-head-continuous`. Upstream is the VLM backbone's mixed image/text tokens, action latents, and diffusion timestep; the middle is a DiT action denoiser; downstream is continuous action prediction. A minimal nanoVLA can work without explicit alternation, but visual geometry and language intent then share one undifferentiated attention mask. A production version also needs padding, multi-camera tokens, token-type metadata, and variable prompt lengths within a batch.

## 自己跑一遍 / Try it yourself

```python
image_mask = [True, True, False, False, True]
backbone_mask = [True, True, True, False, True]
image_ctx = [i for i, ok in enumerate(image_mask) if ok and backbone_mask[i]]
text_ctx = [i for i, ok in enumerate(image_mask) if (not ok) and backbone_mask[i]]

for idx in range(6):
    if idx % 2 == 1:
        print(idx, "self-attn")
    else:
        ctx = text_ctx if idx % 4 == 0 else image_ctx
        print(idx, "cross-attn", ctx)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 cross-attn [2]
1 self-attn
2 cross-attn [0, 1, 4]
3 self-attn
4 cross-attn [2]
5 self-attn
```

中文: 交替发生在层级上，不是把 image/text token 先合成一个平均向量。

English: Alternation happens at the layer level; image and text tokens are not averaged into one context vector.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi pi0 suffix action head** / **openpi pi0 suffix action head**: 也把 action token 作为后缀接入 VLM 表示，只是条件注入方式不同。 / It also attaches action tokens to VLM representations, but conditions them differently.
- **LeRobot pi0 denoise loop** / **LeRobot pi0 denoise loop**: 同一 curriculum slot 下，连续动作 head 也可以通过迭代流匹配产生动作。 / In the same curriculum slot, a continuous action head can also produce actions through iterative flow matching.

## 注意事项 / Caveats / when it breaks

- **mask 语义必须稳定 / Mask semantics must be stable**: `image_mask=True` 到底是哪类 token，必须由 backbone 一致提供。 / The backbone must consistently define what `image_mask=True` means.
- **交替频率是超参 / Alternation frequency is a hyperparameter**: 文字看太少会丢任务，图像看太少会丢几何。 / Too little text loses intent; too little image context loses geometry.

## 延伸阅读 / Further reading

- LeRobot `AlternateVLDiT`: https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/groot/action_head/cross_attention_dit.py#L325-L387
