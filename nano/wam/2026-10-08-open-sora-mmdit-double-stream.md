---
date: 2026-10-08
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L174-L253
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, dit-block]
build_role: dit-block advanced variant, double-stream image/text attention with modulation gates
---

# Open-Sora MM-DiT：图像流和文本流分开调制，一起注意力 / Open-Sora MM-DiT: Modulate Image and Text Separately, Attend Together

> **一句话 / In one line**: 双流 block 先分别调制 image/text token，再拼到一次 attention 里交互，最后用 gate 控制残差写回。 / The double-stream block modulates image and text tokens separately, concatenates them into one attention call, then gates each residual update.

## 为什么重要 / Why this matters

World Action Model 里的视频 latent 和语言/动作条件不是同一种 token。Open-Sora 这个 MM-DiT block 保留两条流各自的 norm、QKV、MLP 和 modulation，但在 attention 中让它们同场交互。这样既保住模态差异，又能让文本条件直接影响视频 token。

Video latents and language/action conditions in a World Action Model are not the same kind of token. This Open-Sora MM-DiT block keeps separate norms, QKV projections, MLPs, and modulation for each stream, but lets them interact inside one attention call. It preserves modality-specific processing while allowing text conditions to shape video tokens.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L174-L253)

```python
@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        # attn is the DoubleStreamBlock;
        # process img and txt separately while both is influenced by text vec

        # vec will interact with image latent and text context
        img_mod1, img_mod2 = attn.img_mod(vec)  # get shift, scale, gate for each mod
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)

        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm as in SD3 paper
        if not attn.img_attn.fused_qkv:
            img_q = rearrange(img_q, "B L H D -> B H L D")
            img_k = rearrange(img_k, "B L H D -> B H L D")
            img_v = rearrange(img_v, "B L H D -> B H L D")

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            txt_q = rearrange(attn.txt_attn.q_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        if not attn.txt_attn.fused_qkv:
            txt_q = rearrange(txt_q, "B L H D -> B H L D")
            txt_k = rearrange(txt_k, "B L H D -> B H L D")
            txt_v = rearrange(txt_v, "B L H D -> B H L D")

        # run actual attention, image and text attention are calculated together by concat different attn heads
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
```

## 逐行讲解 / What's happening

1. **第 179-192 行 / Lines 179-192**:
   - 中文: `Modulation` 把条件向量 `vec` 切成 shift、scale、gate；`double=True` 时同时产出 attention 和 MLP 两套调制。
   - English: `Modulation` slices the conditioning vector `vec` into shift, scale, and gate; with `double=True`, it produces separate modulation for attention and MLP.
1. **第 201-232 行 / Lines 201-232**:
   - 中文: image 和 text 各自做 norm、scale/shift、QKV projection、QK norm；两条流还没混，但都被同一个条件向量影响。
   - English: Image and text each run their own norm, scale/shift, QKV projection, and QK norm. They are not mixed yet, but both are influenced by the same conditioning vector.
1. **第 238-244 行 / Lines 238-244**:
   - 中文: 真正的交互发生在这里：沿 token 维拼接 text/image 的 QKV，跑一次 attention，再按原长度切回来。
   - English: The real interaction happens here: concatenate text/image QKV along the token dimension, run one attention call, then split by the original lengths.
1. **第 246-253 行 / Lines 246-253**:
   - 中文: 每条流用自己的 gate 写回 attention 残差和 MLP 残差，条件可以控制“写多少”。
   - English: Each stream writes back attention and MLP residuals through its own gates, so the condition controls how much update is applied.

## 类比 / The analogy

像双语会议：中文组和英文组各自先整理材料，然后进同一个同传会议室交流，出来后再各自更新自己的纪要。

It is like a bilingual meeting. The Chinese and English teams prepare separately, enter one interpretation room to exchange information, then return to update their own notes.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `dit-block` 的高级形态。nanoWAM 最小版可以把视频 latent、动作 token、文本 token 直接拼成一个序列；生产版更常见的是双流或多流 block：每种模态有自己的 normalization/projection，同时在 attention 层共享信息。这样以后加动作流、状态流、接触流时更容易扩展。

This is an advanced `dit-block`. A minimal nanoWAM can concatenate video latents, action tokens, and text tokens into one sequence. A production version often benefits from double- or multi-stream blocks: each modality keeps its own normalization/projection while attention shares information. That makes future action, state, or contact streams easier to add.

## 自己跑一遍 / Try it yourself

```python
def modulate(x, scale, shift):
    return [(1 + scale) * v + shift for v in x]

img = [1.0, 2.0]
txt = [10.0]
img_m = modulate(img, 0.1, -0.2)
txt_m = modulate(txt, -0.1, 0.5)
mixed_mean = sum(txt_m + img_m) / len(txt_m + img_m)
img = [round(v + 0.2 * mixed_mean, 3) for v in img]
txt = [round(v + 0.1 * mixed_mean, 3) for v in txt]
print(img)
print(txt)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1.827, 2.827]
[10.413]
```

两条流各自调制，但更新来自同一个混合注意力统计；这就是 double-stream 的核心味道。

The two streams are modulated separately, but their updates come from one mixed attention statistic; that is the core feel of double-stream blocks.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion 3 MMDiT** / **Stable Diffusion 3 MMDiT**: image/text 双流调制和联合 attention 是同一类设计。 / Image/text double-stream modulation and joint attention are the same design family.
- **VLA action heads** / **VLA action heads**: action tokens 和 language/vision tokens 也常常分流处理、在 attention 里交互。 / Action tokens and language/vision tokens are also often processed in separate streams and mixed by attention.

## 注意事项 / Caveats / when it breaks

- **token 维度切分必须准确** / **Token splits must be exact**: `txt_q.shape[2]` 决定切回 text/image 的边界，错了就会把模态写乱。 / `txt_q.shape[2]` defines the text/image split; if it is wrong, modalities are written back incorrectly.
- **gate 初始化很关键** / **Gate initialization matters**: gate 太大容易破坏预训练流，太小又会让条件进不来。 / Large gates can disrupt pretrained streams; tiny gates can prevent conditioning from entering.

## 延伸阅读 / Further reading

- [Open-Sora repository](https://github.com/hpcaitech/Open-Sora)
- [Black Forest Labs FLUX architecture family](https://github.com/black-forest-labs/flux)
