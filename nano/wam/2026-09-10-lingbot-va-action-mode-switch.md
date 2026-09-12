---
date: 2026-09-10
topic: wam
source: wam
repo: Robbyant/lingbot-va
file: wan_va/modules/model.py
permalink: https://github.com/Robbyant/lingbot-va/blob/main/wan_va/modules/model.py#L767-L818
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, action-stream, projection]
build_role: action-conditioning advanced variant
---

# LingBot-VA action mode：同一主干切换视频和动作 / LingBot-VA Action Mode: One Backbone Switches Between Video and Action

> **一句话 / In one line**: `action_mode` 让 LingBot-VA 在同一个 transformer forward 里切换 action embedding/projection 和 video patch embedding/projection。 / `action_mode` lets LingBot-VA switch between action embedding/projection and video patch embedding/projection inside the same transformer forward.

## 为什么重要 / Why this matters

中文：WAM 不只预测下一帧，也要生成或解释动作。LingBot-VA 没有为 action 复制完整主干，而是在入口和出口切换：动作走 `action_embedder` 和 `action_proj_out`，视频走 patch MLP 和 `proj_out`；中间共享文本条件、RoPE、time embedding 和 transformer blocks。这把“视频”和“动作”做成同一时序建模问题。

English: A WAM predicts more than future pixels; it must generate or interpret actions. LingBot-VA does not duplicate the full backbone for actions. It switches the input and output adapters: actions use `action_embedder` and `action_proj_out`, while video uses patch MLP and `proj_out`; the middle shares text conditioning, RoPE, time embedding, and transformer blocks. Video and action become one temporal modeling problem.

## 代码 / The code

`Robbyant/lingbot-va` — [`wan_va/modules/model.py`](https://github.com/Robbyant/lingbot-va/blob/main/wan_va/modules/model.py#L767-L818)

```python
if action_mode:  # action input emb
    latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                     'b c f h w -> b (f h w) c')
    latent_hidden_states = self.action_embedder(
        latent_hidden_states)  # B L1 C
else:  # latent input emb
    latent_hidden_states = rearrange(
        input_dict['noisy_latents'],
        'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
        p1=self.patch_size[0],
        p2=self.patch_size[1],
        p3=self.patch_size[2])
    latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)

...

if action_mode:
    latent_hidden_states = self.action_proj_out(latent_hidden_states)
else:
    latent_hidden_states = self.proj_out(latent_hidden_states)
    latent_hidden_states = rearrange(latent_hidden_states,
                                     'b l (n c) -> b (l n) c',
                                     n=math.prod(self.patch_size))
```

## 逐行讲解 / What's happening

1. **第 767-771 行 / Lines 767-771**:
   - 中文: action 输入已经是每个 action token 的向量，只需展平成序列，再投影到 transformer hidden size。
   - English: Action input is already a vector per action token, so it is flattened into a sequence and projected to transformer hidden size.
2. **第 772-780 行 / Lines 772-780**:
   - 中文: 视频 latent 需要按 3D patch 重排，把局部时空块拼成 token 后再投影。
   - English: Video latents need 3D patch rearrangement, turning local spacetime blocks into tokens before projection.
3. **第 781-801 行 / Lines 781-801**:
   - 中文: 两种模式都接同一套 text embedding、RoPE、time embedding 和 block 堆栈。
   - English: Both modes then use the same text embedding, RoPE, time embedding, and transformer block stack.
4. **第 810-816 行 / Lines 810-816**:
   - 中文: 输出端再分叉：action 直接投回 action dim，视频则投回 patch 像素/latent 并 unpatchify。
   - English: The output side branches again: actions project directly to action dimension, while video projects to patch channels and is unpatchified.

## 类比 / The analogy

中文：像同一台翻译机换两个接口。输入是视频时先切成胶片格，输入是动作时先切成控制格；中间的语法模型相同，最后再翻回各自格式。

English: It is like one translator with two adapters. Video input is sliced into film cells, action input into control cells; the grammar model in the middle is shared, and the output adapter converts back to the right format.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文：这是 `action-conditioning` 的 advanced variant，位于 tokenizer/latent encoder 之后、DiT block 之前和 output head 之后。上游给 noisy video latent 或 noisy action latent；中间统一成 `[B, L, C]` token；下游根据 mode 输出 video patch 或 action vector。生产版 nanoWAM 可以用这个设计共享大部分 backbone，但要把 action/video 的 shape schema、time grid、loss mask 和采样循环状态分开管理。

English: This is an advanced `action-conditioning` variant after tokenization/latent encoding, before the DiT blocks, and at the output head. Upstream supplies noisy video latents or noisy action latents; the middle normalizes them to `[B, L, C]` tokens; downstream emits either video patches or action vectors. A production nanoWAM can share most of the backbone this way, but should keep action/video shape schemas, time grids, loss masks, and sampling state separate.

## 自己跑一遍 / Try it yourself

```python
def embed(x, action_mode):
    if action_mode:
        return [f"action-token:{v}" for v in x]
    return [f"video-patch:{v}" for v in x]

def project(tokens, action_mode):
    prefix = "action" if action_mode else "video"
    return [token.replace("token", prefix).replace("patch", prefix) for token in tokens]

for mode in (True, False):
    tokens = embed(["a", "b"], action_mode=mode)
    print(mode, tokens, "->", project(tokens, action_mode=mode))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
True ['action-token:a', 'action-token:b'] -> ['action-action:a', 'action-action:b']
False ['video-patch:a', 'video-patch:b'] -> ['video-video:a', 'video-video:b']
```

中文：共享主干前，必须先把不同模态变成同一宽度的 token。

English: Before sharing the backbone, every modality must become tokens of the same width.

## 注意事项 / Caveats / when it breaks

- **action/video 的 spatial 假设不同** / **Action and video have different spatial assumptions**: action mode 不应误用 patch size 做空间 unpatchify。
- **time embedding 仍要按 token 数扩展** / **Time embedding still follows token count**: repeat 数错了会让条件和 token 错位。
- **共享 block 不代表共享 loss** / **Shared blocks do not imply shared loss**: video denoising loss 和 action loss 应分别聚合。

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM MoT** / **FastWAM MoT**: action expert 和 video expert 在 attention 里通信。
- **GR00T action head** / **GR00T action head**: state/action tokens 被投影后送进 DiT。
- **Open-Sora patch embed** / **Open-Sora patch embed**: 视频先变 token，再由 transformer 处理。

## 延伸阅读 / Further reading

- [LingBot-VA model.py](https://github.com/Robbyant/lingbot-va/blob/main/wan_va/modules/model.py)
- [LingBot-VA repository](https://github.com/Robbyant/lingbot-va)
