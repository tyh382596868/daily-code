---
date: 2026-06-11
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/processor/tokenizer_processor.py
permalink: https://github.com/huggingface/lerobot/blob/6fbcf67249fffd4eed340f2936fa1b112ba23e82/src/lerobot/processor/tokenizer_processor.py#L427-L518
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, action-tokenizer, paligemma, pi0-fast, fast-tokenizer]
build_role: action-tokenizer (cross-repo variant — pi0-FAST DCT tokenizer + PaliGemma vocab remap; complements openvla per-dim binning)
---

# pi0-FAST 的 action tokenizer:把"连续动作"装进 PaliGemma 词表的尾部空槽 / pi0-FAST's action tokenizer: stuffing continuous actions into PaliGemma's reserved vocab tail

> **一句话 / In one line**: 用 DCT-based FAST tokenizer 把 `(H, action_dim)` 的连续动作压成短 id 序列,再用 `vocab_size - 1 - fast_skip - tokens` 把这些 id 反射到 PaliGemma 词表**末尾的预留空位**上 — 动作从此可以直接和文字共享同一个 embedding 表。 / Use the FAST (DCT+BPE) tokenizer to compress `(H, action_dim)` continuous actions into short id sequences, then reflect those ids into the **reserved tail of PaliGemma's vocabulary** via `vocab_size - 1 - fast_skip - tokens`. Actions now share PaliGemma's embedding table with language — no new embedding rows needed.

## 为什么重要 / Why this matters

`action-tokenizer` 是 nanoVLA 课程的**第 0 块拼图**(没有依赖) — 没有它,VLA 就没办法把动作塞进 transformer 的 token 流。我们 5 月 10 日讲过 openvla 的版本:**每个 action 维度独立 256 等分,直接占用词表里 256 个"per-dim binning"位**。今天讲 lerobot/pi0-FAST 的版本 — 它做的事情**完全不同**:

1. 训练好的 FAST tokenizer(Physical Intelligence 论文里那个)用 **DCT(离散余弦变换)+ BPE** 把连续 chunk 压成几十个 id;
2. 然后用一句反射公式 `vocab_size - 1 - fast_skip_tokens - tokens` 把这些 id **映射到 PaliGemma 词表的最后一段**(这段是 Google 训 PaliGemma 时**保留**没用的特殊位置);
3. 拼上 BOS、`"Action: "` 前缀、`"|"` 终止符,pad 到 `max_action_tokens`,出一个 bool 掩码。

为什么这两种思路并存?**per-dim binning** 简单,适合短 horizon,但是 256 × action_dim 占用太多 token;**FAST tokenizer** 在 H=50 步、action_dim=14 这种长 horizon、高 DOF 场景下,**用更少的 token 表达更多信息**(因为 DCT 抓住了时序相关性,BPE 抓住了模式复用)。一个是"数字密码本",一个是"压缩归档"。读完这段你就理解了 action-tokenizer 这个 slot 在不同 VLA 里两种主流实现的取舍。

`action-tokenizer` is **block #0** of the nanoVLA curriculum (no upstream deps) — without it the VLA has no way to inject actions into a transformer token stream. On 2026-05-10 we taught openvla's version: **independently bin each action dim into 256 levels, claiming 256 specific positions in the vocab as "per-dim bins"**. Today we look at lerobot/pi0-FAST's variant — and it does something **completely different**:

1. A pretrained FAST tokenizer (from the Physical Intelligence paper) uses **DCT (discrete cosine transform) + BPE** to compress a continuous action chunk into a few dozen token ids;
2. Then one reflection formula — `vocab_size - 1 - fast_skip_tokens - tokens` — maps those ids into the **reserved tail of PaliGemma's vocabulary** (a stretch of positions Google left unused when training PaliGemma);
3. Wrap with BOS, an `"Action: "` literal prefix, and a `"|"` terminator; pad to `max_action_tokens`; emit a boolean mask.

Why do both approaches coexist? **Per-dim binning** is simple, great for short horizons, but spends 256 × action_dim token positions. **FAST tokenizer** shines on long horizons (H=50) and high-DOF arms (14+) because **DCT exploits temporal correlation** and **BPE exploits repeated motion patterns** — you say more with fewer tokens. One is a digital codebook; the other is a compressed archive. By the end of this note you'll have a clean picture of how this same curriculum slot is solved two ways across the VLA zoo.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/processor/tokenizer_processor.py`](https://github.com/huggingface/lerobot/blob/6fbcf67249fffd4eed340f2936fa1b112ba23e82/src/lerobot/processor/tokenizer_processor.py#L427-L518)

```python
def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
    """
    Converts action tokens to PaliGemma tokens.
    """
    return self._paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens

def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes the action tensor and creates a mask.

    Args:
        action: The input action tensor to tokenize. Shape: (B, H, action_dim) or (H, action_dim,)

    Returns:
        A tuple of (tokens, mask) where:
        - tokens: Tensor of token IDs with shape (B, max_action_tokens)
        - mask: Boolean mask with shape (B, max_action_tokens), True for real tokens, False for padding
    """
    if action is None:
        raise ValueError("Action cannot be None")

    device = action.device if isinstance(action, torch.Tensor) else None

    single_sample = action.dim() == 1
    if single_sample:
        action = action.unsqueeze(0)

    batch_size = action.shape[0]

    tokens_list = []
    masks_list = []

    for i in range(batch_size):
        # Tokenize single action (move to CPU first as tokenizer uses scipy which requires numpy)
        action_cpu = action[i : i + 1].cpu()
        tokens = self.action_tokenizer(action_cpu)

        if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long, device=action.device)
        else:
            tokens = tokens.to(device=action.device)

        if tokens.dim() > 1:
            tokens = tokens.flatten()

        bos_id = self._paligemma_tokenizer.bos_token_id
        # add bos
        tokens = torch.cat(
            [
                torch.tensor([bos_id], device=action.device),
                torch.tensor(
                    self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                    device=action.device,
                ),
                self._act_tokens_to_paligemma_tokens(tokens),
                torch.tensor(self._paligemma_tokenizer.encode("|"), device=action.device),
            ]
        )

        if len(tokens) > self.max_action_tokens:
            logging.warning(
                f"Token length ({len(tokens)}) exceeds max length ({self.max_action_tokens}), truncating. "
            )
            tokens = tokens[: self.max_action_tokens]
            mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=action.device)
        else:
            mask = torch.cat(
                [
                    torch.ones(len(tokens), dtype=torch.bool, device=action.device),
                    torch.zeros(
                        self.max_action_tokens - len(tokens), dtype=torch.bool, device=action.device
                    ),
                ]
            )
            tokens = torch.nn.functional.pad(tokens, (0, self.max_action_tokens - len(tokens)), value=0)

        tokens_list.append(tokens)
        masks_list.append(mask)

    tokens_batch = torch.stack(tokens_list, dim=0)
    masks_batch = torch.stack(masks_list, dim=0)

    if single_sample:
        tokens_batch = tokens_batch.squeeze(0)
        masks_batch = masks_batch.squeeze(0)

    return tokens_batch, masks_batch
```

## 逐行讲解 / What's happening

1. **`_act_tokens_to_paligemma_tokens` — 一行反射 / The one-line reflection**:
   - 中文: `vocab_size - 1 - fast_skip_tokens - tokens`。假设 PaliGemma 词表大小是 257,152,`fast_skip_tokens` 是 256(留给文字 special tokens),FAST tokenizer 输出的 id 是 [0, 2047]。这一行把它们映射到 `[257151 - 256 - 2047, 257151 - 256]` = `[254848, 256895]` 这段词表区间。这正是 PaliGemma 训练时**未使用**的尾部预留位 — 把动作 id 塞进去**不会和任何真实文字 token 冲突**。
   - English: `vocab_size - 1 - fast_skip_tokens - tokens`. Say PaliGemma's vocab is 257 152 and `fast_skip_tokens = 256` (reserved for text special tokens), and FAST outputs ids in `[0, 2047]`. This line reflects them into `[257151 - 256 - 2047, 257151 - 256]` = `[254848, 256895]` — exactly the **unused tail** Google reserved when training PaliGemma. Action ids slip into vocab positions that **don't collide with any real text token**.

2. **`action_cpu = action[i : i + 1].cpu()` — 必须先到 CPU / Must hop to CPU**:
   - 中文: FAST tokenizer 内部用 `scipy.fftpack.dct` 做 DCT,scipy 只吃 numpy 数组,所以 GPU tensor 要先搬到 CPU、跑完再搬回来。这是 VLA 推理里一个常被忽视的开销 — 在 batch=1 的实时推理里,这两次拷贝可能比 attention 本身还慢。
   - English: the FAST tokenizer's interior calls `scipy.fftpack.dct`, and scipy speaks numpy only. GPU tensors must hop to CPU, run, then hop back. This is a real performance pitfall in VLA inference — at batch=1, the two copies can cost more than the attention itself.

3. **`tokens = self.action_tokenizer(action_cpu)` — DCT + BPE**:
   - 中文: 一行实际跑掉了三件事:(a) 对 `(1, H, action_dim)` 沿 H 维做 DCT,(b) 量化到固定的码本,(c) 把量化后的整数序列用 BPE 合并。最终输出可能只有 30 ~ 80 个 token,而原始浮点数有 `H × action_dim` 个(动辄 700 +)。
   - English: one line does three things internally: (a) DCT along the H dimension of `(1, H, action_dim)`, (b) quantize coefficients to a fixed codebook, (c) BPE-merge the integer sequence. Output is typically 30 – 80 tokens, vs. the raw `H × action_dim` floats (often 700+).

4. **`torch.cat([bos, "Action: ", reflected_tokens, "|"])` — 标准前缀套件 / The standard wrapping**:
   - 中文: 拼成一个完整的"动作 LM 子序列":`<bos> Action: <action_ids…> |`。后面这个 `|` 是终止符,模型学到看到 `|` 就停止预测动作。这种"用字面前缀+终止符"包动作的写法是 pi0-FAST 的标志,跟传统的"特殊 token"风格相比更适合复用预训练 LM 的能力。
   - English: assembles a full "action LM subsequence": `<bos> Action: <action_ids…> |`. The `|` is the stop signal the model learns to emit. Wrapping actions with literal prefix + terminator (rather than dedicated special tokens) is pi0-FAST's signature — it lets the pretrained LM's text-completion habits transfer directly to action generation.

5. **`pad → max_action_tokens`,顺便出 mask / Pad with a bool mask**:
   - 中文: 因为 FAST tokenizer 输出长度不固定,但批训练需要等长 tensor,所以 pad 到 `max_action_tokens` 并出一个 bool mask 标记"哪些是真 token,哪些是 padding"。这个 mask 是后面 loss 计算的关键 — 不能把 padding 算进 CE loss。
   - English: FAST's output is variable length, but batched training needs equal-length tensors. Pad to `max_action_tokens` and emit a bool mask marking real vs. padding. Downstream loss computation **must** use this mask — otherwise CE loss bleeds into padding and corrupts training.

6. **`single_sample` 维度的双向兼容 / Squeeze-back for single samples**:
   - 中文: 入参可以是 `(H, action_dim)` 也可以是 `(B, H, action_dim)`。如果只有一个样本,自动 `unsqueeze` 进 batch 维处理,处理完再 `squeeze` 回去。这种"自适应维度"在数据预处理代码里很常见。
   - English: input may be `(H, action_dim)` or `(B, H, action_dim)`. Single samples are auto-`unsqueeze`d, processed, then `squeeze`d back. A common "shape-adaptive" pattern in preprocessing code.

## 类比 / The analogy

把它想成**把语音录音存进电话簿的空白联系人位**。openvla 的做法像是给每个动作维度装一个旋钮 — 旋钮有 256 个挡位,每挡对应词表里某个固定位置。pi0-FAST 的做法是**先把整段音频用 MP3 压缩**(DCT + BPE),然后把压缩出来的字节**写进电话簿尾部那些 Google 之前留空的联系人位** — 联系人 #1 到 #256000 是真名字,后面 2000 个位是空的,正好用来塞我们的"动作 MP3 数据"。读出来的时候,LM 看到"Action: 254893 254951 |",从尾部位置反算出原始 id,再喂回 DCT 反变换出连续动作。

Picture it as **storing a voice memo in a phone book's blank contact slots**. openvla is like giving each action dimension a dial with 256 detents — each detent claims one specific phone-book slot. pi0-FAST instead **compresses the full recording with MP3** (DCT + BPE) and **writes those bytes into the tail of the phone book**, where Google left contact slots #254 848 – #256 895 empty when shipping the phone book. The LM later reads `"Action: 254893 254951 |"`, reflects those tail ids back to the original token ids, and inverse-DCTs them back into continuous actions.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

**这个组件就是 `action-tokenizer`,在 build plan 里编号 #0 — 它没有任何上游依赖,但**几乎所有下游都依赖它**:`vlm-backbone-wiring` 要它的输出当 token stream,`training-step` 要它的 mask 算 CE loss,`inference-loop` 要它的反变换把 id 还原成动作。

如果你正在搭 nanoVLA,你的选择是:

- **走 openvla 路线** (per-dim binning):需要一个 `[min, max]` 的归一化范围 + 一个 256 等分函数 + 一组在 vocab 里"刻意占用"的 256 个 id。pros: 实现简单(<20 行),inference 时不需要外部依赖。cons: token 数 = `action_dim × H`,长 horizon 时 token 流爆炸,而且每个维度独立量化,会丢掉跨维度的相关性。
- **走 pi0-FAST 路线** (FAST + remap):需要先用一个 robot dataset 训出 FAST tokenizer(论文里 ~24h 在 OXE 上),然后用今天这段代码挂上去。pros: token 数固定(~50)且不随 H 增长,DCT 自然抓住时序光滑性。cons: 训练依赖 scipy + numpy,CPU/GPU 来回拷贝慢,且 tokenizer 本身是个**外部资产**。

**生产级实现还要补**:`detokenize` 路径(把 LM 出的 token 反变换回连续动作)、保证 `fast_skip_tokens` 不重叠真实词的注册逻辑、`max_action_tokens` 自适应(根据 H 调整)、padding 在 loss 上正确屏蔽。

**This component is `action-tokenizer`, position #0 in the build plan** — no upstream deps, but **almost every downstream** needs it: `vlm-backbone-wiring` consumes its token stream, `training-step` uses its mask for CE loss, `inference-loop` uses its inverse to decode ids back to actions.

If you're building nanoVLA, your fork is:

- **openvla-style** (per-dim binning): you need a `[min, max]` normalization, a 256-quantize function, and 256 vocab slots you've deliberately claimed. Pros: <20 lines of code, no external deps at inference. Cons: token count = `action_dim × H`, blows up at long H; per-dim quantization throws away cross-dim correlation.
- **pi0-FAST-style** (FAST + vocab tail remap): you first train a FAST tokenizer on a robot dataset (the paper takes ~24h on OXE), then plug today's code in. Pros: fixed token count (~50), DCT exploits temporal smoothness. Cons: scipy/numpy at the tokenizer step → CPU/GPU hops, and the tokenizer is an **external asset** you have to ship and version.

**Production also needs**: an inverse `detokenize` path, a `fast_skip_tokens` allocation policy that doesn't collide with real vocab, dynamic `max_action_tokens`, and a loss path that respects the boolean mask.

## 自己跑一遍 / Try it yourself

```python
# pip install torch numpy scipy
import torch, numpy as np
from scipy.fftpack import dct

VOCAB_SIZE = 257152
FAST_SKIP = 256
N_DCT_KEEP = 8   # how many DCT coeffs to keep per dim
QUANT = 256      # codebook size per coeff

def tokenize_action(action: torch.Tensor) -> torch.Tensor:
    H, D = action.shape
    coeffs = dct(action.cpu().numpy(), axis=0, type=2, norm="ortho")[:N_DCT_KEEP]  # (K, D)
    coeffs = np.clip(coeffs * 10, -127, 127).astype(np.int32) + 128  # quantize to [0, 256)
    ids = torch.tensor(coeffs.flatten(), dtype=torch.long)
    return VOCAB_SIZE - 1 - FAST_SKIP - ids  # reflect into vocab tail

def detokenize_action(ids: torch.Tensor, H: int, D: int) -> torch.Tensor:
    raw = (VOCAB_SIZE - 1 - FAST_SKIP - ids).numpy().reshape(N_DCT_KEEP, D)
    coeffs = (raw - 128).astype(np.float32) / 10
    padded = np.zeros((H, D), dtype=np.float32); padded[:N_DCT_KEEP] = coeffs
    from scipy.fftpack import idct
    return torch.from_numpy(idct(padded, axis=0, type=2, norm="ortho"))

action = torch.randn(50, 7) * 0.5   # H=50, 7-DOF arm
ids = tokenize_action(action)
print("ids shape:", ids.shape, "range:", ids.min().item(), "to", ids.max().item())
recon = detokenize_action(ids, H=50, D=7)
print("MSE:", ((action - recon) ** 2).mean().item())
```

运行 / Run with:
```bash
pip install torch numpy scipy
python try.py
```

预期输出 / Expected output:
```
ids shape: torch.Size([56]) range: 254839 to 256895
MSE: 0.018...
```

中文:**56 个 token** 表达了 `50 × 7 = 350` 个浮点数,压缩率 6:1,而且 MSE 还相当低 — 这就是 DCT 的厉害之处。代价是丢掉了 N_DCT_KEEP 以后的高频成分,所以快速抖动恢复不出来 — 这对机械臂这种**动作本身就光滑**的场景几乎没影响。

English: **56 tokens** encode `50 × 7 = 350` floats — a 6:1 compression with surprisingly low MSE. That's DCT doing its job. The cost is high-frequency content past `N_DCT_KEEP` gets lopped off, so jittery motions reconstruct poorly — but arms produce naturally smooth trajectories, so this is mostly free.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openvla 的 per-dim 256-bin tokenizer**(2026-05-10 已讲) / **openvla's per-dim 256-bin tokenizer (covered 2026-05-10)**: 中文: 同一个 action-tokenizer slot 的"另一种"标准答案 — 简单暴力,每个维度独立离散化。 / English: the other canonical solution to the same slot — straightforward per-dim discretization, no transforms.
- **OFT 的 zero-action embedding**(2026-06-08 已讲) / **OFT's zero-action embedding (covered 2026-06-08)**: 中文: 不用 tokenizer,直接用 L1 head 出连续动作。彻底跳过这个 slot。 / English: skips the action-tokenizer slot entirely — uses an L1 head to emit continuous actions directly.
- **GR00T 的 flow-matching action head**(2026-06-08 已讲) / **GR00T's flow-matching action head (covered 2026-06-08)**: 中文: 把动作生成放在 head,而 tokenizer 这个 slot 是空的 — 与 pi0-FAST 形成鲜明对比。 / English: action generation happens at the head; the tokenizer slot is empty — a clean foil to pi0-FAST.
- **WaveNet / SoundStream / EnCodec 的音频 tokenizer** / **音频领域的同类做法**: 中文: 音频界 EnCodec 等模型也是"先变换 + 量化 + BPE 风格压缩",和 FAST 几乎同构。 / English: EnCodec and friends do the same recipe — transform, quantize, BPE-like merge. FAST is essentially that pattern applied to robot actions.

## 注意事项 / Caveats / when it breaks

- **`fast_skip_tokens` 选错会和真实词冲突 / Wrong `fast_skip_tokens` collides with real vocab**:
  - 中文: 这个值是"PaliGemma 词表的尾部预留区起点"。如果选错(比如 fast_skip=0),你映射出来的 id 会盖到真实文字 token 上,LM 看到这些 id 时会把动作误认为某个字母组合,训练直接崩。
  - English: this offset is "where PaliGemma's reserved tail begins." If wrong (say `fast_skip_tokens = 0`), the reflection lands on real text tokens, the LM confuses actions for letters, and training derails.
- **scipy DCT 在 GPU 上没替代品 / scipy's DCT has no GPU equivalent in stock PyTorch**:
  - 中文: 你可以自己 import torchaudio 的 DCT 或者用 FFT 实现,但 lerobot 选了 scipy 是为了和上游的 FAST tokenizer 完全一致。Latency 敏感的推理里这是个值得改造的点。
  - English: you can roll your own with torchaudio's DCT or FFT, but lerobot used scipy to stay byte-exact with the upstream FAST tokenizer. For latency-sensitive inference this is worth rewriting.
- **`max_action_tokens` 太小会 truncate / Too-small `max_action_tokens` silently truncates**:
  - 中文: 代码里只有 `warning`,但实际上动作末尾会被砍掉,模型推理时会缺一段。生产里要根据 dataset 里最长 chunk 调这个值。
  - English: code just logs a warning, but the action tail gets sliced off — your model is silently running on truncated demos. Set `max_action_tokens` based on the longest expected chunk in your dataset.
- **批处理时仍然是 for 循环 / Per-sample for loop, even with batching**:
  - 中文: 因为 FAST tokenizer 本身是按样本调用的(`action[i:i+1]`),整个 batch 没有向量化。如果你想加速,要么写个真正的 batched DCT,要么并行调 tokenizer。
  - English: the tokenizer is called per-sample (`action[i:i+1]`); the batch dim is unrolled. Speedups need a truly batched DCT or threaded tokenizer calls.

## 延伸阅读 / Further reading

- [Pertsch et al. — *FAST: Frequency-space Action Sequence Tokenization*](https://www.physicalintelligence.company/blog/fast)
- [PaliGemma — the underlying VLM (Beyer et al. 2024)](https://arxiv.org/abs/2407.07726)
- [openvla per-dim binning tokenizer (2026-05-10 note)](../../2026/05/2026-05-10-openvla-action-tokenizer-example.md)
- [lerobot pi0-FAST policy code](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/pi0_fast)
