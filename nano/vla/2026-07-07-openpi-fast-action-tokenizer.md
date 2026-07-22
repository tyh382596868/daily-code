---
date: 2026-07-07
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models/tokenizer.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py#L51-L141
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer]
build_role: action-tokenizer advanced variant
---

# openpi FASTTokenizer：把连续动作塞进语言模型词表尾部 / openpi FASTTokenizer: Put Continuous Actions into the Tail of a Language-Model Vocabulary

> **一句话 / In one line**: `FASTTokenizer` 先用专用 action tokenizer 离散化动作，再把动作 token 映射到 PaliGemma 词表尾部的保留区。 / `FASTTokenizer` discretizes actions with a dedicated action tokenizer, then maps action tokens into reserved slots near the end of the PaliGemma vocabulary.

## 为什么重要 / Why this matters

有些 VLA 不让动作头直接回归连续值，而是把动作当成语言模型的下一个 token 来预测。这样可以复用自回归解码、loss mask 和 prefix-LM attention mask。openpi 的实现把 prompt、state 和 action token 放到同一条 token 序列里，是 pi0-FAST 这类离散动作路线的核心。

Some VLAs do not regress continuous actions directly; they predict action tokens as if actions were language tokens. That reuses autoregressive decoding, loss masks, and prefix-LM attention masks. openpi places prompt, state, and action tokens in one sequence, which is central to the pi0-FAST discrete-action path.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models/tokenizer.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py#L51-L141)

```python
class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128

    def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None):
        cleaned_text = prompt.lower().strip().replace("_", " ")
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode([action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim)[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
```

## 逐行讲解 / What's happening

1. **第 3-8 行 / Lines 3-8 (two tokenizers)**:
   - 中文: PaliGemma tokenizer 负责文字；FAST tokenizer 负责连续动作。
   - English: the PaliGemma tokenizer handles text; the FAST tokenizer handles continuous actions.
2. **第 11-15 行 / Lines 11-15 (state as text prefix)**:
   - 中文: state 先按 `[-1, 1]` 分成 256 桶，再写进 prompt 前缀。
   - English: state is binned into 256 values over `[-1, 1]`, then written into the prompt prefix.
3. **第 18-24 行 / Lines 18-24 (action postfix)**:
   - 中文: action token 被映射进 PaliGemma 词表，再夹在 `Action:` 和 `|` 之间。
   - English: action tokens are mapped into the PaliGemma vocabulary and wrapped between `Action:` and `|`.
4. **第 30-32 行 / Lines 30-32 (masks)**:
   - 中文: prefix 双向看，postfix 自回归；loss 只算动作 postfix。
   - English: the prefix uses bidirectional attention, the postfix is autoregressive, and loss applies only to the action postfix.
5. **第 35-44 行 / Lines 35-44 (decode back)**:
   - 中文: 推理时从模型输出文本里找 `Action:`，再反向映射回 FAST action token。
   - English: inference searches decoded text for `Action:`, then maps back to FAST action tokens.

## 类比 / The analogy

像把机器人动作翻译成一串“专用暗号”，再把暗号贴到一本语言词典最后几页。语言模型只要学会翻页和续写，就能输出动作。

It is like translating robot motion into a sequence of private codes, then placing those codes in the last pages of a language dictionary. The language model only needs to keep reading and writing tokens.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这属于 `action-tokenizer` 组件，位于数据预处理和自回归 VLM 训练目标之间。上游给它 prompt、state、连续 action chunk；下游拿到 `tokens/token_mask/ar_mask/loss_mask` 训练语言模型。如果省掉它，你就要改用连续动作头和 MSE/flow-matching loss。

In nanoVLA, this is the `action-tokenizer` component between data preprocessing and the autoregressive VLM objective. Upstream provides prompt, state, and continuous action chunks; downstream receives `tokens/token_mask/ar_mask/loss_mask` for LM training. If you omit it, you need a continuous action head and an MSE or flow-matching loss instead.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
vocab, skip = 1000, 128
action_tokens = np.array([0, 5, 127])
pg_tokens = vocab - 1 - skip - action_tokens
round_trip = vocab - 1 - skip - pg_tokens
print(pg_tokens.tolist())
print(round_trip.tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[871, 866, 744]
[0, 5, 127]
```

这个映射是可逆的：只要词表尾部区间不被普通文本使用，动作 token 就能安全进出语言模型。

The mapping is reversible: as long as that tail vocabulary range is reserved, action tokens can safely enter and leave the language model.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot pi0-FAST** / **LeRobot pi0-FAST**: 同样把动作装进语言 token 流。 / It also packs actions into a language-token stream.
- **OpenVLA action tokenizer** / **OpenVLA action tokenizer**: 也把连续动作离散成 token，但常用固定 bins。 / It also discretizes continuous actions into tokens, often with fixed bins.

## 注意事项 / Caveats / when it breaks

- **词表尾部必须保留** / **The vocabulary tail must be reserved**: 普通文本如果使用同一区间，会和动作 token 冲突。 / If normal text uses the same range, it collides with action tokens.
- **离散化有精度损失** / **Discretization loses precision**: 高精度控制可能需要更多 token 或连续 head。 / High-precision control may need more tokens or a continuous head.

## 延伸阅读 / Further reading

- [openpi tokenizer.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py)
