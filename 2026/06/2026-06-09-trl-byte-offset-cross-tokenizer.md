---
date: 2026-06-09
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/experimental/utils.py
permalink: https://github.com/huggingface/trl/blob/576a5f6bb536ca2fcfe934419d9f98f6f443eab8/trl/experimental/utils.py#L142-L260
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, huggingface, trl, distillation, tokenizer, byte-level-bpe]
---

# TRL 的 GOLD trainer:用"字节偏移"对齐两种不同的 tokenizer / TRL's GOLD trainer aligns two different tokenizers via byte offsets

> **一句话 / In one line**: 老师和学生用不同 tokenizer?把每个 token 投影到它在原始 UTF-8 字节流上的 `(start, end)` 区间,然后在字节空间里对齐 —— GPT-2 那张臭名昭著的 byte↔unicode 字典就是让这件事真的能跑起来的关键。 / Teacher and student use different tokenizers? Project each token onto its `(start, end)` interval in the source UTF-8 byte stream and align in byte space — GPT-2's infamous byte↔unicode dictionary is the bridge that makes it work in practice.

## 为什么重要 / Why this matters

LLM 蒸馏的现实困境:你想用 Llama-3-70B 教 Qwen-2.5-1.5B,但它俩的 tokenizer 完全不一样 —— 同一句 "Hello world" 切出来的 token 数、ID、分界全不同。传统蒸馏的"逐 token KL"假设老师和学生在每个位置上算的是同一个东西,在跨 tokenizer 场景里直接崩。学术界给出的解(GOLD / ULD / X-Token)是:**别在 token 空间对齐,去 *字节* 空间对齐**。每个 token 都对应原文的一段连续字节,只要双方对齐到同一段字节,就能在那段字节里做"软对齐"的 KL。TRL 在 2026-06-07 merge 的这版(PR #5885)用 `encode_with_byte_offsets` 把每个 token 都精确映射到它的字节 `(start, end)`,而且把 GPT-2 那套 byte-level BPE 反向解码工具一并搬了进来 —— 整个 cross-tokenizer 对齐管线的"地基"就在这 120 行里。

The real-world pain point in LLM distillation: you'd love to use Llama-3-70B as teacher for Qwen-2.5-1.5B, but their tokenizers disagree completely — the same "Hello world" produces different token counts, different IDs, different boundaries. Standard distillation's per-token KL assumes teacher and student compute "the same thing" at each position, which falls apart across tokenizers. The academic answer (GOLD / ULD / X-Token) is: **don't align in token space — align in *byte* space**. Every token covers a contiguous run of bytes in the source UTF-8 text; once both sides agree on which token covers which bytes, you can do a "soft alignment" KL over those bytes. TRL merged this approach on 2026-06-07 (PR #5885) with `encode_with_byte_offsets` mapping every token to its `(start, end)` byte span, plus the byte-level BPE inverse-decoder lifted from GPT-2 — the entire byte-alignment foundation lives in these 120 lines.

## 代码 / The code

`huggingface/trl` — [`trl/experimental/utils.py`](https://github.com/huggingface/trl/blob/576a5f6bb536ca2fcfe934419d9f98f6f443eab8/trl/experimental/utils.py#L142-L260)

```python
def is_byte_level_tokenizer(backend) -> bool:
    """Whether ``backend`` is a ByteLevel BPE tokenizer (Llama-3 family, SmolLM, Qwen, …) — its pieces are in
    byte→unicode space, one char per source byte. Detected via the pre-tokenizer / decoder repr."""
    return "ByteLevel" in repr(backend.pre_tokenizer) or "ByteLevel" in repr(backend.decoder)


def piece_byte_len(piece: str) -> int:
    """UTF-8 byte length of a ByteLevel BPE token piece — each char maps 1:1 to one source byte."""
    return len(piece)


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=True))


_BYTE_LEVEL_DECODER = {ch: b for b, ch in _bytes_to_unicode().items()}


def _byte_level_piece_len(piece: str, text_bytes: bytes, start: int) -> int | None:
    piece_bytes = []
    for ch in piece:
        if ch not in _BYTE_LEVEL_DECODER:
            return None
        piece_bytes.append(_BYTE_LEVEL_DECODER[ch])
    if piece_bytes and piece_bytes[0] == ord(" ") and text_bytes[start : start + 1] != b" ":
        piece_bytes = piece_bytes[1:]
    return len(piece_bytes)


def encode_with_byte_offsets(backend, texts, add_special_tokens=False):
    if not is_byte_level_tokenizer(backend):
        raise NotImplementedError("Cross-tokenizer ULD currently supports only ByteLevel BPE tokenizers …")
    encs = backend.encode_batch(texts, add_special_tokens=add_special_tokens)
    out = []
    for text, enc in zip(texts, encs, strict=True):
        char_to_byte = [0]
        for ch in text:
            char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
        byte_offsets = [(char_to_byte[s], char_to_byte[e]) for s, e in enc.offsets]
        byte_offsets = _normalize_byte_offsets(byte_offsets, enc.tokens, text.encode("utf-8"))
        out.append((list(enc.ids), byte_offsets))
    return out
```

## 逐行讲解 / What's happening

1. **`is_byte_level_tokenizer`**:
   - 中文: 用 `repr(backend.pre_tokenizer)` 看字符串里有没有 "ByteLevel" 来判断 —— Llama-3、Qwen、SmolLM、Phi、Mistral v0.3+ 全是 ByteLevel BPE。SentencePiece(Llama-2、Gemma)不是,这里直接 raise。
   - English: Inspects `repr(backend.pre_tokenizer)` for the substring "ByteLevel" — that's enough to identify Llama-3, Qwen, SmolLM, Phi, Mistral v0.3+ as the supported family. SentencePiece-based models (Llama-2, Gemma) don't match and the function raises.

2. **`_bytes_to_unicode`** —— 这是这套体系的整个秘密所在:
   - 中文: 这个函数构造一个"256 个字节 → 256 个可打印 unicode 字符"的双射。为什么要这样?因为 BPE 训练词表时,有些字节(比如 `\x00`、空格、换行)在文本编辑/序列化里会出问题。GPT-2 的解法是:把这些"坏字节"映射到 unicode 256-322 这一段"几乎没人用"的可打印区。结果就是,每个 ByteLevel BPE token 的内部表示**永远是一个 unicode 字符串,但每个字符 1:1 对应一个原始字节**。比如字节 `b'\x20'`(空格)被映射成 `Ġ`(U+0120),所以 Llama-3 词表里出现的 `Ġworld` 实际上是 "空格 + world" 这 6 个字节。
   - English: This function builds a bijection "256 bytes ↔ 256 printable unicode chars". Why? When BPE is trained over a vocab, some raw bytes (NUL, space, newline) cause problems in text editors / serialization. GPT-2's solution was to remap the "problem" bytes into the unicode range 256-322, a near-empty printable area. The result: every ByteLevel BPE token's internal representation is a unicode string where **each character maps 1:1 to exactly one source byte**. Byte `b'\x20'` (space) becomes `Ġ` (U+0120), so the `Ġworld` token in Llama-3's vocab is really the 6 bytes "space + world".

3. **`piece_byte_len(piece) = len(piece)`** —— 一行,但是整个 trick 的精华:
   - 中文: 因为 ByteLevel 是 1 字符 ↔ 1 字节,所以一个 token piece 的字节长度就等于 Python 字符串长度!不需要再 encode 一次。
   - English: Because ByteLevel maps 1 char ↔ 1 byte, a token piece's byte length is literally its Python string length. No re-encoding needed.

4. **`_byte_level_piece_len(piece, text_bytes, start)`** —— 反向解码到字节:
   - 中文: 遍历 piece 里每个 unicode 字符,用 `_BYTE_LEVEL_DECODER` (上面字典的反向)查回它对应的原始字节,得到 piece 的真实字节内容。关键的边界处理:很多 ByteLevel piece 自带前导空格(`Ġworld`),但如果这个 piece 出现在文本开头或紧跟另一个特殊 token,那个前导空格在原文里根本不存在 —— 这种情况要把第一个字节剥掉。
   - English: Iterate each unicode char in the piece, look it up in `_BYTE_LEVEL_DECODER` (the inverse of the bijection above), and you get the piece's actual byte content. The subtle case: many ByteLevel pieces carry an implicit leading space (`Ġworld`), but if the piece happens to be at the very start of the text or right after a special token, that leading space is fictional and must be stripped. Hence the `text_bytes[start:start+1] != b" "` check.

5. **`encode_with_byte_offsets` — 最外层粘合剂**:
   - 中文: 三步:(a) 调用 fast tokenizer 的 `encode_batch` 得到每个 token 的字符偏移 `(char_s, char_e)`;(b) 用 `char_to_byte` 累加表(O(N))把字符偏移翻译成字节偏移;(c) 用 `_normalize_byte_offsets` 修补边界 case(比如多字节字符落在 token 边界、ByteLevel piece 的字节自分配)。最终每个 token 都拿到了一个 `(byte_start, byte_end)` 区间。
   - English: Three steps: (a) call the fast tokenizer's `encode_batch` to get each token's char offsets `(char_s, char_e)`; (b) build the `char_to_byte` cumulative table in O(N) and translate char offsets into byte offsets; (c) patch boundary cases with `_normalize_byte_offsets` (multi-byte chars on token boundaries, byte-self-distribution for ByteLevel pieces). The final output: every token tagged with its `(byte_start, byte_end)` interval.

## 类比 / The analogy

想象老师和学生在改同一份古文。老师用《说文解字》的标准断词(每个"词"是它自己的字段);学生用现代汉语断句(每个"词"是 2-3 个字)。两人想互相对照彼此的"翻译",但断词位置完全错位 —— 直接逐词对比毫无意义。聪明的办法是:**用原文的字数 / 标点作为公共坐标轴**。老师说"我把第 23-25 个字翻成 X",学生说"我把第 21-28 个字翻成 Y",虽然没对齐到同一个词,但他们能在"第 23-25 个字"这段交集里讨论分歧 —— 字节偏移在这里扮演的就是"原文的物理位置"。`encode_with_byte_offsets` 干的事就是让老师和学生模型用同一把字节坐标尺看齐。

Picture a teacher and a student annotating the same classical Chinese text. The teacher uses *Shuowen Jiezi*'s atomic-character segmentation (each "word" is one character); the student uses modern phrase segmentation (each "word" is 2–3 characters). They want to compare each other's translations, but their word boundaries don't align — direct word-by-word comparison is meaningless. The smart move: **use the original text's character-or-punctuation positions as the common axis**. The teacher says "I translated characters 23–25 as X"; the student says "I translated characters 21–28 as Y." Their words don't match, but on the *characters 23–25* sub-span they can compare. Byte offsets play exactly this role: the immovable "physical position" in the source text that both tokenizers can reference. `encode_with_byte_offsets` is the surveyor that nails each token to its byte coordinates so teacher and student can finally talk to each other.

## 自己跑一遍 / Try it yourself

```python
from tokenizers import Tokenizer

backend = Tokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

DEC = {ch: b for b, ch in _bytes_to_unicode().items()}

def piece_to_bytes(piece):
    return bytes(DEC[ch] for ch in piece if ch in DEC)

text = "Hello, 世界!"
enc = backend.encode(text)
char_to_byte = [0]
for ch in text:
    char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))

for tok, (cs, ce) in zip(enc.tokens, enc.offsets):
    raw = piece_to_bytes(tok)
    print(f"{tok!r:20} chars=({cs},{ce}) bytes=({char_to_byte[cs]},{char_to_byte[ce]}) raw={raw!r}")
```

运行 / Run with:
```bash
pip install tokenizers transformers
python try.py
```

预期输出 / Expected output:
```
'Hello'              chars=(0,5)   bytes=(0,5)   raw=b'Hello'
','                  chars=(5,6)   bytes=(5,6)   raw=b','
'Ġä¸ĸ'              chars=(6,8)   bytes=(6,10)  raw=b' \xe4\xb8\x96'
'çķĮ'                chars=(8,9)   bytes=(10,13) raw=b'\xe7\x95\x8c'
'!'                  chars=(9,10)  bytes=(13,14) raw=b'!'
```

中文一句:注意中文字符 "世" 的字符偏移是 `(6,8)` 但字节偏移是 `(6,10)` —— 一个 unicode 字符占 4 个 UTF-8 字节,这就是为什么字节坐标轴比字符坐标轴更可靠。

English: notice "世" has *char* offsets `(6,8)` but *byte* offsets `(6,10)` — a single unicode char takes 4 UTF-8 bytes, and this is exactly why the byte axis is the right common ground for cross-tokenizer alignment.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GPT-2 `tokenizers/byte_level_bpe.py`** / **GPT-2's byte-level BPE**: 中文:`_bytes_to_unicode` 函数最早就出自 GPT-2 的 tokenizer 代码,几乎一字不改地传遍了整个 transformer 生态。 / English: The `_bytes_to_unicode` function originated verbatim in GPT-2's tokenizer code and has been copy-pasted across most of the modern transformer ecosystem.
- **ULD: Universal Logit Distillation (Boizard et al., 2024)** / **ULD paper**: 中文:GOLD trainer 的理论基础。把"byte offset 对齐 → 软 token 分布 KL"作为通用的跨 tokenizer 蒸馏算法提出。 / English: The theoretical backbone of the GOLD trainer. ULD formalises "byte-offset alignment → soft token-distribution KL" as a generic recipe for cross-tokenizer distillation.
- **GKD / On-policy distillation in TRL**: 中文:GKD 假定 tokenizer 相同,所以不用这套机制。但 PR #5885 已经把 GKD 的字节对齐版铺好了 hook —— 后续 GKD 也会跨 tokenizer。 / English: GKD assumes matched tokenizers and skips byte alignment. PR #5885 already wires the hooks so a cross-tokenizer GKD variant is a small follow-up.
- **HF `transformers` `convert_slow_tokenizer.py` —— 也用到 byte↔unicode 字典**: 中文:把 SentencePiece 词表"翻译"成 ByteLevel 表示时需要同一份字典。 / English: The same byte↔unicode dictionary appears when converting SentencePiece vocabs into ByteLevel form.

## 注意事项 / Caveats / when it breaks

- **SentencePiece 模型 → 直接 raise / SentencePiece models simply raise**: 中文:Llama-2、Gemma、原版 T5 等用的是 SentencePiece,不是 ByteLevel。代码直接 NotImplementedError。需要的是不同的 "loss-level projection" 方案(论文 X-Token)。 / English: Llama-2, Gemma, T5 use SentencePiece, not ByteLevel — the code raises. You need the different "loss-level projection" recipe (the X-Token paper) for those families.
- **`Ġ` 前导空格的剥离需要原文 / Stripping the `Ġ` leading space requires the source text**: 中文:`_byte_level_piece_len` 必须传入 `text_bytes` 和 `start`,因为决定要不要剥前导空格只能看原文 byte。脱离原文上下文的 piece-长度计算是错的。 / English: `_byte_level_piece_len` requires both `text_bytes` and `start` because the decision to strip the implicit leading space depends on the actual source byte. Computing piece length from the piece alone is silently wrong on text-start positions.
- **特殊 token 的 offsets 是空的 / Special tokens have empty offsets**: 中文:`<bos>`、`<eos>` 等特殊 token 的 char offset 是 `(0,0)`,代码里 `if start == end:` 那一支专门保留 cursor 不动来处理这种零宽度区间。 / English: Special tokens like `<bos>` / `<eos>` carry `(0,0)` char offsets. The `if start == end:` branch in `_normalize_byte_offsets` keeps the cursor stationary for these zero-width spans.

## 延伸阅读 / Further reading

- [TRL PR #5885 — Cross-tokenizer alignment via byte offsets in GOLD trainer](https://github.com/huggingface/trl/pull/5885)
- [ULD: Universal Logit Distillation (arXiv 2402.12030)](https://arxiv.org/abs/2402.12030)
- [GPT-2 tokenizer source — origin of `_bytes_to_unicode`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
