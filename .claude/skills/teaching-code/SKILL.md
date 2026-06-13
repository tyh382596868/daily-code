---
name: teaching-code
description: |
  Code teaching framework that walks through any code snippet using the user's own
  three-step learning workflow: (1) inputs/outputs with shapes, (2) operations and
  shape transitions as arrows, (3) the actual code line by line.

  Use ONLY when the user explicitly invokes this skill (e.g. `/teaching-code`,
  "teaching code", "teaching_code", "讲解这段代码", "教我这段代码") and provides
  a code target (file path, function/class name, pasted snippet, or external
  reference like "SmolVLA's eager_attention_forward"). Do NOT auto-invoke for
  generic code-explanation questions — the user wants explicit activation.

  Output: an explanation in the exact three-step structure, code-grounded with
  file:line citations, ending with a short "go write your own version" hook.
---

# Teaching Code

A code-walkthrough format that mirrors the user's own coding workflow
(see `coding_workflow.md`). The point is to teach the user **to think the way
they would write the code from scratch**, not to dump explanations they have to
passively absorb.

## When to invoke

User explicitly invokes this skill **and** provides a code target. Examples:

- `/teaching-code src/foo.py:42`
- `/teaching-code MyModel.forward`
- `/teaching-code SmolVLA 的 eager_attention_forward`
- `/teaching-code` followed by a pasted code block

If the user invokes the skill without a target, ask: "想讲哪段代码?给我一个
文件路径、函数名、或者粘贴一段。"

## Execution: strict three steps, no shortcuts

### Step 1: 输入 / 输出 (shapes first, always)

State up front:

- **输入**: tensor shape(s), dtype, device, and the **semantic meaning of every
  dimension**. Example: `video: (B, K, 3, H, W) — B 个样本, K 帧, 3 通道, H×W 像素`
- **输出**: same — shape, dtype, semantic meaning.

If shapes are not derivable from the snippet alone, **stop and check the caller**
(read upstream code, find the example invocation, look at the test). Do **not**
fabricate shapes from intuition. If the upstream isn't reachable, ask the user
for the call site before proceeding.

Use a table when multiple inputs are involved (image + language + state + action
type situations).

### Step 2: 数据流 + 形状变化 (arrows, no Python API yet)

Sketch the data flow as a sequence of arrows, one operation per arrow.
Each arrow shows the shape **before → after**.

Example:

```
(B, K, 3, H, W)
  → patchify →             (B*K, dim, grid, grid)
  → flatten + transpose →  (B, K, N, dim)
  → 加位置编码 →            (B, K, N, dim)   形状不变
  → 注意力 × L 层 →         (B, K, N, dim)   形状不变
  → 丢历史帧 →              (B, N, dim)
```

Rules for Step 2:

- One arrow = one conceptually distinct operation
- Annotate "形状不变" when shape doesn't change (helps spot identity ops)
- **Do not reference the actual Python API yet** — describe operations in words.
  ("patchify", not "nn.Conv2d"). This is the "画箭头" stage from the workflow doc.
- The output of Step 2 **is** the forward-pass skeleton the user would write
  themselves. Keep that property — if Step 2 doesn't read like a skeleton they
  could write from, it's too detailed.

### Step 3: 具体代码实现 (now show the actual code)

Walk through the actual code in narrative order (top-to-bottom of the function),
mapping each line back to the arrows from Step 2. For every non-trivial line:

- Cite `file_path:line_number` so the user can navigate.
- Explain **what tensor op is happening** (the "what").
- Explain **why this particular API call** was chosen — the "每个箭头我打算用
  什么工具" principle. E.g. "用 `view` 而不是 `reshape`,因为这里张量保证
  contiguous"; "用 `cat(dim=1)` 而不是 `stack`,因为要在已有维度上拼接"。
- For dtype/device-sensitive ops (upcasting Q/K to float32 before logits,
  GQA expand, RoPE applying only to Q/K, embedding `× sqrt(d_model)`, etc.),
  call them out explicitly — Step 2's shape table doesn't catch these and
  they're the most-bug-prone lines.

**Be code-grounded**: if the code is reachable, read it with the Read tool
**before** quoting. Do not paraphrase API behavior from memory. This is the
discipline the user has been enforcing on me throughout the VLA deep-dives,
and the skill exists to bake it in.

## Output style rules

- **Mirror the workflow doc structure exactly** — three numbered sections with
  the same Chinese names so the user instantly recognizes the format.
- **Code-grounded**: every behavioral claim ties to a specific `file:line`.
  Vibes-based "probably this does X" is banned.
- **End with a short "go try it yourself" hook**: one suggestion of what the
  user could now write from scratch given this understanding. Tie it to their
  nanoVLA / nanoWAM build when relevant. Example:
  "现在你可以试着自己写一个 `eager_attention_forward` 的最小版,
  先固定 `num_heads=1` 跳过 GQA,等跑通再加。"
- **No emojis.** Plain text + code blocks + tables.
- Prose default to Chinese. Code identifiers stay as-is.

## Mandatory self-check before responding

1. Did I read the target code (file_path:line_number cited)?
   If only a snippet was given and it references undefined names, did I ask
   for the upstream context instead of guessing?
2. Did I state shapes for **every** input and output, not just one?
3. Are Step 2 arrows free of Python API references?
4. Does every Step 3 explanation cite a `file:line`?
5. Did I call out dtype/device-sensitive operations explicitly?
6. Did I end with a "go write your own" hook?

If any answer is no, fix before sending.

## Anti-patterns (the failure modes this skill exists to prevent)

- Jumping straight to code without Step 1/2 — this is the dominant failure mode
  the workflow doc was designed to fix.
- Fabricating shapes from intuition when the actual code is reachable.
- Paraphrasing API behavior from memory.
- Quoting the entire function up front, then walking through it — walk through
  it incrementally instead, one logical chunk at a time.
- Over-explaining trivial lines (`super().__init__()` doesn't need a paragraph).
- Treating Step 2 as a place to dump implementation details — it's a skeleton,
  not a transcript.
