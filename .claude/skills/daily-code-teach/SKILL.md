---
name: daily-code-teach
description: |
  Daily code teaching note writer (Step 2 of 2 pipeline). Reads candidates from
  /tmp/daily_code_candidates.json, fetches the actual code, writes 2 educational
  markdown notes (one from tracked repos, one from a trending project) in
  **bilingual Chinese + English** format, updates INDEX.md and topic indexes.

  Triggers: "write daily code notes", "跑一下 daily code 笔记", debugging Step 2 only.
---

> **Before starting**: Say "📝 Writing today's code lessons" and announce today's date.

# Daily Code Teach

You are the educational note writer. Take candidates from Step 1, fetch the actual
source code, and produce 2 high-quality teaching notes in **bilingual format
(Chinese + English)**.

## Prerequisites

Check that `/tmp/daily_code_candidates.json` exists. If not, tell the user to run
`/daily-code-fetch` first and stop.

## Step 1: Fetch the actual code

For each candidate (`tracked` and `trending`):

1. Use the cached clone from `{cache_dir}/{repo_name}` (Step 1 already cloned them).
2. Read the file specified by `file` and the line range `lines`.
3. If `lines` only narrows to a function/class boundary approximately, expand to the
   nearest logical unit (don't cut mid-function).
4. Strip excessive comments only if they obscure the teaching point; otherwise keep
   the original code verbatim.

## Step 2: Write the teaching note

Use this **exact template** for each note. Output language: **bilingual — Chinese (中文) first, then English**, in every prose section. Code blocks, frontmatter, file paths, and identifiers stay as-is (do NOT translate code, do NOT duplicate code blocks).

```markdown
---
date: YYYY-MM-DD
topic: robotics | diffusion | infrastructure
source: tracked | trending
repo: owner/name
file: path/to/file.py
permalink: https://github.com/.../blob/{sha}/path#L20-L95
difficulty: beginner | intermediate | advanced
read_time: ~10 min
tags: [code-of-the-day, {topic}, {specific-technique}]
---

# {中文标题} / {English Title}

> **一句话 / In one line**: {一句话中文总结} / {one-sentence English summary}

## 为什么重要 / Why this matters

{中文段落：解释这段代码为何值得学习、解决了什么实际问题。}

{English paragraph: same content, not a word-for-word translation — write naturally in English.}

## 代码 / The code

`{repo}` — [`{file}`]({permalink})

```python
{the actual code, verbatim or lightly trimmed — single shared block, no translation}
```

## 逐行讲解 / What's happening

1. **第 N 行 / Line N (`some_call(...)`)**:
   - 中文: ...
   - English: ...
2. **第 M-K 行 / Lines M-K (the loop)**:
   - 中文: ...
   - English: ...

(对于较长的步骤,可以改写成"中文段落 + 空行 + English paragraph"的形式,只要每个步骤的两种语言都到位。)

## 类比 / The analogy

{中文段落:用一个具体的日常事物作类比。}

{English paragraph: same analogy, written natively in English.}

## 自己跑一遍 / Try it yourself

```python
# < 30 lines — single shared code block, no translation
```

运行 / Run with:
```bash
pip install {deps}
python try.py
```

预期输出 / Expected output:
```
{what you should see}
```

{中文一两句:指出这个示例里最值得注意的现象。}

{English: same observation, written natively.}

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **{Example 1 中文名}** / **{Example 1 English name}**: {中文描述} / {English description}
- ...

## 注意事项 / Caveats / when it breaks

- **{中文要点}** / **{English point}**: {中文解释} / {English explanation}
- ...

## 延伸阅读 / Further reading

- {Link title — links stay as one entry, no need to translate URL/title}
- ...
```

### Bilingual formatting rules

1. **Section headings**: `## {中文标题} / {English Title}` — Chinese first, slash separator, English second.
2. **Body prose**: write the Chinese paragraph first, then a blank line, then the English paragraph. Do NOT interleave sentence-by-sentence — keep each language as a contiguous block so each reads naturally.
3. **Bullet/numbered lists**: either split each item into `中文: ...` / `English: ...` sub-bullets, OR write each list twice (Chinese list followed by English list) — choose whichever reads cleaner for that section.
4. **Inline identifiers / file paths / numbers / URLs**: never translate. `keys.shape[2]` stays `keys.shape[2]` in both languages.
5. **Code blocks**: appear ONCE. Comments inside code stay in the original language of the source file (don't translate the source's own comments).
6. **Frontmatter fields**: stay in English (`topic`, `difficulty`, `read_time`, etc.).
7. **Tone**: the Chinese version should sound like a native Chinese tech blogger (口语化、技术准确), not a literal translation. Likewise English should sound native. They convey the *same* ideas, but in each language's natural voice.

### Quality bar (mandatory)

1. **Code is real** — verbatim or lightly trimmed from the actual file.
2. **Permalink works** — pinned to commit SHA, not branch name.
3. **Analogy is concrete** — names a specific physical/everyday object or scenario.
4. **Try-it example runs** — fewer than 30 lines, declares its deps.
5. **Bilingual completeness** — every prose section has BOTH a Chinese block and an English block. Skipping either is a defect.
6. **Length** — 250-600 lines total (bilingual roughly doubles the prose budget vs. monolingual).

## Step 3: Save and index

Save each note to `{YYYY}/{MM}/{YYYY-MM-DD}-{slug}.md`.

Update INDEX.md (newest first), README.md (Latest section, last 5), topics/{topic}.md.

## Step 4: Commit (optional)

If the user mentioned pushing: stage all files, commit with message
`daily code: YYYY-MM-DD {topic} ({tracked-repo} + {trending-repo})`, push.

## Notes

- Do NOT skip the trending note.
- Always produce **bilingual** output — Chinese + English in every prose section.
  Past notes (before this skill update) may be English-only; do NOT retroactively
  rewrite them unless the user explicitly asks.
