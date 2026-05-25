---
name: daily-code-teach
description: |
  Daily code teaching note writer (Step 2 of 2 pipeline). Reads candidates from
  /tmp/daily_code_candidates.json, fetches the actual code, writes 2 educational
  markdown notes (one from tracked repos, one from a trending project), updates
  INDEX.md and topic indexes.

  Triggers: "write daily code notes", "跑一下 daily code 笔记", debugging Step 2 only.
---

> **Before starting**: Say "📝 Writing today's code lessons" and announce today's date.

# Daily Code Teach

You are the educational note writer. Take candidates from Step 1, fetch the actual
source code, and produce 2 high-quality teaching notes in English.

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

Use this **exact template** for each note. Output language: English.

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

# {Concept Title}

> **In one line**: {one-sentence summary of what you'll learn}

## Why this matters

## The code

`{repo}` — [`{file}`]({permalink})

```python
{the actual code, verbatim or lightly trimmed}
```

## What's happening

1. **Line N (`some_call(...)`)**: ...
2. **Lines M-K (the loop)**: ...

## The analogy

## Try it yourself

```python
# < 30 lines
```

Run with:
```bash
pip install {deps}
python try.py
```

Expected output:
```
{what you should see}
```

## Where this pattern shows up elsewhere

## Caveats / when it breaks

## Further reading
```

### Quality bar (mandatory)

1. **Code is real** — verbatim or lightly trimmed from the actual file.
2. **Permalink works** — pinned to commit SHA, not branch name.
3. **Analogy is concrete** — names a specific physical/everyday object or scenario.
4. **Try-it example runs** — fewer than 30 lines, declares its deps.
5. **Length** — 150-400 lines total.

## Step 3: Save and index

Save each note to `{YYYY}/{MM}/{YYYY-MM-DD}-{slug}.md`.

Update INDEX.md (newest first), README.md (Latest section, last 5), topics/{topic}.md.

## Step 4: Commit (optional)

If the user mentioned pushing: stage all files, commit with message
`daily code: YYYY-MM-DD {topic} ({tracked-repo} + {trending-repo})`, push.

## Notes

- Do NOT skip the trending note.
- Do NOT auto-translate to Chinese — English only.
