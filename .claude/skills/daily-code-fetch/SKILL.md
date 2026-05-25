---
name: daily-code-fetch
description: |
  Daily code teaching-point fetcher (Step 1 of 2 pipeline). Scans tracked GitHub repos
  + discovers one trending project, picks 1 teaching point from each, outputs candidate
  list to /tmp/daily_code_candidates.json for the teach step.

  Triggers: "fetch code candidates", "跑一下 daily code 抓取", debugging Step 1 only.
---

> **Before starting**: Say "🔎 Scanning repos for today's teaching point" and announce today's date.

# Daily Code Fetch

You are the daily code candidate selector. Your job: pick 2 high-quality teaching points
from a curated repo list + 1 fresh project, output a JSON candidate file. **Do not write
the educational notes** — that's the teach step's job.

## Step 0: Load configuration

Working directory contains:
- `.config/tracked-repos.json` — the user's curated repo list
- `INDEX.md` — past entries (use this for dedup)

If `.config/tracked-repos.json` does not exist, this is first-time setup:
1. Ask the user for 3-8 GitHub repo URLs they want to track (one per line)
2. Ask which topic each falls under: `robotics` | `diffusion` | `infrastructure`
3. Write the config in this shape:
   ```json
   {
     "tracked_repos": [
       {"url": "https://github.com/openvla/openvla", "topic": "robotics", "branch": "main"},
       {"url": "https://github.com/huggingface/lerobot", "topic": "robotics", "branch": "main"}
     ],
     "trending_query": "language:python (robotics OR \"world model\" OR \"diffusion policy\") stars:>200 pushed:>2026-04-01",
     "cache_dir": "/tmp/daily_code_cache",
     "topic_rotation": ["robotics", "diffusion", "infrastructure"]
   }
   ```
4. Also create the repo skeleton (see "Repo skeleton" section below) if it doesn't exist yet.

## Step 1: Pick today's topic

Read `topic_rotation` from config. Find the last entry in `INDEX.md` and pick the
next topic in rotation. If INDEX.md is empty, use the first topic. This keeps notes balanced
across categories.

## Step 2: Scan tracked repos

For each repo in `tracked_repos` matching today's topic (fallback: any repo if no match):

1. **Shallow clone or pull** to `{cache_dir}/{repo_name}`:
   ```bash
   if [ -d "$CACHE/$NAME" ]; then
     git -C "$CACHE/$NAME" fetch --depth=100 origin && git -C "$CACHE/$NAME" reset --hard origin/$BRANCH
   else
     git clone --depth=100 --branch $BRANCH $URL "$CACHE/$NAME"
   fi
   ```

2. **Find candidate files**. Run inside the clone:
   ```bash
   git -C "$REPO" log --since=30.days --name-only --pretty=format: -- '*.py' | \
     sort -u | head -50
   ```
   Filter out:
   - `test_*.py`, `*_test.py`, `tests/`
   - `__init__.py`, `setup.py`, `conftest.py`
   - Files < 30 lines
   - Files in `docs/`, `examples/` only (these are OK but lower priority)

3. **Pick 1 teaching point per repo**. Open the candidate file and look for:
   - A self-contained function or class (40-120 lines)
   - With docstring or recent commit explaining intent
   - That demonstrates a non-obvious technique (not boilerplate)
   - Cross-check against `INDEX.md` to avoid repeats

## Step 3: Find one trending project

Use GitHub search API to find one fresh, high-quality project matching `trending_query`:

```bash
curl -s "https://api.github.com/search/repositories?q=$(echo "$QUERY" | jq -sRr @uri)&sort=updated&order=desc&per_page=20" \
  -H "Accept: application/vnd.github.v3+json"
```

Filter results:
- Stars: 200-50000 (skip mega-projects already over-covered)
- Updated within last 14 days
- Has a README in English
- Not already in `tracked_repos`
- Not already covered in `INDEX.md` (last 60 days)

Pick 1 project. Identify its most interesting single file (typically `main.py`, the core
algorithm module, or whatever the README points to as "the implementation").

## Step 4: Output candidates JSON

Write `/tmp/daily_code_candidates.json` with two entries:

```json
{
  "date": "YYYY-MM-DD",
  "topic": "robotics",
  "tracked": {
    "repo": "openvla/openvla",
    "repo_url": "https://github.com/openvla/openvla",
    "file": "prismatic/vla/action_tokenizer.py",
    "lines": "20-95",
    "raw_url": "https://raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/action_tokenizer.py",
    "permalink": "https://github.com/openvla/openvla/blob/{commit_sha}/prismatic/vla/action_tokenizer.py#L20-L95",
    "concept_hint": "Discretizing continuous robot actions into language model tokens",
    "why_interesting": "Shows the bridge trick that lets a VLM output robot actions without architectural changes"
  },
  "trending": {
    "repo": "owner/name",
    "repo_url": "...",
    "stars": 1234,
    "file": "...",
    "lines": "10-80",
    "raw_url": "...",
    "permalink": "...",
    "concept_hint": "...",
    "why_interesting": "..."
  }
}
```

`permalink` must pin to the specific commit SHA so the link doesn't break when the file
changes. Get it via `git -C $REPO rev-parse HEAD`.

## Repo skeleton (first-time setup only)

If the repo skeleton does not exist, create:

```
./
├── README.md                # latest entries + how to use
├── INDEX.md                 # archive index, auto-updated by teach step
├── .config/
│   └── tracked-repos.json   # user's curated list
├── topics/
│   ├── robotics.md          # auto-generated topic index
│   ├── diffusion.md
│   └── infrastructure.md
└── YYYY/
    └── MM/                  # entries land here as YYYY-MM-DD-{slug}.md
```

Use the README template at the bottom of this file.

## Output

Tell the user:
- Today's topic
- Which tracked repo + file was picked
- Which trending project was picked
- Prompt: ready to run `/daily-code-teach`

## README.md template (first-time only)

```markdown
# Daily Code

One curated code teaching point every day, picked from tracked open-source projects
in robotics, diffusion models, and ML infrastructure — plus one freshly-discovered
quality project.

## Latest

<!-- auto-updated by daily-code-teach -->

## Topics

- [Robotics](topics/robotics.md)
- [Diffusion / World Model](topics/diffusion.md)
- [Infrastructure](topics/infrastructure.md)

## Full archive

See [INDEX.md](INDEX.md).

## How it's built

Generated by the [daily-code](.claude/skills/daily-code/SKILL.md) skill. Each entry
follows a fixed template: code snippet → step-by-step explanation → analogy → minimal
runnable example.

## Configure

Edit [.config/tracked-repos.json](.config/tracked-repos.json) to change which repos are scanned.
```
