---
name: daily-code-fetch
description: |
  Daily code teaching-point fetcher (Step 1 of 2 pipeline). Scans tracked GitHub repos
  + pytorch/pytorch + a rotating HuggingFace main library + a rotating VLA repo + a
  rotating World-Action-Model repo + discovers one trending project, picks 1 teaching
  point from each, outputs candidate list to /tmp/daily_code_candidates.json for the
  teach step.

  Triggers: "fetch code candidates", "跑一下 daily code 抓取", debugging Step 1 only.
---

> **Before starting**: Say "🔎 Scanning repos for today's teaching point" and announce today's date.

# Daily Code Fetch

You are the daily code candidate selector. Your job: pick **6** high-quality teaching points
— one from the curated topic-rotation list, one from `pytorch/pytorch`, one from a HuggingFace
main library, one from a VLA repo, one from a World-Action-Model (WAM) repo, and one fresh
trending project — then output a JSON candidate file. **Do not write the educational notes**
— that's the teach step's job.

> **Core teaching goal for `vla` and `wam`**: these two tracks are now **curriculum-driven**.
> Instead of rotating across repos, each day picks the **next uncovered component** in the
> nanoVLA / nanoWAM build plan (defined in `.config/nano-curriculum.json`), respecting
> dependencies. The goal is that after one full curriculum cycle the user has touched every
> building block needed to assemble a nanoVLA / nanoWAM from scratch. Broad-coverage breadth
> across robotics repos lives in the **tracked** track, not here.

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

## Step 2: Scan tracked repos (breadth coverage)

> **Role of this track**: `tracked` is the **broad-coverage** channel — it rotates over
> all `tracked_repos` matching today's topic and surfaces interesting code regardless of
> whether it slots into the nanoVLA / nanoWAM curriculum. On `robotics` days this is where
> non-curriculum robotics work (new datasets, evaluation pipelines, sim-to-real tricks,
> hardware interfaces, alternative architectures the curriculum did not pre-plan for) gets
> highlighted.

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

## Step 3: Pick one PyTorch teaching point (REQUIRED, every day)

Read `pytorch_repo` from the config. **Shallow-clone** (the full repo is huge):

```bash
PT_URL=$(jq -r .pytorch_repo.url .config/tracked-repos.json)
PT_DEPTH=$(jq -r .pytorch_repo.clone_depth .config/tracked-repos.json)
PT_BRANCH=$(jq -r .pytorch_repo.branch .config/tracked-repos.json)
if [ -d "$CACHE/pytorch" ]; then
  git -C "$CACHE/pytorch" fetch --depth=$PT_DEPTH origin "$PT_BRANCH" && git -C "$CACHE/pytorch" reset --hard "origin/$PT_BRANCH"
else
  git clone --depth=$PT_DEPTH --branch "$PT_BRANCH" --single-branch "$PT_URL" "$CACHE/pytorch"
fi
```

Only scan files under the `scan_paths` listed in the config (e.g. `torch/nn`,
`torch/optim`, `torch/_dynamo`, …). Exclude `exclude_paths`. Pick a self-contained
function/class teaching a non-obvious PyTorch internal — examples that read well:

- a fused kernel wrapper in `torch/nn/functional.py`
- an optimizer step (`torch/optim/adamw.py`)
- a DDP/FSDP utility (`torch/distributed/...`)
- an autograd Function (`torch/autograd/...`)
- a `torch.compile` / Dynamo pass (`torch/_dynamo/...`)

Pick something **different from yesterday's pytorch entry** (check `INDEX.md` for `pytorch` topic).

## Step 4: Pick one HuggingFace teaching point (REQUIRED, every day)

Read `huggingface_repos` from config. **Rotate**: pick the next HF library that hasn't
been used in the last N=`len(huggingface_repos)` days (check `INDEX.md` `huggingface` topic).
Shallow-clone:

```bash
git clone --depth=50 --branch main --single-branch "$HF_URL" "$CACHE/$HF_NAME"
```

Pick a 40-150 line teaching point that demonstrates a HF-specific pattern:
- `transformers`: model wiring, generation utility, trainer hook
- `diffusers`: scheduler step, pipeline composition
- `accelerate`: device-mesh / FSDP wrapper, gradient-accumulation context
- `peft`: LoRA injection, adapter merging
- `datasets`: streaming, sharding, map-fusion
- `trl`: PPO/DPO loss, reward-model wrapper
- `tokenizers`: BPE training, normalization pipeline

Avoid pure boilerplate. Prefer recent commits.

## Step 5: Pick one VLA teaching point — CURRICULUM MODE (REQUIRED, every day)

Read `.config/nano-curriculum.json`. The `nano_vla` section has:
- `plan` — ordered list of nanoVLA components, each with `id`, `depends_on`,
  `candidate_repos`, `search_hints`, optional `good_examples`
- `covered` — list of `{id, date, note, repo, via}` records of components already taught

**Pick the next item to teach** as follows:

1. Build `covered_ids = {entry.id for entry in nano_vla.covered}`.
2. Iterate `plan` in order. The first item where:
   - `item.id not in covered_ids`, AND
   - every `dep in item.depends_on` is in `covered_ids`
   …is **today's VLA pick**.
3. If no item is eligible (all covered, or all remaining items are blocked):
   - **All covered case**: pick any covered item from `plan` and find a *different repo's*
     implementation of the same component (cross-repo variant), or move to "advanced
     variants" mode — teach a non-trivial extension (e.g. discrete vs. continuous action
     head, LoRA vs. full fine-tune). Set `curriculum_id` to the original component id
     anyway so the teach step can decide whether to re-cover it.
   - **All blocked case**: report which dep is missing and pick the lowest-index uncovered
     item *ignoring* deps — this should be rare and worth flagging to the user.

**Find a file implementing the chosen component:**

1. For each repo in `item.candidate_repos` (in order):
   - Shallow-clone or pull to `{cache_dir}/{repo_name}`:
     ```bash
     git clone --depth=50 --branch main --single-branch "$URL" "$CACHE/$NAME"
     ```
   - Search for files matching `item.search_hints`:
     ```bash
     # find files whose names match any search_hint, restricted to *.py
     for hint in $HINTS; do
       find "$CACHE/$NAME" -name "*.py" -not -path "*/tests/*" | grep -i "$hint" | head -5
     done
     # OR grep for class/function definitions
     git -C "$CACHE/$NAME" grep -l -E "class.*${hint}|def.*${hint}" -- '*.py'
     ```
   - For each candidate file: pick a 40-150 line self-contained class/function. Prefer
     files listed in `item.good_examples` if available.
2. **Cross-check `INDEX.md`** — the chosen file must not be the same `permalink` as an
   existing note (same SHA + same file path).
3. If no repo yields a good file, fall back to the next eligible curriculum item.

**Output**: in addition to the standard JSON fields, the `vla` entry must include
`"curriculum_id": "<item.id>"` so the teach step knows which curriculum item to mark
covered.

## Step 6: Pick one WAM teaching point — CURRICULUM MODE (REQUIRED, every day)

Exactly the same logic as Step 5, but read `nano_wam` from `.config/nano-curriculum.json`
and use `wam_repos`-aligned `candidate_repos`. The `wam` entry in candidates JSON also
gets `"curriculum_id": "<item.id>"`.

## Step 7: Find one trending project

Use GitHub search API to find one fresh, high-quality project matching `trending_query`:

```bash
curl -s "https://api.github.com/search/repositories?q=$(echo "$QUERY" | jq -sRr @uri)&sort=updated&order=desc&per_page=20" \
  -H "Accept: application/vnd.github.v3+json"
```

Filter results:
- Stars: 200-50000 (skip mega-projects already over-covered)
- Updated within last 14 days
- Has a README in English
- Not already in `tracked_repos` / `pytorch_repo` / `huggingface_repos` / `vla_repos` / `wam_repos`
- Not already covered in `INDEX.md` (last 60 days)

Pick 1 project. Identify its most interesting single file (typically `main.py`, the core
algorithm module, or whatever the README points to as "the implementation").

## Step 8: Output candidates JSON

Write `/tmp/daily_code_candidates.json` with **six** entries:

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
    "concept_hint": "...",
    "why_interesting": "..."
  },
  "pytorch": {
    "repo": "pytorch/pytorch",
    "repo_url": "https://github.com/pytorch/pytorch",
    "file": "torch/optim/adamw.py",
    "lines": "20-95",
    "permalink": "https://github.com/pytorch/pytorch/blob/{sha}/torch/optim/adamw.py#L20-L95",
    "concept_hint": "...",
    "why_interesting": "..."
  },
  "huggingface": {
    "repo": "huggingface/peft",
    "repo_url": "https://github.com/huggingface/peft",
    "file": "src/peft/tuners/lora/layer.py",
    "lines": "...",
    "permalink": "https://github.com/huggingface/peft/blob/{sha}/...",
    "concept_hint": "...",
    "why_interesting": "..."
  },
  "vla": {
    "curriculum_id": "vision-encoder",
    "repo": "huggingface/nanoVLM",
    "repo_url": "https://github.com/huggingface/nanoVLM",
    "file": "models/vision_transformer.py",
    "lines": "20-95",
    "permalink": "https://github.com/huggingface/nanoVLM/blob/{sha}/models/vision_transformer.py#L20-L95",
    "concept_hint": "which architectural component this is — should match the curriculum item's title",
    "why_interesting": "WHY this component matters for building a complete VLA from scratch",
    "nano_vla_mapping": "what role this same component plays in your nanoVLA / production VLA"
  },
  "wam": {
    "curriculum_id": "vae-encoder-decoder",
    "repo": "Wan-Video/Wan2.1",
    "repo_url": "https://github.com/Wan-Video/Wan2.1",
    "file": "wan/modules/vae.py",
    "lines": "20-95",
    "permalink": "https://github.com/Wan-Video/Wan2.1/blob/{sha}/wan/modules/vae.py#L20-L95",
    "concept_hint": "which architectural component this is — should match the curriculum item's title",
    "why_interesting": "WHY this component matters for building a complete WAM from scratch",
    "nano_wam_mapping": "what role this same component plays in your nanoWAM / production WAM"
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
│   ├── infrastructure.md
│   ├── pytorch.md
│   ├── huggingface.md
│   ├── vla.md               # NEW: components for building nanoVLA / production VLA
│   └── wam.md               # NEW: components for building nanoWAM / production WAM
└── YYYY/
    └── MM/                  # entries land here as YYYY-MM-DD-{slug}.md
```

Use the README template at the bottom of this file.

## Output

Tell the user:
- Today's topic
- Which tracked repo + file was picked
- Which pytorch / huggingface / vla / wam files were picked
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
- [PyTorch](topics/pytorch.md)
- [Hugging Face](topics/huggingface.md)
- [VLA — build your own](topics/vla.md)
- [WAM — build your own](topics/wam.md)

## Full archive

See [INDEX.md](INDEX.md).

## How it's built

Generated by the [daily-code](.claude/skills/daily-code/SKILL.md) skill. Each entry
follows a fixed template: code snippet → step-by-step explanation → analogy → minimal
runnable example.

## Configure

Edit [.config/tracked-repos.json](.config/tracked-repos.json) to change which repos are scanned.
```
