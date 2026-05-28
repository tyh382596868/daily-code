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

> **Core teaching goal for `vla` and `wam`**: every pick must be a building block that the
> user could plausibly translate into their own from-scratch implementation (`nanoVLA` or
> `nanoWAM`) or a production system. Prefer **architectural components** (model wiring,
> tokenizer, action head, scheduler, loss, training loop, conditioning logic) over peripheral
> utilities (logging, configs, eval harnesses). Across a full rotation cycle, the picks should
> collectively cover the major pieces needed to build the system end-to-end.

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

## Step 5: Pick one VLA teaching point (REQUIRED, every day)

Read `vla_repos` from config. **Rotate**: pick the next VLA repo that hasn't been used in
the last N=`len(vla_repos)` days (check `INDEX.md` `vla` topic). Shallow-clone:

```bash
git clone --depth=50 --branch main --single-branch "$VLA_URL" "$CACHE/$VLA_NAME"
```

**Bias the pick toward components needed to build a nano/production VLA.** Prefer files that
implement one of:

- **vision / observation encoder** — how images become tokens (e.g. SigLIP / DINO patch embed,
  multi-camera fusion, proprioception encoder)
- **action tokenizer / action head** — discrete action bins, continuous regressor, flow-matching
  head, diffusion head (e.g. openvla `action_tokenizer.py`, openpi flow-matching head)
- **VLM backbone wiring** — how prefix tokens, image tokens, action tokens are interleaved and
  fed to the transformer
- **training step** — the actual loss assembly: cross-entropy on discrete actions, flow-matching
  loss on continuous actions, BC + RT-2 style autoregressive
- **action chunking / horizon prediction** — how multi-step action sequences are produced and
  unrolled (RT-style, openvla-oft style)
- **fine-tune script / LoRA application** — the practical entry-point for adapting to a new robot
- **inference loop** — how observations are streamed and actions are produced at runtime

Avoid dataset glue, logging, configs, and pure eval harnesses unless they teach a unique idea.
Pick something **different from yesterday's vla entry**.

## Step 6: Pick one World-Action-Model (WAM) teaching point (REQUIRED, every day)

Read `wam_repos` from config. **Rotate**: pick the next WAM repo that hasn't been used in
the last N=`len(wam_repos)` days (check `INDEX.md` `wam` topic). Shallow-clone:

```bash
git clone --depth=50 --branch main --single-branch "$WAM_URL" "$CACHE/$WAM_NAME"
```

**Bias the pick toward components needed to build a nano/production World Action Model.** Prefer
files that implement one of:

- **VAE / latent encoder / decoder** — pixel ↔ latent compression, temporal 3D VAE
- **DiT-style backbone block** — spatial / temporal attention, adaLN-Zero modulation,
  patch embedding, rotary position
- **noise scheduler / flow matcher** — DDPM / DDIM / rectified-flow training and sampling
  schedules
- **conditioning logic** — text encoder (T5 / CLIP / Qwen) attached via cross-attention,
  action conditioning, classifier-free guidance dropout
- **training loop** — full denoising-step loss assembly with the right preconditioning
- **inference loop** — sampler (Euler / Heun), step counts, CFG combine, latent → pixel decode
- **temporal compression / frame fusion** — patchifying space + time, causal video masking
- **action-frame fusion** — how robot actions are projected into the same latent stream as
  visual frames (this is the "A" in WAM)

Avoid dataset glue, configs, and visualization scripts unless they encode a unique algorithmic
idea. Pick something **different from yesterday's wam entry**.

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
    "repo": "openvla/openvla",
    "repo_url": "https://github.com/openvla/openvla",
    "file": "prismatic/vla/action_tokenizer.py",
    "lines": "20-95",
    "permalink": "https://github.com/openvla/openvla/blob/{sha}/prismatic/vla/action_tokenizer.py#L20-L95",
    "concept_hint": "which architectural component this is (action tokenizer / vision encoder / action head / training step / ...)",
    "why_interesting": "WHY this component matters for building a complete VLA from scratch",
    "nano_vla_mapping": "what role this same component plays in your nanoVLA / production VLA"
  },
  "wam": {
    "repo": "Wan-Video/Wan2.1",
    "repo_url": "https://github.com/Wan-Video/Wan2.1",
    "file": "wan/modules/dit.py",
    "lines": "20-95",
    "permalink": "https://github.com/Wan-Video/Wan2.1/blob/{sha}/wan/modules/dit.py#L20-L95",
    "concept_hint": "which architectural component this is (VAE / DiT block / scheduler / conditioning / training step / sampler / ...)",
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
