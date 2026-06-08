---
name: daily-code
description: |
  Daily code learning entry point. Trigger when user says "daily code", "today's code",
  "今日 daily code", "code of the day", "学一段代码", "今日学点代码".

  Internally chains two steps: fetch candidate teaching points (tracked-rotation repo,
  pytorch/pytorch, a HuggingFace main library, one VLA repo, one World-Action-Model repo,
  and one trending project), then write educational markdown notes. Output: **6** notes
  per day saved to the daily-code repo.
---

# Daily Code

One-line entry for the daily code learning system. The user should normally just say:

- `daily code`
- `今日 daily code`
- `today's code`

## Execution principle

1. Auto-invoke `/daily-code-fetch` (Step 1: pick teaching point candidates).
2. After Step 1, auto-invoke `/daily-code-teach` (Step 2: write educational notes,
   commit, push, open a PR, and squash-merge it into `main` — all automatic, no
   review needed). See `/daily-code-teach` Step 4 for the exact flow and the
   "don't merge yet" opt-out.
3. After all done, tell the user in one sentence:
   - How many notes were generated (always 6: tracked + pytorch + huggingface + vla + wam + trending)
   - Where they were saved
   - That the content is now on `main` (cite the PR number and merge commit sha)

> **`vla` and `wam` tracks have a special teaching goal**: the user is building their own
> `nanoVLA` / `nanoWAM` and a production-scale version from scratch. Picks must be
> *components* (vision encoder, action head, scheduler, DiT block, VAE, training loop,
> conditioning, etc.) — not configs, logging, or eval glue — and notes must explicitly
> map each component to its role in a from-scratch build (see daily-code-teach SKILL.md
> for the mandatory "在 nanoVLA / nanoWAM 中的位置" section).

## Important constraints

- Do NOT ask the user to manually run the sub-skills first; this entry runs the full pipeline.
- If the user explicitly wants only one step (e.g. "just fetch candidates"), then hand off to the corresponding sub-skill.
- If the user wants to add/remove tracked repos, point them to `.config/tracked-repos.json`.

## First-time setup

If `.config/tracked-repos.json` does not exist:
1. Create the repo skeleton (see daily-code-fetch SKILL.md)
2. Ask the user for their list of tracked repo URLs and write them to the config
3. Then proceed with the fetch step

## Output format

All notes are written in **bilingual format — Chinese (中文) + English** in every prose
section, following the code-snippet + explanation + analogy template defined in
`/daily-code-teach`. Code blocks appear once and are not translated.
