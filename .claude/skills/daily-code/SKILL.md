---
name: daily-code
description: |
  Daily code learning entry point. Trigger when user says "daily code", "today's code",
  "今日 daily code", "code of the day", "学一段代码", "今日学点代码".

  Internally chains two steps: fetch candidate teaching points from tracked repos +
  one trending project, then write educational markdown notes. Output: 2 notes per day
  saved to the daily-code repo (one from a tracked repo, one from a discovered project).
---

# Daily Code

One-line entry for the daily code learning system. The user should normally just say:

- `daily code`
- `今日 daily code`
- `today's code`

## Execution principle

1. Auto-invoke `/daily-code-fetch` (Step 1: pick teaching point candidates).
2. After Step 1, auto-invoke `/daily-code-teach` (Step 2: write educational notes).
3. After all done, tell the user in one sentence:
   - How many notes were generated (always 2: tracked + trending)
   - Where they were saved
   - Whether INDEX.md was refreshed

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

All notes are written in **English**, following the code-snippet + explanation + analogy template defined in `/daily-code-teach`.
