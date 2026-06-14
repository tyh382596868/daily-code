# Repository Working Rules

This file is read by Claude Code on every session in this repo. Rules here are
non-negotiable.

## 1. Code grounding (查阅代码,不能瞎想)

When the user asks a question about how a specific codebase / module / function
works, **every behavioural claim MUST cite a real `file:line`** read with the
Read / Grep / Glob tools in the current session.

- Do NOT paraphrase API behaviour from memory ("应该是这样", "I think it does X").
- Do NOT invent shapes / dtypes / control flow from intuition when the source
  is reachable. If a clone is missing, clone it first, then read.
- Cite as `path/to/file.py:LINE` (single line) or `path/to/file.py:START-END`
  (range). Citations let the user verify the claim by clicking through.
- If a claim is unavoidable conjecture (e.g. "the author probably did this
  because…"), **mark it explicitly as a hypothesis**, not as fact.

This rule applies to:
- FastWAM / lingbot-va / dreamzero / openpi / lerobot / Wan2.x / Open-Sora /
  Isaac-GR00T / nanoVLM / openvla / openvla-oft (the curated repos under
  `.config/tracked-repos.json`)
- pytorch/pytorch, huggingface/transformers / diffusers / accelerate / peft /
  trl / tokenizers / nanoVLM, and any other repo discussed in a daily-code note
- Any code the user pastes or names in conversation

The cost of one extra Read/Grep call is much lower than the cost of one wrong
claim about a repo the user is about to build on.

## 2. teaching-code skill invocation

The `teaching-code` skill is **slash-command-only**: invoke it only when the
user literally types `/teaching-code` in their message. Natural-language
phrasings ("讲解 X", "教我 X", "teach me Y", "explain Z", "walk me through W")
are NOT triggers — answer those in normal conversational style.

See `.claude/skills/teaching-code/SKILL.md` for the full rule.

## 3. daily-code pipeline output discipline

When invoked via the `daily-code` / `daily-code-fetch` / `daily-code-teach`
skills, always produce **6** bilingual notes (中文 + English in every prose
section): tracked + pytorch + huggingface + vla + wam + trending. Default
end-of-pipeline behaviour is to commit, push, open a PR, and squash-merge
to `main` — see `.claude/skills/daily-code-teach/SKILL.md` Step 4.

## 4. Git operations

Develop on the branch named in the session's "Git Development Branch
Requirements" block (typically `claude/*`). Never force-push to `main`. Always
go through a PR + squash-merge. Don't delete the head branch — the sandbox
can't push deletes; the user cleans up the remote branch on GitHub.
