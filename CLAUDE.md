# Claude Code Instructions

Read `AGENTS.md` in this directory for full instructions on developing Reachy Mini applications.

## Family fork (alfred-bot-001)

- `xiaobai` is the branch installed on the robot; `main` only tracks upstream. Never push to `upstream` (pollen-robotics).
- Work in your own git worktree on a `feat/<task>` branch cut from `origin/xiaobai`; never switch branches or commit in the shared checkout.
- Deploy (HF Space `Max-robot/reachy_mini_xiaobai`) only from a pushed `xiaobai` commit, after diffing the Space snapshot against `git archive origin/xiaobai`.
- Full rules: AI_Home `docs/development-practices.md`; deployment steps: AI_Home `docs/xiaobai-handover.md` §5.2.
