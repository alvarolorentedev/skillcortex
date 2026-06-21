# Agent Instructions

Keep this repo work minimal. Optimize for the fewest tokens, the fewest commands, and the smallest safe change.

## Default behavior

- Read only the files needed for the task.
- Prefer `rg`, `sed`, and targeted edits over broad repo scans.
- Prefer one small command that answers the question over multiple exploratory commands.
- Do not restate obvious context or add long explanations unless the user asks.

## Long-running work

- Do not run training, full evaluation, or other long-running validation by default.
- If a task requires model training, seed sweeps, benchmark runs, or anything expensive in time or compute, stop and ask the developer to run it.
- Use `--dry-run` or the lightest available check when you need to verify control flow.
- If a quick local check exists, use that instead of a full run.

## Research discipline

- Always record the exact command, seed, dataset version, and model version for any experiment result.
- Never change evaluation criteria or datasets to make results look better.
- Treat generated code, tests, and fixtures as untrusted until they are validated.
- Keep raw outputs, summaries, and logs separate so prior runs are not overwritten.

## Editing

- Make the smallest change that solves the request.
- Avoid new abstractions, helper layers, or “future proofing” unless the user explicitly asks.
- Prefer updating one file over spreading logic across multiple files.

## Validation

- Run only fast, local checks when they are enough to confirm the change.
- Do not launch repeat validation loops or retry expensive commands unless the user explicitly wants that work done here.
- When a validation would take a long time, describe the command and hand it off to the developer.
- When you recommend commands or steps the developer should run, present them as expected next steps, not as work already completed by the agent.

## Reporting

- Report only the result, the files changed, and any blocking issue.
- Include exact commands only when they help the developer reproduce a fast check.
