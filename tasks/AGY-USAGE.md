# Antigravity CLI (`agy`) — verified usage pattern

Verified 2026-09-19 by smoke test (created a file in cwd, exit 0, ~11 s round trip).

## Headless / non-interactive invocation (the one we use)

```bash
cd <working directory>            # agy operates on the current working directory
agy --dangerously-skip-permissions --print "$(cat tasks/TASK-XX-name.md)"
```

- `--print` / `-p` / `--prompt`  : run a single prompt non-interactively and print the response
- `--dangerously-skip-permissions`: auto-approve every tool call (user-authorised for this project)
- Working directory = shell cwd. There is **no** explicit `--cwd` flag; use `--add-dir <path>`
  to expose extra directories (repeatable). Always `cd Deep_CNN` first.
- `--model <id>`                  : optional; `agy models` lists ids. Useful ones:
  `claude-opus-4-6-thinking`, `claude-sonnet-4-6`, `gemini-3.1-pro-high`, `gemini-3.8-flash-high`
- `--effort low|medium|high`      : reasoning effort
- `--print-timeout 0s`            : wait until the turn completes (default)
- `--output-format json`          : machine-readable result (optional)
- `-c` / `--continue`             : continue the most recent conversation (for feedback rounds)

## Standard delegation recipe

```bash
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
git add -A && git commit -qm "pre-delegation snapshot: TASK-XX"     # rollback point
agy --dangerously-skip-permissions --model claude-opus-4-6-thinking --print \
  "Read the file tasks/TASK-XX-name.md in the current directory and carry out that task exactly. \
   Respect every scope restriction in it. When done, print a short report: files changed, \
   commands run, results, and whether each acceptance criterion passed." \
  2>&1 | tee tasks/logs/TASK-XX.agy.log
```

Then the reviewer runs the acceptance commands from the spec and inspects `git diff`.

## Feedback round

```bash
agy --dangerously-skip-permissions -c --print "Feedback on TASK-XX: <what failed>. Fix and re-run acceptance."
```

## Failure handling
If `agy` reports login/quota problems, stop and tell the project owner (they will log in).
