---
name: ralph
description: "Ralph Wiggum Loop - iterative implement/test/fix cycle that keeps going until all tests pass and lint is clean. Use when you need to work through a task with rigorous QA."
user_invocable: true
---

# Ralph Wiggum Development Loop

You are entering a Ralph loop - an iterative development cycle that does not stop until ALL goals are achieved with zero failures.

## Loop Protocol

For each iteration:

1. **IMPLEMENT** - Make the code changes needed for the current task
2. **LINT** - Run `python -m ruff check src/ tests/` and fix any errors
3. **TEST** - Run `python -m pytest tests/ -q --tb=short` and check results
4. **EVALUATE** - If lint or tests fail:
   - Diagnose the root cause (read the error, don't guess)
   - Fix the issue
   - Go back to step 2 (do NOT skip lint after a fix)
5. **VERIFY** - Only when lint is clean AND all tests pass:
   - Confirm the original goal is achieved
   - If yes, report success and exit the loop
   - If no (goal not fully met), go back to step 1

## Rules

- **Never skip steps.** Every iteration runs lint AND tests.
- **Never declare success with failing tests.** Zero tolerance.
- **Fix forward, don't revert.** If a change breaks something, fix the breakage rather than undoing the change (unless the change was fundamentally wrong).
- **Be honest about results.** If something doesn't work, say so and diagnose why.
- **Track iterations.** Report "Ralph Loop iteration N" at the start of each cycle.
- **Maximum 10 iterations.** If not resolved after 10 iterations, stop and report what's blocking.

## Usage

When the user invokes `/ralph`, ask them what task to accomplish, then enter the loop. If they provide the task inline (e.g., `/ralph fix the web UI slider`), start immediately.

## Output Format

At the end of the loop, report:
```
Ralph Loop Complete
- Iterations: N
- Lint: PASSED
- Tests: X passed, Y skipped, Z failed
- Goal: [achieved/not achieved]
- Changes made: [list of files changed]
```
