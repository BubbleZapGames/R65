---
description: "Performs code review of recent changes using the code-reviewer agent"
allowed-tools:
  [
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git status:*)",
    "Bash(git merge-base:*)",
    "Read",
    "Glob",
    "Grep",
    "Task",
  ]
---

# Claude Command: Code Review

Performs a thorough code review of recent changes, focusing on correctness, 65816 hardware concerns, and architecture.

## Determine Scope

Determine what to review based on user arguments:

- **No arguments**: Default to branch diff vs master (`git diff master...HEAD`)
- **`--uncommitted`**: Review staged + unstaged changes (`git diff HEAD`)
- **`--staged`**: Review only staged changes (`git diff --cached`)
- **`<commit>..<commit>`**: Review the specified commit range

If the current branch IS master, fall back to uncommitted changes.

Run `git diff` with the appropriate range to get the full diff. Also run `git log --oneline` for the same range to understand commit structure.

## Read Relevant Documentation

Based on the files and subsystems touched in the diff, selectively read docs that are relevant. Use these heuristics:

- Changes to `codegen/` → read `docs/code-generation.md`, `docs/register-allocation.md`
- Changes to `typeck/` → read `docs/type-system.md`
- Changes to `frontend/` or parser → read `docs/macros.md`, `docs/control-flow.md`
- Changes to `hir/` → read `docs/mode-transition-analysis.md`
- Changes to calling convention or function handling → read `docs/abi-models.md`
- Changes to pointer/memory code → read `docs/pointers-memory.md`
- Changes to interrupt handling → read `docs/interrupt-mode-transition.md`
- Changes to struct/array codegen → read `docs/struct-array-indexing.md`
- Changes to operators → read `docs/operators.md`

Also always read CLAUDE.md for project context. Do NOT read every doc — only those relevant to the changed code.

## Review Directives

Analyze the diff with the following focus areas, in priority order:

### 1. Correctness and Bugs (Critical)
- Logic errors, off-by-one mistakes, uninitialized state
- Missing edge cases (e.g., empty arrays, zero-length, boundary values)
- Incorrect control flow (missing breaks, wrong branch conditions)
- Type mismatches or incorrect casts
- Regressions — does this change break existing behavior?

### 2. 65816 Hardware Concerns (Critical/Warning)
- Register clobbers: does the code inadvertently overwrite A, X, Y, or STATUS when a value is still live?
- Mode mismatches: m8/m16 or x8/x16 mode not matching the operation size
- Bank boundary violations: cross-bank access without proper far pointers or DBR setup
- Pull instruction flag clobbers: PLA/PLX/PLY/PLB/PLD all set N and Z flags — flag-dependent code after pulls is a bug
- Missing REP/SEP: 16-bit operations emitted while in 8-bit mode or vice versa
- Stack frame correctness: do prologue/epilogue bytes balance? Are stack-relative offsets correct?

### 3. Architecture and Design (Warning/Suggestion)
- Does the change fit the existing architecture and patterns in the codebase?
- Are new abstractions justified, or do they add unnecessary complexity?
- Is there duplicated logic that should be consolidated?
- Are module boundaries respected (e.g., codegen shouldn't do type checking)?

### 4. Test Coverage (Warning)
- New logic paths or bug fixes should have corresponding tests
- Flag when a change adds or modifies behavior without test coverage
- Check both unit tests (Rust `#[test]`) and end-to-end tests (`.r65` test files)

### 5. Performance (Moderate)
- Only flag clear performance issues, not micro-optimizations
- Redundant mode switches (REP/SEP pairs that cancel out)
- O(n) lookups where O(1) alternatives exist (e.g., HashMap vs linear scan)
- Unnecessary allocations in hot paths
- Do NOT nitpick individual cycle counts unless the waste is egregious

## Output Format

Present findings grouped by priority. Use this structure:

### Critical Issues
Items that are likely bugs or will cause incorrect behavior. These MUST be addressed.
```
**[FILE:LINE]** Description of the issue
  → Suggested fix or investigation
```

### Warnings
Items that are probably wrong or could cause subtle issues.
```
**[FILE:LINE]** Description of the concern
  → Recommendation
```

### Suggestions
Improvements to architecture, clarity, or maintainability. Optional but recommended.
```
**[FILE:LINE]** Description of the suggestion
  → Proposed approach
```

### Nits
Minor style or cosmetic observations. Low priority.

### Summary
End with a 2-3 sentence summary: what the change does, overall assessment (looks good / needs work / has blockers), and the most important action item if any.

## Rules

- Do run `python -m pytest r65/tests`
- Do NOT suggest adding comments, docstrings, or type annotations to unchanged code
- Do NOT flag style issues in code that wasn't modified in this diff
- Focus review on the CHANGED lines, but consider surrounding context for correctness
- If the diff is very large (>500 lines), use the Task tool with subagent_type=Explore to parallelize review across subsystems
- Be specific: always cite file paths and line numbers, and explain WHY something is a problem
- Avoid false positives — if you're unsure whether something is a bug, say so rather than asserting it is one
