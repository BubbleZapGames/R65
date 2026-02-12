---
description: "Creates well-formatted commits with conventional commit messages"
allowed-tools:
  [
    "Bash(git add:*)",
    "Bash(git status:*)",
    "Bash(git commit:*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
  ]
---

# Claude Command: Commit

Creates well-formatted commits with conventional commit messages.

## Usage

```
/commit
/commit --no-verify
```

## Process

1. Check staged files and auto-stages if none are found
2. Analyze diff for multiple logical changes
3. Suggest splitting if needed
4. Creates conventional format commit messages
5. Preforms the commit with proper formatting

## Best Practices
- Make atomic commits that address a single concern
- Use appropriate type prefixes (feat, fix, docs, etc)
- Write clear, consise descriptions in the imperative mood

## Notes
- Only commit staged files if any exist
- **NEVER add Claude signature to commits**
