#!/bin/bash
# Setup git hooks for the R65 project

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/scripts/githooks"

echo "Configuring git to use hooks from scripts/githooks/..."
git config core.hooksPath "$HOOKS_DIR"

echo "Ensuring hooks are executable..."
chmod +x "$HOOKS_DIR"/*

echo "Done. Git hooks installed."
