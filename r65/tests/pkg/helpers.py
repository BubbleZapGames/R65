# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Package trees and git repositories for the r65.pkg tests."""

import subprocess
from pathlib import Path


def package(directory: Path, name: str, version: str = "0.1.0", deps: str = "",
            lib: bool = True, extra: str = "", files: dict = None) -> Path:
    """Write a package: an r65.toml (with [lib] unless `lib` is False) and
    `files` ({relative path: text}) under it."""
    directory.mkdir(parents=True, exist_ok=True)
    text = f'[package]\nname = "{name}"\nversion = "{version}"\n{extra}\n'
    if lib:
        text += '[lib]\ninclude_dir = "."\n'
    if deps:
        text += f"[dependencies]\n{deps}\n"
    (directory / "r65.toml").write_text(text)
    for rel, body in (files or {}).items():
        (directory / rel).parent.mkdir(parents=True, exist_ok=True)
        (directory / rel).write_text(body)
    return directory / "r65.toml"


def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout


def repository(directory: Path, members: list) -> Path:
    """An empty git repository whose root r65.toml is a workspace of `members`."""
    directory.mkdir(parents=True)
    git(directory, "init", "-q", "-b", "main")
    (directory / "r65.toml").write_text(
        "[workspace]\nmembers = [" + ", ".join(f'"{m}"' for m in members) + "]\n")
    return directory


def release(repo: Path, member: str, version: str, deps: str = "", files: dict = None,
            tag: str = None) -> str:
    """Commit `member` at `version` and tag it `member-v<version>` (or `tag`);
    returns the commit."""
    package(repo / member, member, version, deps,
            files=files or {f"{member}.r65": f"// {member} {version}\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"{member} {version}")
    git(repo, "tag", "-a", tag or f"{member}-v{version}", "-m", version)
    return git(repo, "rev-parse", "HEAD").strip()
