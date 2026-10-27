# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
r65pkg - the R65 package tool.

The only thing that writes manifests, writes the lockfile, or runs git; see
docs/packaging-system.md.

    r65pkg init [--lib|--bin] [NAME]
    r65pkg add NAME --path DIR | --git URL [--version REQ | --tag T | --rev R | --branch B]
    r65pkg remove NAME
    r65pkg fetch | update [NAME...] | tree | list | check | path [--include-flags] | tag NAME
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from r65 import __version__ as R65_VERSION
from r65.pkg.lockfile import LOCK_FILENAME, read_lock, write_lock
from r65.pkg.manifest import (
    MANIFEST_FILENAME, Manifest, PackageError, discover_manifest,
    find_workspace_root, load_manifest,
)
from r65.pkg.resolve import DEPS_DIRNAME, Resolution, check_build, resolve
from r65.pkg.version import Requirement, format_version, parse_version

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_INCLUDE = re.compile(r'include!\(\s*"([^"]+)"\s*\)')


def compiler_version():
    return parse_version(R65_VERSION)


# ---------------------------------------------------------------------------
# Locating manifests

def _root(args) -> Manifest:
    """The package r65pkg acts on: --manifest, or the nearest r65.toml with
    a [package] at or above the current directory."""
    if args.manifest:
        return load_manifest(args.manifest)
    path = discover_manifest(Path.cwd())
    if path is None:
        raise PackageError(f"no {MANIFEST_FILENAME} here or in any parent directory "
                           f"(create one with `r65pkg init`)")
    manifest = load_manifest(path)
    if not manifest.is_package:
        raise PackageError(f"{path} is a workspace root, not a package: run r65pkg in a "
                           f"member's directory, or pass --manifest")
    return manifest


def _lock_path(root: Manifest) -> Path:
    return root.dir / LOCK_FILENAME


def _remote(root: Manifest, update: Optional[List[str]] = None, offline: bool = False):
    from r65.pkg.fetch import RemoteGit
    lock = _lock_path(root)
    locked = read_lock(lock) if lock.is_file() else {}
    return RemoteGit(root.dir / DEPS_DIRNAME, locked, update=update, offline=offline)


def _resolve_and_lock(root: Manifest, update: Optional[List[str]] = None,
                      offline: bool = False, quiet: bool = False) -> Resolution:
    resolution = resolve(root, _remote(root, update, offline))
    lock = _lock_path(root)
    old = read_lock(lock) if lock.is_file() else None
    new = {p.name: p.locked() for p in resolution.ordered()}
    if new != old and (new or old is not None):
        write_lock(lock, new)
        if not quiet:
            for name in sorted(set(new) | set(old or {})):
                was, now = (old or {}).get(name), new.get(name)
                if was != now:
                    if now is None:
                        print(f"  Removing {name}")
                    elif was is None:
                        print(f"    Locking {name} {format_version(now.version)}")
                    else:
                        print(f"   Updating {name} {format_version(was.version)} -> "
                              f"{format_version(now.version)}")
    return resolution


# ---------------------------------------------------------------------------
# Manifest editing: one inline-table line per dependency in [dependencies]

def _toml_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _edit_dependency(path: Path, name: str, line: Optional[str]) -> None:
    """Set (or with `line` None, delete) the `name = ...` line of the
    manifest's [dependencies] table, keeping everything else as written."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "[dependencies]"), None)
    if start is None:
        if line is None:
            raise PackageError(f"{path}: no [dependencies] table")
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", "[dependencies]", line]
    else:
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].lstrip().startswith("[")), len(lines))
        key = re.compile(rf'^\s*"?{re.escape(name)}"?\s*=')
        at = next((i for i in range(start + 1, end) if key.match(lines[i])), None)
        if at is not None and not lines[at].rstrip().endswith("}"):
            raise PackageError(f"{path}: '{name}' spans several lines; edit it by hand")
        if line is None:
            if at is None:
                raise PackageError(f"{path}: no dependency named '{name}'")
            del lines[at]
        elif at is not None:
            lines[at] = line
        else:
            last = max((i for i in range(start, end) if lines[i].strip()), default=start)
            lines.insert(last + 1, line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    load_manifest(path)  # what was written must still parse


# ---------------------------------------------------------------------------
# Commands

def cmd_init(args) -> None:
    path = Path.cwd() / MANIFEST_FILENAME
    if path.exists():
        raise PackageError(f"{path} already exists")
    name = args.name or Path.cwd().name
    if not _NAME.match(name):
        raise PackageError(f"'{name}' is not a valid package name "
                           f"(letters, digits, '_' and '-', not starting with a digit)")
    lines = ["[package]", f"name    = {_toml_string(name)}", 'version = "0.1.0"', ""]
    if args.lib:
        lines += ["[lib]", 'include_dir = "."']
    else:
        lines += ["[bin]", 'main = "src/main.r65"', "", "[dependencies]"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created {'library' if args.lib else 'application'} package '{name}' in {path}")


def cmd_add(args) -> None:
    root = _root(args)
    if not _NAME.match(args.name):
        raise PackageError(f"'{args.name}' is not a valid package name")
    parts = []
    if args.path:
        if args.git or args.tag or args.rev or args.branch:
            raise PackageError("--path takes no --git, --tag, --rev or --branch")
        parts.append(f"path = {_toml_string(args.path)}")
        if args.version:
            parts.append(f"version = {_toml_string(args.version)}")
    elif args.git:
        refs = [(k, getattr(args, k)) for k in ("version", "tag", "rev", "branch") if getattr(args, k)]
        if len(refs) > 1:
            raise PackageError("give at most one of --version, --tag, --rev and --branch")
        if not refs:
            # The newest release, as a caret requirement: "0.4" for 0.4.2.
            from r65.pkg.fetch import release_tags
            tags = release_tags(args.git, args.name)
            if not tags:
                raise PackageError(f"{args.git} has no {args.name}-vX.Y.Z release tags; "
                                   f"name a --tag, --rev or --branch")
            newest = max(tags)
            refs = [("version", caret_of(newest))]
        if refs[0][0] == "version":
            Requirement.parse(refs[0][1])
        parts += [f"git = {_toml_string(args.git)}", f"{refs[0][0]} = {_toml_string(refs[0][1])}"]
    else:
        raise PackageError("give --path DIR or --git URL")
    original = root.path.read_text(encoding="utf-8")
    _edit_dependency(root.path, args.name, f"{args.name} = {{ {', '.join(parts)} }}")
    try:
        resolution = _resolve_and_lock(load_manifest(root.path), offline=args.offline)
    except PackageError:
        root.path.write_text(original, encoding="utf-8")
        raise
    pkg = resolution.packages[args.name]
    print(f"Added {pkg.label} ({pkg.source})")


def caret_of(version) -> str:
    """The caret requirement that admits `version` and its compatible successors."""
    major, minor, patch = version
    if major:
        return f"{major}.{minor}"
    if minor:
        return f"0.{minor}"
    return f"0.0.{patch}"


def cmd_remove(args) -> None:
    root = _root(args)
    _edit_dependency(root.path, args.name, None)
    _resolve_and_lock(load_manifest(root.path), offline=True)
    print(f"Removed {args.name}")


def cmd_fetch(args) -> None:
    resolution = _resolve_and_lock(_root(args), offline=args.offline)
    n = len(resolution.packages)
    print(f"Fetched {n} package{'s' if n != 1 else ''}")


def cmd_update(args) -> None:
    root = _root(args)
    for name in args.names:
        if name not in _every_dependency_name(root):
            raise PackageError(f"'{name}' is not a dependency of {root.label}")
    _resolve_and_lock(root, update=list(args.names), offline=args.offline)


def _every_dependency_name(root: Manifest) -> set:
    lock = _lock_path(root)
    return set(root.dependencies) | (set(read_lock(lock)) if lock.is_file() else set())


def cmd_tree(args) -> None:
    resolution = resolve(_root(args))
    root = resolution.root
    print(root.label)

    def show(manifest: Manifest, prefix: str, trail: set) -> None:
        names = sorted(manifest.dependencies)
        for i, name in enumerate(names):
            pkg = resolution.packages[name]
            last = i == len(names) - 1
            again = " (*)" if name in trail else ""
            print(f"{prefix}{'└── ' if last else '├── '}{pkg.label} ({pkg.source}){again}")
            if not again:
                show(pkg.manifest, prefix + ("    " if last else "│   "), trail | {name})

    show(root, "", {root.name})


def cmd_list(args) -> None:
    for pkg in resolve(_root(args)).ordered():
        print(f"{pkg.name:<16} {format_version(pkg.version):<9} {pkg.source}\n"
              f"{'':<16} include: {pkg.include_dir}")


def cmd_path(args) -> None:
    dirs = [str(d) for d in resolve(_root(args)).include_dirs()]
    print(" ".join(f"-I {d}" for d in dirs) if args.include_flags else "\n".join(dirs))


def cmd_check(args) -> None:
    root = _root(args)
    problems: List[str] = []
    warnings: List[str] = []
    workspace = find_workspace_root(root)
    if workspace is not None:
        for member, directory in zip(workspace.members, workspace.member_dirs()):
            if not (directory / MANIFEST_FILENAME).is_file():
                warnings.append(f"{workspace.path}: workspace member '{member}' has no "
                                f"{MANIFEST_FILENAME}")
    try:
        resolution = resolve(root)
        # A library checked alone has no build of its own: take the
        # conditions it requires as given.
        cfg = root.bin_cfg if root.bin_cfg is not None else list(root.cfg)
        check_build(resolution, compiler_version(), cfg)
    except PackageError as e:
        problems.append(str(e))
        resolution = None
    if resolution is not None:
        problems += _ambiguous_files(resolution)
        warnings += _bare_cross_package_includes(resolution)
    for w in warnings:
        print(f"warning: {w}")
    for p in problems:
        print(f"error: {p}")
    if problems:
        sys.exit(1)
    print(f"{root.label}: ok ({len(resolution.packages)} package"
          f"{'s' if len(resolution.packages) != 1 else ''})")


def _package_files(pkg) -> Dict[str, Path]:
    return {f.relative_to(pkg.include_dir).as_posix(): f
            for f in pkg.include_dir.rglob("*.r65")
            if f.is_file() and ".r65" not in f.relative_to(pkg.include_dir).parts[:-1]}


def _ambiguous_files(resolution: Resolution) -> List[str]:
    """Files two packages both provide: a bare include of one is an error."""
    owners: Dict[str, List[str]] = {}
    for pkg in resolution.ordered():
        for rel in _package_files(pkg):
            owners.setdefault(rel, []).append(pkg.label)
    return [f"'{rel}' is provided by {' and '.join(who)}: include it qualified"
            for rel, who in sorted(owners.items()) if len(who) > 1]


def _bare_cross_package_includes(resolution: Resolution) -> List[str]:
    """A package including another package's file by bare name, which only
    works while nothing else provides that name."""
    found = []
    files = {p.name: _package_files(p) for p in resolution.ordered()}
    for pkg in resolution.ordered():
        for rel, path in files[pkg.name].items():
            text = path.read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("//"):
                    continue
                for target in _INCLUDE.findall(line):
                    if (path.parent / target).is_file():
                        continue
                    first = target.split("/", 1)[0]
                    if first in files and "/" in target:
                        continue
                    owners = [o for o, fs in files.items() if o != pkg.name and target in fs]
                    if owners:
                        found.append(f"{path}:{n}: include!(\"{target}\") reaches {owners[0]} by "
                                     f"bare name; write include!(\"{owners[0]}/{target}\")")
    return found


def cmd_tag(args) -> None:
    from r65.pkg.fetch import git
    start = load_manifest(args.manifest) if args.manifest else None
    if start is None:
        path = discover_manifest(Path.cwd())
        if path is None:
            raise PackageError(f"no {MANIFEST_FILENAME} here or above")
        start = load_manifest(path)
    workspace = find_workspace_root(start) if start.is_package else start
    candidates = [start] if start.name == args.name else []
    if workspace is not None:
        for directory in workspace.member_dirs():
            path = directory / MANIFEST_FILENAME
            if path.is_file() and load_manifest(path).name == args.name:
                candidates.append(load_manifest(path))
    if not candidates:
        raise PackageError(f"no package named '{args.name}' here or in the workspace")
    member = candidates[0]
    tag = f"{member.name}-v{format_version(member.version)}"
    top = Path(git("rev-parse", "--show-toplevel", cwd=member.dir).strip())
    rel = member.path.relative_to(top.resolve()).as_posix()
    committed = git("show", f"HEAD:{rel}", cwd=top)
    if committed != member.path.read_text(encoding="utf-8"):
        raise PackageError(f"{member.path} differs from HEAD: commit it before tagging, so "
                           f"the tag releases the version it names")
    if git("tag", "--list", tag, cwd=top).strip():
        raise PackageError(f"tag {tag} already exists: bump [package] version first")
    git("tag", "-a", tag, "-m", f"{member.name} {format_version(member.version)}", cwd=top)
    print(f"Tagged {tag} (push it with: git push origin {tag})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="r65pkg", description="R65 package tool")
    parser.add_argument("--manifest", metavar="PATH", help=f"use this {MANIFEST_FILENAME}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help=f"write an {MANIFEST_FILENAME} in the current directory")
    kind = p.add_mutually_exclusive_group()
    kind.add_argument("--lib", action="store_true", help="a library others include")
    kind.add_argument("--bin", action="store_true", help="an application (the default)")
    p.add_argument("name", nargs="?")
    p.set_defaults(run=cmd_init)

    p = sub.add_parser("add", help="add a dependency and lock it")
    p.add_argument("name")
    p.add_argument("--path")
    p.add_argument("--git", metavar="URL")
    p.add_argument("--version", metavar="REQ")
    p.add_argument("--tag")
    p.add_argument("--rev")
    p.add_argument("--branch")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(run=cmd_add)

    p = sub.add_parser("remove", help="drop a dependency and its lock entry")
    p.add_argument("name")
    p.set_defaults(run=cmd_remove)

    p = sub.add_parser("fetch", help=f"lock any new dependencies and clone into {DEPS_DIRNAME}/")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(run=cmd_fetch)

    p = sub.add_parser("update", help="re-resolve to the newest versions allowed")
    p.add_argument("names", nargs="*", metavar="NAME")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(run=cmd_update)

    p = sub.add_parser("tree", help="print the dependency graph")
    p.set_defaults(run=cmd_tree)
    p = sub.add_parser("list", help="resolved packages: version, source, include dir")
    p.set_defaults(run=cmd_list)
    p = sub.add_parser("check", help="validate without compiling")
    p.set_defaults(run=cmd_check)

    p = sub.add_parser("path", help="print resolved include dirs")
    p.add_argument("--include-flags", action="store_true", help="as -I flags")
    p.set_defaults(run=cmd_path)

    p = sub.add_parser("tag", help="tag NAME-v<version> from a member's manifest")
    p.add_argument("name")
    p.set_defaults(run=cmd_tag)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except PackageError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
