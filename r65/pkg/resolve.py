# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Dependency resolution: from a root manifest to one version of each package
and the ordered include directories they provide.

Every package name unifies to exactly one version - textual inclusion into a
flat namespace cannot hold two. A path dependency or a root `[patch]` names
its directory; a git dependency's version comes from a `GitSource`:

- `LockedGit` (here, read-only) takes it from `r65.lock` and the clone under
  `.r65/deps/`, which is what `r65c` uses - it never touches the network;
- `r65.pkg.fetch.RemoteGit` lists the repository's `NAME-vX.Y.Z` tags and
  clones the one chosen, which is what `r65pkg` uses.

The walk is repeated until the chosen packages stop changing, since a package
chosen in one round adds the requirements of its own manifest in the next.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from r65.pkg.lockfile import LOCK_FILENAME, LockedPackage, read_lock
from r65.pkg.manifest import (
    MANIFEST_FILENAME, Dependency, Manifest, PackageError, find_workspace_root,
    load_manifest,
)
from r65.pkg.version import Version, at_least, format_version

DEPS_DIRNAME = ".r65/deps"

# A requirement on a package: who asks for it, and how.
Edge = Tuple[str, Dependency]


@dataclass
class ResolvedPackage:
    name: str
    version: Version
    manifest: Manifest
    source: str                    # the lockfile's source string
    rev: Optional[str] = None
    requirers: List[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return self.manifest.dir

    @property
    def include_dir(self) -> Path:
        return (self.manifest.dir / self.manifest.include_dir).resolve()

    @property
    def label(self) -> str:
        return f"{self.name} {format_version(self.version)}"

    def locked(self) -> LockedPackage:
        return LockedPackage(self.name, self.version, self.source, self.rev)


@dataclass
class Resolution:
    root: Manifest                 # the package being built
    packages: Dict[str, ResolvedPackage]
    order: List[str]               # dependencies first, then alphabetical
    edges: Dict[str, List[Edge]] = field(default_factory=dict)

    def ordered(self) -> List[ResolvedPackage]:
        return [self.packages[n] for n in self.order]

    def include_dirs(self) -> List[Path]:
        seen, dirs = set(), []
        for pkg in self.ordered():
            if pkg.include_dir not in seen:
                seen.add(pkg.include_dir)
                dirs.append(pkg.include_dir)
        return dirs


class GitSource(Protocol):
    def choose(self, name: str, url: str, edges: List[Edge]) -> ResolvedPackage: ...


def same_repository(a: str, b: str) -> bool:
    def norm(url: str) -> str:
        url = url.strip().rstrip("/")
        return url[:-4] if url.endswith(".git") else url
    return norm(a) == norm(b)


def find_member(repo_dir: Path, name: str) -> Manifest:
    """The package called `name` in a repository checkout: a member of its
    root [workspace], or the root itself."""
    root_path = repo_dir / MANIFEST_FILENAME
    if not root_path.is_file():
        raise PackageError(f"{repo_dir}: the repository has no {MANIFEST_FILENAME} at its root")
    root = load_manifest(root_path)
    if root.name == name:
        return root
    for member in root.member_dirs():
        path = member / MANIFEST_FILENAME
        if path.is_file():
            manifest = load_manifest(path)
            if manifest.name == name:
                return manifest
    where = "its [workspace] members" if root.members is not None else "its root package"
    raise PackageError(f"{repo_dir}: no package named '{name}' among {where}")


def _requirement_lines(edges: List[Edge]) -> str:
    return "\n".join(f"  {who} requires {dep.describe()}" for who, dep in edges)


def check_requirements(name: str, version: Version, edges: List[Edge], what: str) -> None:
    """Every range on `name` admits the version chosen for it."""
    bad = [(who, dep) for who, dep in edges if dep.version and not dep.version.matches(version)]
    if bad:
        raise PackageError(
            f"{what} {name} {format_version(version)} does not satisfy every requirement on it\n"
            + _requirement_lines(edges))


class LockedGit:
    """Git packages as `r65.lock` pins them and `r65pkg fetch` cloned them."""

    def __init__(self, root_dir: Path):
        self.lock_path = root_dir / LOCK_FILENAME
        self.deps_dir = root_dir / DEPS_DIRNAME
        self.lock = read_lock(self.lock_path) if self.lock_path.is_file() else None

    def choose(self, name: str, url: str, edges: List[Edge]) -> ResolvedPackage:
        if self.lock is None:
            raise PackageError(f"{self.lock_path.parent}: no {LOCK_FILENAME} for the git "
                               f"dependency '{name}': run `r65pkg fetch`")
        entry = self.lock.get(name)
        if entry is None or not entry.is_git:
            raise PackageError(f"{self.lock_path}: '{name}' is not locked: run `r65pkg fetch`")
        stale = not same_repository(entry.git_url, url)
        for _, dep in edges:
            if dep.version and not dep.version.matches(entry.version):
                stale = True
            ref = dep.exact_ref
            if ref and ref != entry.ref and not (ref[0] == "rev" and entry.rev.startswith(ref[1])):
                stale = True
        if stale:
            raise PackageError(
                f"{self.lock_path}: '{name}' is locked to {format_version(entry.version)} "
                f"({entry.source}), which no longer matches\n{_requirement_lines(edges)}\n"
                f"  run `r65pkg update {name}`")
        clone = self.deps_dir / name
        head = clone / ".git" / "HEAD"
        # With no rev the package is a copy from the installed r65, not a
        # clone: its manifest's version is all there is to check.
        fetched = (head.is_file() and head.read_text(encoding="utf-8").strip() == entry.rev
                   if entry.rev else (clone / MANIFEST_FILENAME).is_file())
        if not fetched:
            raise PackageError(f"{clone}: '{name}' {format_version(entry.version)} is not "
                               f"fetched: run `r65pkg fetch`")
        manifest = find_member(clone, name)
        if manifest.version != entry.version:
            raise PackageError(f"{manifest.path}: version {format_version(manifest.version)} "
                               f"but {LOCK_FILENAME} says {format_version(entry.version)}: "
                               f"run `r65pkg fetch`")
        return ResolvedPackage(name, entry.version, manifest, entry.source, entry.rev)


def _path_source(root_dir: Path, directory: Path) -> str:
    import os
    return "path+" + Path(os.path.relpath(directory, root_dir)).as_posix()


def resolve(root: Manifest, git: Optional[GitSource] = None) -> Resolution:
    """Resolve `root`'s dependency graph. `git` defaults to `LockedGit`."""
    if not root.is_package:
        return Resolution(root, {}, [])
    workspace = find_workspace_root(root)
    patches = (workspace or root).patch
    git = git or LockedGit(root.dir)
    loaded: Dict[Path, Manifest] = {}

    def manifest_at(directory: Path, name: str, why: str) -> Manifest:
        if directory not in loaded:
            path = directory / MANIFEST_FILENAME
            if not path.is_file():
                raise PackageError(f"{why}: no {MANIFEST_FILENAME} in {directory}")
            loaded[directory] = load_manifest(path)
        manifest = loaded[directory]
        if manifest.name != name:
            raise PackageError(f"{why}: {manifest.path} is package '{manifest.name}', not '{name}'")
        return manifest

    def walk(chosen: Dict[str, ResolvedPackage]) -> Dict[str, List[Edge]]:
        edges: Dict[str, List[Edge]] = {}
        queue, seen = [root], {root.name}
        while queue:
            manifest = queue.pop(0)
            for name, dep in manifest.dependencies.items():
                edges.setdefault(name, []).append((manifest.label, dep))
                if name in chosen and name not in seen:
                    seen.add(name)
                    queue.append(chosen[name].manifest)
        return edges

    def choose(name: str, edges: List[Edge]) -> ResolvedPackage:
        if name == root.name:
            raise PackageError(f"'{name}' depends on itself\n{_requirement_lines(edges)}")
        if name in patches:
            patch = patches[name]
            manifest = manifest_at(patch.path_dir(), name, f"[patch] {name}")
            check_requirements(name, manifest.version, edges, "patched")
            pkg = ResolvedPackage(name, manifest.version, manifest,
                                  _path_source(root.dir, manifest.dir))
        else:
            kinds = {dep.is_git for _, dep in edges}
            if len(kinds) > 1:
                raise PackageError(f"'{name}' is required both by path and from git; "
                                   f"unify them, or `[patch]` it\n{_requirement_lines(edges)}")
            if kinds == {False}:
                dirs = {dep.path_dir() for _, dep in edges}
                if len(dirs) > 1:
                    raise PackageError(f"'{name}' is required from two directories\n"
                                       f"{_requirement_lines(edges)}")
                manifest = manifest_at(dirs.pop(), name, f"dependency '{name}'")
                check_requirements(name, manifest.version, edges, "path package")
                pkg = ResolvedPackage(name, manifest.version, manifest,
                                      _path_source(root.dir, manifest.dir))
            else:
                urls = [dep.git for _, dep in edges]
                if any(not same_repository(u, urls[0]) for u in urls):
                    raise PackageError(f"'{name}' is required from two repositories\n"
                                       f"{_requirement_lines(edges)}")
                pkg = git.choose(name, urls[0], edges)
        if pkg.manifest.include_dir is None:
            raise PackageError(f"{pkg.manifest.path}: '{name}' is not a library (it has no "
                               f"[lib] table), so nothing can depend on it")
        pkg.requirers = sorted({who for who, _ in edges})
        return pkg

    chosen: Dict[str, ResolvedPackage] = {}
    for _ in range(64):
        edges = walk(chosen)
        fresh = {name: (chosen[name] if name in chosen and _unchanged(chosen[name], e)
                        else choose(name, e))
                 for name, e in edges.items()}
        if _key(fresh) == _key(chosen):
            return Resolution(root, fresh, _order(fresh), edges)
        chosen = fresh
    raise PackageError(f"{root.path}: dependency resolution did not settle")


def _unchanged(pkg: ResolvedPackage, edges: List[Edge]) -> bool:
    """A package already chosen stays chosen while its requirements still hold
    and come from the same requirers."""
    return (sorted({who for who, _ in edges}) == pkg.requirers
            and all(not dep.version or dep.version.matches(pkg.version) for _, dep in edges))


def _key(chosen: Dict[str, ResolvedPackage]):
    return sorted((n, p.version, str(p.dir)) for n, p in chosen.items())


def _order(packages: Dict[str, ResolvedPackage]) -> List[str]:
    """Dependencies before their dependents; ties alphabetical."""
    deps = {n: {d for d in p.manifest.dependencies if d in packages} for n, p in packages.items()}
    order: List[str] = []
    while deps:
        ready = sorted(n for n, ds in deps.items() if not ds - set(order))
        if not ready:
            raise PackageError(f"dependency cycle among: {', '.join(sorted(deps))}")
        order.append(ready[0])
        del deps[ready[0]]
    return order


def check_build(resolution: Resolution, compiler_version: Version, cfg: List[str]) -> None:
    """The build requirements every package states: the oldest r65c it
    builds with, and the cfg conditions it must be built with."""
    from r65.pkg.version import format_version as fv
    for manifest in [resolution.root] + [p.manifest for p in resolution.ordered()]:
        if manifest.r65_version and not at_least(compiler_version, manifest.r65_version):
            raise PackageError(f"{manifest.label} requires r65c >= {manifest.r65_version}; "
                               f"this is r65c {fv(compiler_version)}")
        missing = [c for c in manifest.cfg if c not in cfg]
        if missing:
            raise PackageError(f"{manifest.label} must be built with "
                               f"{' '.join('--cfg ' + c for c in missing)}")
