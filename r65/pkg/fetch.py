# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Git for `r65pkg`: listing a repository's release tags, cloning the commit a
package resolves to into `.r65/deps/<name>/`, and pinning it.

Only the `r65pkg` tool imports this; `r65c` never runs git.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from r65.pkg.lockfile import LockedPackage, git_source
from r65.pkg.manifest import MANIFEST_FILENAME, PackageError, load_manifest
from r65.pkg.resolve import Edge, ResolvedPackage, find_member, same_repository
from r65.pkg.version import Version, format_version, parse_version, VersionError


def git(*args, cwd: Optional[Path] = None) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        raise PackageError("git is not installed (r65pkg needs it for git dependencies)")
    if done.returncode != 0:
        raise PackageError(f"git {' '.join(args)} failed:\n  {done.stderr.strip()}")
    return done.stdout


def _clone_url(url: str) -> str:
    """A local repository path as a file:// URL, so shallow clones work."""
    if "://" in url or re.match(r"^[\w.-]+@[\w.-]+:", url):
        return url
    return Path(url).expanduser().resolve().as_uri()


def _remote_tags(url: str, pattern: str) -> Dict[str, str]:
    """The repository's tags matching `pattern`: tag -> commit."""
    out = git("ls-remote", "--tags", _clone_url(url), f"refs/tags/{pattern}")
    commits: Dict[str, str] = {}
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        tag = ref[len("refs/tags/"):]
        if tag.endswith("^{}"):           # an annotated tag, peeled to its commit
            commits[tag[:-3]] = sha
        else:
            commits.setdefault(tag, sha)
    return commits


def release_tags(url: str, name: str) -> Dict[Version, Tuple[str, str]]:
    """The repository's `NAME-vX.Y.Z` tags: version -> (tag, commit)."""
    tags = {}
    for tag, sha in _remote_tags(url, f"{name}-v*").items():
        try:
            tags[parse_version(tag[len(name) + 2:])] = (tag, sha)
        except VersionError:
            continue
    return tags


def released_packages(url: str) -> List[str]:
    """The names the repository has `NAME-vX.Y.Z` release tags for."""
    names = set()
    for tag in _remote_tags(url, "*-v*"):
        name, _, version = tag.rpartition("-v")
        try:
            parse_version(version)
        except VersionError:
            continue
        names.add(name)
    return sorted(names)


def packages_at(url: str, kind: str, value: str) -> List[str]:
    """The packages in the repository at a tag, branch or commit: its root
    package, or the packages among its [workspace] members."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "repo"
        checkout(url, kind, value, dest)
        root_path = dest / MANIFEST_FILENAME
        if not root_path.is_file():
            raise PackageError(f"{url} has no {MANIFEST_FILENAME} at its root")
        root = load_manifest(root_path)
        if root.is_package:
            return [root.name]
        members = (load_manifest(d / MANIFEST_FILENAME) for d in root.member_dirs()
                   if (d / MANIFEST_FILENAME).is_file())
        return sorted(m.name for m in members if m.is_package)


def _head(dest: Path) -> Optional[str]:
    if not (dest / ".git").is_dir():
        return None
    try:
        return git("rev-parse", "HEAD", cwd=dest).strip()
    except PackageError:
        return None


def checkout(url: str, kind: str, value: str, dest: Path, rev: Optional[str] = None) -> str:
    """Clone `url` at a tag, branch or commit (`kind` "rev") into `dest`,
    detached, and return the commit. With `rev`, the checkout must end at that
    commit - a moved tag or an advanced branch falls back to fetching it - and
    one already there is kept."""
    want = rev or (value if kind == "rev" else None)
    head = _head(dest)
    if want and head and head.startswith(want):
        return head
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = _clone_url(url)
    if kind in ("tag", "branch"):
        git("clone", "--quiet", "--depth", "1", "--branch", value, source, str(dest))
        git("checkout", "--quiet", "--detach", cwd=dest)
        head = _head(dest)
        if not want or head.startswith(want):
            return head
        shutil.rmtree(dest)
    git("init", "--quiet", str(dest))
    try:
        git("fetch", "--quiet", "--depth", "1", source, want, cwd=dest)
        git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=dest)
    except PackageError:
        # Servers that refuse a bare commit get a full fetch.
        git("fetch", "--quiet", "--tags", source, "+refs/heads/*:refs/remotes/origin/*", cwd=dest)
        git("checkout", "--quiet", "--detach", want, cwd=dest)
    return _head(dest)


class RemoteGit:
    """Git packages resolved against their repositories. A package locked
    already keeps its locked version while it still satisfies every
    requirement, unless it is being updated."""

    def __init__(self, deps_dir: Path, locked: Dict[str, LockedPackage],
                 update: Optional[List[str]] = None, offline: bool = False):
        self.deps_dir = deps_dir
        self.locked = locked
        self.update_all = update == []
        self.update = set(update or [])
        self.offline = offline

    def _keep_locked(self, name: str) -> bool:
        return not self.update_all and name not in self.update and name in self.locked

    def choose(self, name: str, url: str, edges: List[Edge]) -> ResolvedPackage:
        pins = {dep.exact_ref for _, dep in edges if dep.exact_ref}
        ranges = [dep.version for _, dep in edges if dep.version]
        if len(pins) > 1:
            raise self._conflict(name, edges, "two different exact commits")
        locked = self.locked.get(name)
        if (locked and locked.is_git and same_repository(locked.git_url, url)
                and self._keep_locked(name)
                and all(r.matches(locked.version) for r in ranges)
                and (not pins or next(iter(pins)) == locked.ref)):
            kind, value = locked.ref
            if locked.rev is None:
                copy = self._installed_copy(name, locked.version)
                if copy is not None:
                    return copy
            return self._materialize(name, url, kind, value, locked.version, locked.rev)
        if self.offline:
            raise PackageError(f"'{name}' needs resolving against {url}, but --offline was given")
        if pins:
            kind, value = next(iter(pins))
            return self._materialize(name, url, kind, value, None, None, edges)
        tags = release_tags(url, name)
        fits = sorted(v for v in tags if all(r.matches(v) for r in ranges))
        if not fits:
            have = ", ".join(format_version(v) for v in sorted(tags)) or "none"
            raise self._conflict(name, edges, f"no release satisfies them all "
                                              f"(tagged releases of '{name}': {have})")
        version = fits[-1]
        tag, _ = tags[version]
        return self._materialize(name, url, "tag", tag, version, None)

    def _installed_copy(self, name: str, version: Version) -> Optional[ResolvedPackage]:
        """The copy `r65x init` put in place of a clone, if it is still there at
        the locked version; it stays unpinned (no rev) until replaced."""
        dest = self.deps_dir / name
        if (dest / ".git").exists() or not (dest / "r65.toml").is_file():
            return None
        manifest = find_member(dest, name)
        if manifest.version != version:
            return None
        locked = self.locked[name]
        return ResolvedPackage(name, version, manifest, locked.source, None)

    def _materialize(self, name, url, kind, value, version, rev, edges=None) -> ResolvedPackage:
        dest = self.deps_dir / name
        head = checkout(url, kind, value, dest, rev)
        manifest = find_member(dest, name)
        if kind == "tag" and value.startswith(f"{name}-v"):
            # A release tag must agree with the manifest it releases.
            claimed = value[len(name) + 2:]
            if format_version(manifest.version) != claimed:
                raise PackageError(f"tag {value} of {url} holds {name} "
                                   f"{format_version(manifest.version)}, not {claimed}: "
                                   f"the tag disagrees with its manifest")
        if version is not None and manifest.version != version:
            raise PackageError(f"{manifest.path}: {name} {format_version(manifest.version)} "
                               f"at {value}, expected {format_version(version)}")
        if edges:
            bad = [dep for _, dep in edges if dep.version and not dep.version.matches(manifest.version)]
            if bad:
                raise self._conflict(name, edges, f"{value} holds "
                                                  f"{format_version(manifest.version)}")
        return ResolvedPackage(name, manifest.version, manifest,
                               git_source(url, kind, value), head)

    @staticmethod
    def _conflict(name: str, edges: List[Edge], why: str) -> PackageError:
        lines = "\n".join(f"  {who} requires {dep.describe()}" for who, dep in edges)
        return PackageError(f"cannot resolve '{name}': {why}\n{lines}\n"
                            f"  note: raise a requirement, or `[patch]` '{name}' in the root manifest")
