# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
The `r65.toml` manifest: reading, validating and discovering it.

    [package]   name, version, r65-version, cfg, repository
    [lib]       include_dir
    [bin]       main, cfg
    [dependencies]  NAME = { path = ".." } | { git = "..", version | tag | rev | branch = ".." }
    [workspace] members
    [patch]     NAME = { path = ".." }

Unknown keys are errors, so a misspelt key never silently does nothing.
Read-only: nothing here writes a file or runs git.
"""

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from r65.pkg.version import Requirement, VersionError, parse_version, Version, at_least

MANIFEST_FILENAME = "r65.toml"

_TOP_KEYS = {"package", "lib", "bin", "dependencies", "workspace", "patch"}
_PACKAGE_KEYS = {"name", "version", "r65-version", "cfg", "repository"}
_LIB_KEYS = {"include_dir"}
_BIN_KEYS = {"main", "cfg"}
_WORKSPACE_KEYS = {"members"}
_DEP_KEYS = {"path", "git", "version", "tag", "rev", "branch"}
_GIT_REFS = ("tag", "rev", "branch")


class PackageError(Exception):
    """A manifest, lockfile or resolution problem, reported to the user as is."""


@dataclass(frozen=True)
class Dependency:
    """One `[dependencies]` (or `[patch]`) entry. `base` is the directory
    holding the manifest that declares it, which a `path` is relative to."""
    name: str
    base: Path
    path: Optional[str] = None
    git: Optional[str] = None
    version: Optional[Requirement] = None
    tag: Optional[str] = None
    rev: Optional[str] = None
    branch: Optional[str] = None

    @property
    def is_git(self) -> bool:
        return self.git is not None

    @property
    def exact_ref(self) -> Optional[tuple]:
        """(kind, value) for a `tag`, `rev` or `branch` pin, else None."""
        for kind in _GIT_REFS:
            value = getattr(self, kind)
            if value is not None:
                return kind, value
        return None

    def path_dir(self) -> Path:
        return (self.base / self.path).resolve()

    def describe(self) -> str:
        if self.path is not None:
            return f'path "{self.path}"'
        ref = self.exact_ref
        what = f'{ref[0]} "{ref[1]}"' if ref else f'version "{self.version}"'
        return f'{what} from {self.git}'


@dataclass
class Manifest:
    path: Path                      # the r65.toml file
    name: Optional[str] = None      # None: a workspace root that is not a package
    version: Optional[Version] = None
    r65_version: Optional[str] = None
    cfg: List[str] = field(default_factory=list)
    repository: Optional[str] = None
    include_dir: Optional[str] = None   # [lib]; None: not a library
    bin_main: Optional[str] = None
    bin_cfg: Optional[List[str]] = None
    dependencies: Dict[str, Dependency] = field(default_factory=dict)
    members: Optional[List[str]] = None  # [workspace]
    patch: Dict[str, Dependency] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def is_package(self) -> bool:
        return self.name is not None

    @property
    def label(self) -> str:
        from r65.pkg.version import format_version
        return f"{self.name} {format_version(self.version)}"

    def member_dirs(self) -> List[Path]:
        return [(self.dir / m).resolve() for m in (self.members or [])]


def _fail(path: Path, message: str):
    raise PackageError(f"{path}: {message}")


def _check_keys(path: Path, table: dict, allowed: set, where: str):
    if not isinstance(table, dict):
        _fail(path, f"[{where}] must be a table")
    unknown = sorted(set(table) - allowed)
    if unknown:
        _fail(path, f"unknown key{'s' if len(unknown) > 1 else ''} in [{where}]: "
                    f"{', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})")


def _string_list(path: Path, value, where: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _fail(path, f"{where} must be a list of strings")
    return list(value)


def _string(path: Path, value, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, f"{where} must be a non-empty string")
    return value


def _dependency(path: Path, name: str, spec, where: str, patch: bool) -> Dependency:
    if isinstance(spec, str):
        _fail(path, f'{where}.{name}: a bare version needs a source - write '
                    f'{name} = {{ git = "...", version = "{spec}" }}')
    _check_keys(path, spec, _DEP_KEYS, f"{where}.{name}")
    kw = {k: _string(path, spec[k], f"{where}.{name}.{k}") for k in spec}
    has_path, has_git = "path" in kw, "git" in kw
    if has_path == has_git:
        _fail(path, f"{where}.{name}: give exactly one of `path` or `git`")
    refs = [k for k in ("version",) + _GIT_REFS if k in kw]
    if has_path and any(k in kw for k in _GIT_REFS):
        _fail(path, f"{where}.{name}: `{refs[0]}` needs `git`, not `path`")
    if has_git and len(refs) != 1:
        _fail(path, f"{where}.{name}: a git dependency takes exactly one of "
                    f"`version`, `tag`, `rev` or `branch`")
    if patch and has_git:
        _fail(path, f"{where}.{name}: a patch must be a `path`")
    version = None
    if "version" in kw:
        try:
            version = Requirement.parse(kw.pop("version"))
        except VersionError as e:
            _fail(path, f"{where}.{name}: {e}")
    return Dependency(name=name, base=path.parent, version=version, **kw)


def load_manifest(path) -> Manifest:
    path = Path(path).resolve()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise PackageError(f"{path}: cannot read: {e.strerror or e}")
    except tomllib.TOMLDecodeError as e:
        raise PackageError(f"{path}: invalid TOML: {e}")
    _check_keys(path, data, _TOP_KEYS, "top level")
    if "package" not in data and "workspace" not in data:
        _fail(path, "needs a [package] or a [workspace] table")
    m = Manifest(path=path)

    if "package" in data:
        pkg = data["package"]
        _check_keys(path, pkg, _PACKAGE_KEYS, "package")
        for key in ("name", "version"):
            if key not in pkg:
                _fail(path, f"[package] needs `{key}`")
        m.name = _string(path, pkg["name"], "package.name")
        try:
            m.version = parse_version(_string(path, pkg["version"], "package.version"))
            if "r65-version" in pkg:
                m.r65_version = _string(path, pkg["r65-version"], "package.r65-version")
                at_least((0, 0, 0), m.r65_version)
        except VersionError as e:
            _fail(path, f"[package] {e}")
        if "cfg" in pkg:
            m.cfg = _string_list(path, pkg["cfg"], "package.cfg")
        if "repository" in pkg:
            m.repository = _string(path, pkg["repository"], "package.repository")
    for table in ("lib", "bin", "dependencies"):
        if table in data and "package" not in data:
            _fail(path, f"[{table}] needs a [package]")

    if "lib" in data:
        _check_keys(path, data["lib"], _LIB_KEYS, "lib")
        m.include_dir = _string(path, data["lib"].get("include_dir", "."), "lib.include_dir")
        if not (m.dir / m.include_dir).is_dir():
            _fail(path, f"lib.include_dir '{m.include_dir}' is not a directory")
    if "bin" in data:
        _check_keys(path, data["bin"], _BIN_KEYS, "bin")
        if "main" in data["bin"]:
            m.bin_main = _string(path, data["bin"]["main"], "bin.main")
        if "cfg" in data["bin"]:
            m.bin_cfg = _string_list(path, data["bin"]["cfg"], "bin.cfg")
            if any(c.split("=", 1)[0].strip() == "abi" for c in m.bin_cfg):
                _fail(path, "bin.cfg cannot set 'abi'; the ABI is chosen with --abi")
    for name, spec in data.get("dependencies", {}).items():
        if name == m.name:
            _fail(path, f"[dependencies]: '{name}' cannot depend on itself")
        m.dependencies[name] = _dependency(path, name, spec, "dependencies", patch=False)

    if "workspace" in data:
        _check_keys(path, data["workspace"], _WORKSPACE_KEYS, "workspace")
        m.members = _string_list(path, data["workspace"].get("members", []), "workspace.members")
    for name, spec in data.get("patch", {}).items():
        m.patch[name] = _dependency(path, name, spec, "patch", patch=True)
    return m


def discover_manifest(start) -> Optional[Path]:
    """The nearest `r65.toml` at or above `start` (a file or a directory)."""
    here = Path(start).resolve()
    if not here.is_dir():
        here = here.parent
    for directory in (here, *here.parents):
        candidate = directory / MANIFEST_FILENAME
        if candidate.is_file():
            return candidate
    return None


def find_workspace_root(manifest: Manifest) -> Optional[Manifest]:
    """The workspace that lists this package as a member: the manifest itself
    when it has a [workspace], else the nearest one above it that does."""
    if manifest.members is not None:
        return manifest
    for directory in manifest.dir.parents:
        candidate = directory / MANIFEST_FILENAME
        if candidate.is_file():
            root = load_manifest(candidate)
            if root.members is not None and manifest.dir in root.member_dirs():
                return root
    return None
