# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""r65.toml parsing and version requirements (docs/packaging-system.md)."""

import pytest

from r65.pkg.manifest import PackageError, discover_manifest, find_workspace_root, load_manifest
from r65.pkg.version import Requirement, VersionError, at_least, parse_version


def _manifest(tmp_path, text, name="r65.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.parametrize("req, inside, outside", [
    ("0.4", ["0.4.0", "0.4.9"], ["0.3.9", "0.5.0"]),
    ("0.4.2", ["0.4.2", "0.4.7"], ["0.4.1", "0.5.0"]),
    ("1.2", ["1.2.0", "1.9.0"], ["1.1.9", "2.0.0"]),
    ("0.0.3", ["0.0.3"], ["0.0.4", "0.0.2"]),
    ("0", ["0.0.1", "0.9.0"], ["1.0.0"]),
    ("=0.4.1", ["0.4.1"], ["0.4.2", "0.4.0"]),
])
def test_caret_requirements(req, inside, outside):
    r = Requirement.parse(req)
    assert all(r.matches(parse_version(v)) for v in inside)
    assert not any(r.matches(parse_version(v)) for v in outside)


def test_bad_versions_are_errors():
    for text in ("0.4", "v1.0.0", "1.0.0-beta"):
        with pytest.raises(VersionError):
            parse_version(text)
    with pytest.raises(VersionError):
        Requirement.parse(">=1.0")
    assert at_least((0, 3, 0), "0.3") and not at_least((0, 2, 9), "0.3")


def test_application_manifest(tmp_path):
    m = load_manifest(_manifest(tmp_path, '''
[package]
name = "pacman"
version = "0.1.0"
r65-version = "0.2"

[bin]
main = "src/main.r65"
cfg = ["snes"]

[dependencies]
stdlib = { git = "https://example.com/R65", version = "0.4" }
q10 = { path = "../q10" }
'''))
    assert m.label == "pacman 0.1.0" and m.bin_cfg == ["snes"] and m.include_dir is None
    assert m.dependencies["stdlib"].version.matches((0, 4, 2))
    assert m.dependencies["q10"].path_dir() == (tmp_path / "../q10").resolve()


def test_bin_cfg_cannot_set_the_abi(tmp_path):
    path = _manifest(tmp_path, '[package]\nname = "a"\nversion = "0.1.0"\n[bin]\ncfg = ["abi=Pascal"]\n')
    with pytest.raises(PackageError, match="bin.cfg cannot set 'abi'"):
        load_manifest(path)


@pytest.mark.parametrize("deps, message", [
    ('x = { git = "u" }', "exactly one of `version`, `tag`, `rev` or `branch`"),
    ('x = { git = "u", tag = "a", rev = "b" }', "exactly one of"),
    ('x = { path = "a", git = "u", version = "1" }', "exactly one of `path` or `git`"),
    ('x = { path = "a", tag = "t" }', "needs `git`"),
    ('x = { git = "u", version = "1", package = "y" }', "unknown key in [dependencies.x]: package"),
    ('x = "1.0"', "a bare version needs a source"),
    ('x = { git = "u", version = ">=1" }', "invalid version requirement"),
])
def test_bad_dependencies_are_errors(tmp_path, deps, message):
    path = _manifest(tmp_path, f'[package]\nname = "a"\nversion = "0.1.0"\n[dependencies]\n{deps}\n')
    with pytest.raises(PackageError, match=None) as e:
        load_manifest(path)
    assert message in str(e.value)


def test_unknown_keys_and_tables_are_errors(tmp_path):
    with pytest.raises(PackageError, match="unknown key in \\[package\\]: abi"):
        load_manifest(_manifest(tmp_path, '[package]\nname = "a"\nversion = "0.1.0"\nabi = "x"\n'))
    with pytest.raises(PackageError, match="unknown key in \\[top level\\]: features"):
        load_manifest(_manifest(tmp_path, '[package]\nname = "a"\nversion = "0.1.0"\n[features]\n'))
    with pytest.raises(PackageError, match="a patch must be a `path`"):
        load_manifest(_manifest(tmp_path, '[workspace]\n[patch]\nx = { git = "u", tag = "t" }\n'))


def test_workspace_root_lists_members_and_is_found_from_one(tmp_path):
    _manifest(tmp_path, '[workspace]\nmembers = ["lib", "game"]\n')
    (tmp_path / "game" / "src").mkdir(parents=True)
    game = _manifest(tmp_path / "game", '[package]\nname = "game"\nversion = "0.1.0"\n')
    assert discover_manifest(tmp_path / "game" / "src" / "main.r65") == game
    root = find_workspace_root(load_manifest(game))
    assert root is not None and root.path == tmp_path / "r65.toml"
    # A package the workspace does not list is not in it.
    (tmp_path / "other").mkdir()
    other = _manifest(tmp_path / "other", '[package]\nname = "other"\nversion = "0.1.0"\n')
    assert find_workspace_root(load_manifest(other)) is None
