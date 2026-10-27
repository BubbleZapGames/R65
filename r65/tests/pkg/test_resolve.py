# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Resolution without the network: path dependencies, [patch], unification,
build requirements, and what r65c makes of the result."""

import subprocess
import sys

import pytest

from r65.pkg.manifest import PackageError, load_manifest
from r65.pkg.resolve import check_build, resolve
from r65.tests.pkg.helpers import package


def _app(tmp_path, deps, extra=""):
    return load_manifest(package(tmp_path / "app", "app", deps=deps, lib=False, extra=extra))


def test_path_dependencies_unify_and_order(tmp_path):
    package(tmp_path / "base", "base")
    package(tmp_path / "mid", "mid", deps='base = { path = "../base" }')
    r = resolve(_app(tmp_path, 'mid = { path = "../mid" }\nbase = { path = "../base", version = "0.1" }'))
    assert r.order == ["base", "mid"]
    assert r.include_dirs() == [(tmp_path / "base").resolve(), (tmp_path / "mid").resolve()]
    assert r.packages["base"].requirers == ["app 0.1.0", "mid 0.1.0"]


def test_one_package_from_two_directories_is_an_error(tmp_path):
    package(tmp_path / "base", "base")
    package(tmp_path / "base2", "base")
    package(tmp_path / "mid", "mid", deps='base = { path = "../base2" }')
    with pytest.raises(PackageError, match="'base' is required from two directories"):
        resolve(_app(tmp_path, 'mid = { path = "../mid" }\nbase = { path = "../base" }'))


def test_unsatisfied_requirement_names_every_requirer(tmp_path):
    package(tmp_path / "base", "base", version="0.2.0")
    package(tmp_path / "mid", "mid", deps='base = { path = "../base", version = "0.1" }')
    with pytest.raises(PackageError) as e:
        resolve(_app(tmp_path, 'mid = { path = "../mid" }\nbase = { path = "../base" }'))
    assert "base 0.2.0 does not satisfy" in str(e.value)
    assert 'mid 0.1.0 requires path "../base"' in str(e.value)


def test_patch_redirects_a_git_dependency_and_must_satisfy_it(tmp_path):
    """A workspace's [patch] points a git dependency at a local directory;
    a patched version outside the requirement is an error, not ignored."""
    (tmp_path / "r65.toml").write_text(
        '[workspace]\nmembers = ["std", "app"]\n[patch]\nstd = { path = "std" }\n')
    package(tmp_path / "std", "std", version="0.4.3")
    app = _app(tmp_path, 'std = { git = "https://example.com/nowhere", version = "0.4" }')
    r = resolve(app)
    assert r.packages["std"].dir == (tmp_path / "std").resolve()
    assert r.packages["std"].source == "path+../std"

    package(tmp_path / "std", "std", version="0.5.0")
    with pytest.raises(PackageError, match="patched std 0.5.0 does not satisfy"):
        resolve(app)


def test_unpatched_git_dependency_needs_the_lockfile(tmp_path):
    app = _app(tmp_path, 'std = { git = "https://example.com/nowhere", version = "0.4" }')
    with pytest.raises(PackageError, match="run `r65pkg fetch`"):
        resolve(app)


def test_a_dependency_must_be_a_library(tmp_path):
    package(tmp_path / "tool", "tool", lib=False)
    with pytest.raises(PackageError, match="'tool' is not a library"):
        resolve(_app(tmp_path, 'tool = { path = "../tool" }'))


def test_build_requirements(tmp_path):
    package(tmp_path / "fast", "fast", extra='r65-version = "0.3"\ncfg = ["snes"]')
    r = resolve(_app(tmp_path, 'fast = { path = "../fast" }'))
    with pytest.raises(PackageError, match=r"fast 0.1.0 requires r65c >= 0.3; this is r65c 0.2.0"):
        check_build(r, (0, 2, 0), ["snes"])
    with pytest.raises(PackageError, match="fast 0.1.0 must be built with --cfg snes"):
        check_build(r, (0, 3, 0), [])
    check_build(r, (0, 3, 1), ["snes", "debug"])


# ---------------------------------------------------------------------------
# Through r65c

def _r65c(cwd, *args):
    return subprocess.run([sys.executable, "-m", "r65.compiler.main", *args],
                          cwd=cwd, capture_output=True, text=True)


_MAIN = '#[entry]\nfn main() {{ {body} }}\n'


def test_qualified_include_bare_ambiguity_and_local_shadow(tmp_path):
    package(tmp_path / "one", "one", files={"math.r65": "fn one_math() { A = 1; }\n"})
    package(tmp_path / "two", "two", files={"math.r65": "fn two_math() { A = 2; }\n"})
    package(tmp_path / "app", "app", lib=False,
            deps='one = { path = "../one" }\ntwo = { path = "../two" }')
    src = tmp_path / "app" / "src"
    src.mkdir()
    main = src / "main.r65"

    main.write_text('include!("two/math.r65")\n' + _MAIN.format(body="two_math();"))
    done = _r65c(tmp_path / "app", "src/main.r65", "-o", "out.asm", "-q")
    assert done.returncode == 0, done.stderr

    main.write_text('include!("math.r65")\n' + _MAIN.format(body=""))
    done = _r65c(tmp_path / "app", "src/main.r65", "-o", "out.asm", "-q")
    assert "'math.r65' is provided by 2 packages" in done.stderr
    assert "include it qualified ('one/math.r65')" in done.stderr

    # A file next to the includer wins over every package, by either spelling.
    (src / "math.r65").write_text("fn local_math() { A = 3; }\n")
    main.write_text('include!("math.r65")\n' + _MAIN.format(body="local_math();"))
    assert _r65c(tmp_path / "app", "src/main.r65", "-o", "out.asm", "-q").returncode == 0


def test_qualified_and_bare_include_of_one_file_compile_it_once(tmp_path):
    package(tmp_path / "one", "one", files={"util.r65": "fn util() { A = 1; }\n"})
    package(tmp_path / "app", "app", lib=False, deps='one = { path = "../one" }')
    (tmp_path / "app" / "main.r65").write_text(
        'include!("one/util.r65")\ninclude!("util.r65")\n' + _MAIN.format(body="util();"))
    done = _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q",
                 "--depfile", "out.d")
    assert done.returncode == 0, done.stderr
    deps = (tmp_path / "app" / "out.d").read_text()
    assert str((tmp_path / "one" / "util.r65").resolve()) in deps
    assert str((tmp_path / "app" / "r65.toml").resolve()) in deps


def test_r65c_reports_build_requirements_and_takes_bin_cfg(tmp_path):
    package(tmp_path / "hw", "hw", extra='cfg = ["snes"]',
            files={"hw.r65": "#[cfg(snes)]\nfn hw() { A = 1; }\n"})
    package(tmp_path / "app", "app", lib=False, deps='hw = { path = "../hw" }')
    (tmp_path / "app" / "main.r65").write_text(
        'include!("hw/hw.r65")\n' + _MAIN.format(body="hw();"))
    done = _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q")
    assert "hw 0.1.0 must be built with --cfg snes" in done.stderr

    # [bin] cfg supplies it, and --cfg adds to it; a key=value clash is an error.
    manifest = tmp_path / "app" / "r65.toml"
    manifest.write_text(manifest.read_text().replace(
        "[dependencies]", '[bin]\ncfg = ["snes", "target=snes"]\n[dependencies]'))
    assert _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q").returncode == 0
    assert _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q", "--cfg", "debug").returncode == 0
    done = _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q", "--cfg", "target=nes")
    assert "--cfg target=nes contradicts [bin] cfg 'target=snes'" in done.stderr
    # --no-manifest compiles as before packages: the include is not found.
    done = _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q", "--no-manifest")
    assert "file not found: 'hw/hw.r65'" in done.stderr


def test_redefinition_names_both_sites_and_the_package(tmp_path):
    package(tmp_path / "one", "one", files={"a.r65": "fn clash() { A = 1; }\n"})
    package(tmp_path / "app", "app", lib=False, deps='one = { path = "../one" }')
    (tmp_path / "app" / "main.r65").write_text(
        'include!("one/a.r65")\nfn clash() { A = 2; }\n' + _MAIN.format(body="clash();"))
    done = _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q")
    assert "Redefinition of 'clash'" in done.stderr
    assert f"first defined at {(tmp_path / 'one' / 'a.r65').resolve()}:1:1 (one 0.1.0)" in done.stderr
    assert "main.r65:2:1" in done.stderr


def test_macro_defined_in_two_files_is_an_error(tmp_path):
    package(tmp_path / "one", "one", files={"m.r65": "macro_rules! twice($x:expr) { $x; $x; }\n"})
    package(tmp_path / "app", "app", lib=False, deps='one = { path = "../one" }')
    (tmp_path / "app" / "main.r65").write_text(
        'include!("one/m.r65")\nmacro_rules! twice($x:expr) { $x; }\n' + _MAIN.format(body=""))
    done = _r65c(tmp_path / "app", "main.r65", "-o", "out.asm", "-q")
    assert "macro 'twice!' is defined in two files" in done.stderr
    assert "(one 0.1.0)" in done.stderr
