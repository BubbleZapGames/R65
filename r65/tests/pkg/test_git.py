# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Git dependencies against throwaway local repositories: one repository
serving two independently versioned packages, caret unification over its
release tags, exact pins, the tag/manifest check, the lockfile and r65pkg."""

import shutil
import subprocess
import sys

import pytest

from r65.pkg.cli import main as r65pkg
from r65.pkg.fetch import RemoteGit
from r65.pkg.lockfile import read_lock
from r65.pkg.manifest import PackageError, load_manifest
from r65.pkg.resolve import resolve
from r65.tests.pkg.helpers import git, package, release, repository

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def repo(tmp_path):
    """`mono` holds std (0.4.0, 0.4.2, 0.5.0) and q10 (0.2.0, needing std 0.4.1+)."""
    r = repository(tmp_path / "mono", ["std", "q10"])
    release(r, "std", "0.4.0")
    release(r, "q10", "0.2.0", deps=f'std = {{ git = "{r}", version = "0.4.1" }}')
    release(r, "std", "0.4.2")
    release(r, "std", "0.5.0")
    return r


def _app(tmp_path, deps):
    return package(tmp_path / "app", "app", deps=deps, lib=False)


def _fetch(manifest_path, update=None):
    root = load_manifest(manifest_path)
    lock = root.dir / "r65.lock"
    return resolve(root, RemoteGit(root.dir / ".r65/deps", read_lock(lock) if lock.is_file() else {},
                                   update=update))


def test_one_repository_serves_two_versioned_packages(tmp_path, repo):
    r = _fetch(_app(tmp_path, f'q10 = {{ git = "{repo}", version = "0.2" }}\n'
                              f'std = {{ git = "{repo}", version = "0.4" }}'))
    assert {n: p.version for n, p in r.packages.items()} == {"q10": (0, 2, 0), "std": (0, 4, 2)}
    # Each is the member of that name, at its own release tag.
    assert r.packages["std"].source.endswith("?tag=std-v0.4.2")
    assert (r.packages["std"].dir / "std.r65").read_text() == "// std 0.4.2\n"
    assert r.order == ["std", "q10"]


def test_ranges_unify_to_the_newest_release_all_allow(tmp_path, repo):
    # app wants 0.4.0+, q10 0.4.1+: the newest 0.4.x, never 0.5.0.
    r = _fetch(_app(tmp_path, f'q10 = {{ git = "{repo}", version = "0.2" }}\n'
                              f'std = {{ git = "{repo}", version = "0.4.0" }}'))
    assert r.packages["std"].version == (0, 4, 2)


def test_no_release_satisfying_everyone_names_both(tmp_path, repo):
    with pytest.raises(PackageError) as e:
        _fetch(_app(tmp_path, f'q10 = {{ git = "{repo}", version = "0.2" }}\n'
                              f'std = {{ git = "{repo}", version = "0.5" }}'))
    text = str(e.value)
    assert "cannot resolve 'std': no release satisfies them all" in text
    assert 'app 0.1.0 requires version "0.5"' in text and 'q10 0.2.0 requires version "0.4.1"' in text


def test_exact_pins(tmp_path, repo):
    rev = git(repo, "rev-list", "-n1", "std-v0.4.0").strip()
    r = _fetch(_app(tmp_path, f'std = {{ git = "{repo}", rev = "{rev}" }}'))
    assert r.packages["std"].version == (0, 4, 0) and r.packages["std"].rev == rev
    # A pin outside a range on the same package is refused.
    with pytest.raises(PackageError, match="std-v0.4.0 holds 0.4.0"):
        _fetch(_app(tmp_path, f'q10 = {{ git = "{repo}", version = "0.2" }}\n'
                              f'std = {{ git = "{repo}", tag = "std-v0.4.0" }}'))


def test_a_tag_disagreeing_with_its_manifest_is_refused(tmp_path):
    r = repository(tmp_path / "bad", ["std"])
    release(r, "std", "0.3.0", tag="std-v0.4.0")
    with pytest.raises(PackageError, match="tag std-v0.4.0 .* holds std 0.3.0, not 0.4.0"):
        _fetch(_app(tmp_path, f'std = {{ git = "{r}", version = "0.4" }}'))


def test_lockfile_pins_until_update(tmp_path, repo, monkeypatch):
    app = _app(tmp_path, f'std = {{ git = "{repo}", version = "0.4" }}')
    monkeypatch.chdir(app.parent)
    r65pkg(["fetch"])
    locked = read_lock(app.parent / "r65.lock")["std"]
    assert locked.version == (0, 4, 2) and locked.source == f"git+{repo}?tag=std-v0.4.2"

    # r65c resolves from the lock and the clone, with no git at all.
    assert resolve(load_manifest(app)).packages["std"].version == (0, 4, 2)

    release(repo, "std", "0.4.5")
    r65pkg(["fetch"])  # locked and still allowed: kept
    assert read_lock(app.parent / "r65.lock")["std"].version == (0, 4, 2)
    r65pkg(["update"])
    assert read_lock(app.parent / "r65.lock")["std"].version == (0, 4, 5)

    # A requirement the lock no longer meets makes it stale for r65c.
    app.write_text(app.read_text().replace('version = "0.4"', 'version = "0.5"'))
    with pytest.raises(PackageError, match="run `r65pkg update std`"):
        resolve(load_manifest(app))


def test_r65pkg_add_remove_tree(tmp_path, repo, monkeypatch, capsys):
    app = package(tmp_path / "app", "app", lib=False)
    monkeypatch.chdir(app.parent)
    r65pkg(["add", "q10", "--git", str(repo)])   # newest release, as a caret range
    assert 'q10 = { git = "' + str(repo) + '", version = "0.2" }' in app.read_text()
    assert set(read_lock(app.parent / "r65.lock")) == {"q10", "std"}
    capsys.readouterr()
    r65pkg(["tree"])
    out = capsys.readouterr().out
    assert "app 0.1.0\n└── q10 0.2.0" in out and "    └── std 0.4.2" in out
    r65pkg(["remove", "q10"])
    assert "q10" not in app.read_text() and read_lock(app.parent / "r65.lock") == {}


def _single_package_repo(directory, name, version):
    """A repository whose root r65.toml is the package itself."""
    directory.mkdir(parents=True)
    git(directory, "init", "-q", "-b", "main")
    package(directory, name, version, files={f"{name}.r65": "\n"})
    git(directory, "add", "-A")
    git(directory, "commit", "-q", "-m", version)
    return directory


def test_r65pkg_add_url_names_the_package_for_you(tmp_path, monkeypatch):
    tween = _single_package_repo(tmp_path / "tween", "tween", "0.2.1")
    app = package(tmp_path / "app", "app", lib=False)
    monkeypatch.chdir(app.parent)
    with pytest.raises(SystemExit):
        r65pkg(["add", str(tween)])   # no release tags and no ref to read
    r65pkg(["add", str(tween), "--branch", "main"])
    assert 'tween = { git = "' + str(tween) + '", branch = "main" }' in app.read_text()
    git(tween, "tag", "-a", "tween-v0.2.1", "-m", "0.2.1")
    r65pkg(["add", str(tween)])
    assert 'tween = { git = "' + str(tween) + '", version = "0.2" }' in app.read_text()


def test_r65pkg_add_url_of_several_packages_needs_a_name(tmp_path, repo, monkeypatch, capsys):
    app = package(tmp_path / "app", "app", lib=False)
    monkeypatch.chdir(app.parent)
    with pytest.raises(SystemExit):
        r65pkg(["add", str(repo)])
    assert "several packages (q10, std)" in capsys.readouterr().err
    assert "[dependencies]" not in app.read_text()


def test_r65pkg_tag_releases_the_committed_manifest_version(tmp_path, monkeypatch):
    r = repository(tmp_path / "ws", ["std"])
    release(r, "std", "0.1.0")
    monkeypatch.chdir(r / "std")
    package(r / "std", "std", "0.2.0")
    with pytest.raises(SystemExit):
        r65pkg(["tag"])   # the bump is not committed yet
    git(r, "commit", "-qam", "bump")
    r65pkg(["tag"])
    assert git(r, "tag", "--list", "std-v0.2.0").strip() == "std-v0.2.0"
    with pytest.raises(SystemExit):
        r65pkg(["tag"])   # already tagged


def test_r65pkg_tag_version_bumps_commits_and_tags(tmp_path, monkeypatch):
    r = repository(tmp_path / "ws", ["std"])
    release(r, "std", "0.1.0")
    monkeypatch.chdir(r / "std")
    r65pkg(["tag", "0.2.1"])
    assert load_manifest(r / "std" / "r65.toml").version == (0, 2, 1)
    assert git(r, "status", "--porcelain").strip() == ""
    assert git(r, "log", "-1", "--format=%s", "std-v0.2.1").strip() == "std 0.2.1"
    with pytest.raises(SystemExit):
        r65pkg(["tag", "0.1.0"])   # taken


def test_r65pkg_tag_needs_a_package_manifest_in_the_cwd(tmp_path, monkeypatch):
    r = repository(tmp_path / "ws", ["std"])
    release(r, "std", "0.1.0", files={"src/std.r65": "\n"})
    for cwd in (r, r / "std" / "src"):   # a workspace root; no r65.toml here
        monkeypatch.chdir(cwd)
        with pytest.raises(SystemExit):
            r65pkg(["tag", "0.2.0"])
    assert git(r, "tag", "--list", "std-v0.2.0").strip() == ""


def test_compile_a_fetched_git_dependency(tmp_path, repo, monkeypatch):
    release(repo, "std", "0.4.6", files={"std.r65": "fn from_std() { A = 7; }\n"})
    app = _app(tmp_path, f'std = {{ git = "{repo}", version = "0.4" }}')
    (app.parent / "main.r65").write_text(
        'include!("std/std.r65")\n#[entry]\nfn main() { from_std(); }\n')
    run = lambda: subprocess.run([sys.executable, "-m", "r65.compiler.main", "main.r65", "-q",
                                  "-o", "out.asm"], cwd=app.parent, capture_output=True, text=True)
    assert "run `r65pkg fetch`" in run().stderr
    monkeypatch.chdir(app.parent)
    r65pkg(["fetch"])
    done = run()
    assert done.returncode == 0, done.stderr
    assert "from_std:" in (app.parent / "out.asm").read_text()


def _seeded(tmp_path, url, version="0.4.2"):
    """An app as `r65x init` leaves it: std copied (no clone) and locked as
    the release it is, with no commit."""
    from r65.pkg.lockfile import LockedPackage, git_source, write_lock
    from r65.pkg.version import parse_version
    app = _app(tmp_path, f'std = {{ git = "{url}", version = "0.4" }}')
    package(app.parent / ".r65/deps/std", "std", version,
            files={"std.r65": "fn from_copy() { A = 1; }\n"})
    write_lock(app.parent / "r65.lock", {"std": LockedPackage(
        "std", parse_version(version), git_source(url, "tag", f"std-v{version}"))})
    return app


def test_an_installed_copy_needs_no_network_until_updated(tmp_path, repo, monkeypatch):
    # Nothing answers at this URL: resolving and fetching must not ask it.
    app = _seeded(tmp_path, str(tmp_path / "unreachable"))
    assert resolve(load_manifest(app)).packages["std"].rev is None
    monkeypatch.chdir(app.parent)
    r65pkg(["fetch"])
    assert read_lock(app.parent / "r65.lock")["std"].rev is None
    assert not (app.parent / ".r65/deps/std/.git").exists()


def test_a_missing_copy_is_cloned_and_pinned(tmp_path, repo, monkeypatch):
    app = _seeded(tmp_path, str(repo))
    shutil.rmtree(app.parent / ".r65")        # a fresh checkout of the project
    monkeypatch.chdir(app.parent)
    r65pkg(["fetch"])
    locked = read_lock(app.parent / "r65.lock")["std"]
    assert locked.version == (0, 4, 2) and locked.rev == git(repo, "rev-list", "-n1", "std-v0.4.2").strip()


def test_update_replaces_the_copy_with_the_newest_release(tmp_path, repo, monkeypatch):
    app = _seeded(tmp_path, str(repo), version="0.4.0")
    release(repo, "std", "0.4.7")
    monkeypatch.chdir(app.parent)
    r65pkg(["update"])
    locked = read_lock(app.parent / "r65.lock")["std"]
    assert locked.version == (0, 4, 7) and locked.rev
    assert resolve(load_manifest(app)).packages["std"].rev == locked.rev
