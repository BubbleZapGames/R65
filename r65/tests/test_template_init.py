"""r65x init: template subdirectories keep their layout in the project root."""

from r65.cli import TemplateManager


def test_snes_tests_dir_lands_in_project_root(tmp_path):
    project = tmp_path / "demo"
    TemplateManager().copy_template("snes", project, "demo")

    assert (project / "tests" / "example.script").is_file()
    assert "../build/DEMO.smc" in (project / "tests" / "Makefile").read_text()
    assert not (project / "src" / "tests").exists()
    assert not (project / "src" / "Makefile").exists()


def test_stdlib_is_a_package_dependency_not_a_copy(tmp_path):
    project = tmp_path / "demo"
    manager = TemplateManager()
    manager.copy_template("snes", project, "demo")
    repository, version = manager.stdlib_release()
    major, minor, _ = version.split(".")

    assert not (project / "src" / "lib").exists()
    assert (f'stdlib = {{ git = "{repository}", version = "{major}.{minor}" }}'
            in (project / "r65.toml").read_text())
    assert 'include!("stdlib/sneslib.r65")' in (project / "src" / "main.r65").read_text()
    assert ".r65/" in (project / ".gitignore").read_text()


def test_stdlib_starts_as_the_installed_copy_locked_to_its_release(tmp_path):
    from r65.pkg.lockfile import read_lock
    from r65.pkg.manifest import load_manifest
    from r65.pkg.resolve import resolve

    project = tmp_path / "demo"
    manager = TemplateManager()
    manager.copy_template("snes", project, "demo")
    repository, version = manager.stdlib_release()

    copy = project / ".r65" / "deps" / "stdlib"
    assert (copy / "sneslib.r65").read_text() == (manager.stdlib_dir / "sneslib.r65").read_text()
    locked = read_lock(project / "r65.lock")["stdlib"]
    assert locked.source == f"git+{repository}?tag=stdlib-v{version}" and locked.rev is None
    # r65c builds from the copy with no fetch.
    assert resolve(load_manifest(project / "r65.toml")).packages["stdlib"].dir == copy.resolve()
