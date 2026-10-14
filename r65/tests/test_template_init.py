"""r65x init: template subdirectories keep their layout in the project root."""

from r65.cli import TemplateManager


def test_snes_tests_dir_lands_in_project_root(tmp_path):
    project = tmp_path / "demo"
    TemplateManager().copy_template("snes", project, "demo")

    assert (project / "tests" / "example.script").is_file()
    assert "../build/DEMO.smc" in (project / "tests" / "Makefile").read_text()
    assert not (project / "src" / "tests").exists()
    assert not (project / "src" / "Makefile").exists()
