# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""The CLI's dump paths run the same frontend, and report errors the same way,
as a normal compile."""
import sys

import pytest

from r65.compiler.main import main


def _run_cli(monkeypatch, tmp_path, source, *flags):
    path = tmp_path / "prog.r65"
    path.write_text(source)
    monkeypatch.setattr(sys, "argv", ["r65c", str(path), *flags])
    main()


def test_dump_hir_sees_builtin_traits(monkeypatch, tmp_path, capsys):
    source = """
    struct P { x: u8 }
    impl ToString for P { fn to_string(far *self, buf: far *u8) -> u16 { return 0; } }
    fn main() {}
    """
    _run_cli(monkeypatch, tmp_path, source, "--dump-hir")
    assert "HIRImplDecl" in capsys.readouterr().out


def test_dump_ast_formats_syntax_error(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, tmp_path, "fn main( {\n", "--dump-ast")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "error: unexpected token '{'" in err
    assert "Traceback" not in err


def test_assembly_on_stdout_carries_no_pass_statistics(monkeypatch, tmp_path, capsys):
    # The optimizer statistics are diagnostics: on stdout they would sit in
    # front of the assembly and break `r65c prog.r65 > prog.asm`.
    source = """
    #[lowram]
    static mut N: u8;
    #[entry]
    fn main() -> ! { N = N + 1; loop {} }
    """
    _run_cli(monkeypatch, tmp_path, source)
    out = capsys.readouterr().out
    assert out.lstrip().startswith(";")
    assert "folding" not in out and "optimization(s) applied" not in out
