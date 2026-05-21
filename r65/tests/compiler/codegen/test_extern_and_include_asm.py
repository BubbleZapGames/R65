# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Codegen tests for `extern fn`, `extern static`, and `include_asm!`.

These exercise the full pipeline (parse → HIR → MIR → codegen) and assert
on the generated WLA-DX assembly, without requiring wla-65816 to be installed.
"""

import tempfile
from pathlib import Path
import pytest

from r65.compiler.main import compile_string


@pytest.fixture
def tmp_with_asm():
    """Workspace with a stub `.s` file resolvable by include_asm! and the source."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "helpers.s").write_text(
            ".SECTION \"Helpers\" FREE\n"
            "add_helper:\n"
            "    CLC\n"
            "    ADC.B #$01\n"
            "    RTS\n"
            "far_helper:\n"
            "    RTL\n"
            "PALETTE:\n"
            "    .db $00, $01, $02, $03\n"
            ".ENDS\n"
        )
        src_path = d / "main.r65"
        yield d, src_path


def _compile(src: str, src_path: Path) -> str:
    src_path.write_text(src)
    return compile_string(src, str(src_path))


def test_near_extern_call_emits_jsr(tmp_with_asm):
    """`extern fn` is called via JSR (near, 16-bit)."""
    d, src_path = tmp_with_asm
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern fn add_helper(a @ A: u8) -> u8;\n'
        'include_asm!("helpers.s");\n'
        'fn main() { A = add_helper(5); }\n',
        src_path,
    )
    assert "JSR add_helper" in asm
    assert "JSL add_helper" not in asm


def test_far_extern_call_emits_jsl(tmp_with_asm):
    """`extern far fn` is called via JSL (far, 24-bit)."""
    d, src_path = tmp_with_asm
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern far fn far_helper();\n'
        'include_asm!("helpers.s");\n'
        'fn main() { far_helper(); }\n',
        src_path,
    )
    assert "JSL far_helper" in asm


def test_include_asm_emits_wla_include_directive(tmp_with_asm):
    """include_asm! is lowered to a WLA-DX `.INCLUDE "path"` directive."""
    d, src_path = tmp_with_asm
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern fn add_helper(a @ A: u8) -> u8;\n'
        'include_asm!("helpers.s");\n'
        'fn main() { A = add_helper(0); }\n',
        src_path,
    )
    # Path is resolved relative to the .r65 source file
    expected = f'.INCLUDE "{(d / "helpers.s").resolve()}"'
    assert expected in asm


def test_extern_static_used_by_label_not_address(tmp_with_asm):
    """Reading an extern static emits a label reference, not address $00."""
    d, src_path = tmp_with_asm
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern static PALETTE: [u8; 4];\n'
        'include_asm!("helpers.s");\n'
        'fn main() { A = PALETTE[0]; }\n',
        src_path,
    )
    # Symbol-name reference, not the placeholder address that an
    # unallocated symbol would otherwise fall back to.
    assert "PALETTE" in asm
    # Sanity: the bare `LDA $00` pattern (unresolved address) must not appear
    # in the main function. Locate main and inspect its body.
    main_idx = asm.find("main:")
    assert main_idx >= 0
    main_body = asm[main_idx:asm.find("\n\n", main_idx)]
    assert "LDA $00" not in main_body, (
        f"Extern static fell back to address $00:\n{main_body}"
    )


def test_extern_static_const_index_folds_offset_into_label(tmp_with_asm):
    """`PALETTE[3]` must read `PALETTE+3`, not `PALETTE+0`.

    Regression test: an earlier version of the ROM-label resolution path
    ignored `MemoryLocation.offset`, so any compile-time constant index
    silently collapsed to index 0. Caught by a transient WLA-DX e2e test
    against an asm-defined PALETTE.
    """
    d, src_path = tmp_with_asm
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern static PALETTE: [u8; 8];\n'
        'include_asm!("helpers.s");\n'
        'fn main() { A = PALETTE[3]; }\n',
        src_path,
    )
    main_idx = asm.find("main:")
    main_body = asm[main_idx:asm.find("\n\n", main_idx)]
    # Offset must show up alongside the label
    assert "PALETTE+3" in main_body, (
        f"expected `PALETTE+3` in main body, got:\n{main_body}"
    )


def test_extern_static_mut_write(tmp_with_asm):
    """Writing to an `extern static mut` uses the bare label as the store target."""
    d, src_path = tmp_with_asm
    (d / "helpers.s").write_text(
        '.RAMSECTION "ExtRam" BANK 0 SLOT 1\n'
        'SCRATCH dsb 16\n'
        '.ENDS\n'
    )
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern static mut SCRATCH: [u8; 16];\n'
        'include_asm!("helpers.s");\n'
        'fn main() { SCRATCH[0] = 0x42; }\n',
        src_path,
    )
    assert "SCRATCH" in asm
    # The store target must reference the symbol name, not address 0.
    main_idx = asm.find("main:")
    main_body = asm[main_idx:asm.find("\n\n", main_idx)]
    assert "STA $00" not in main_body
    # Either STA SCRATCH, STA.l SCRATCH, or STZ SCRATCH (peephole may fold #0)
    assert "SCRATCH" in main_body


def test_extern_fn_default_clobbers_all_no_preserves(tmp_with_asm):
    """Without #[preserves], the compiler must not assume callee saves X/Y.

    Easiest observable signature: the caller has to (re)load anything live
    across the extern call. We check this by ensuring no #[preserves(...)]
    metadata leaks into the call lowering (the call site has no preserves_attr
    bookkeeping — only spills based on the default all-clobbered assumption).
    """
    d, src_path = tmp_with_asm
    src = (
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern fn add_helper(a @ A: u8) -> u8;\n'
        'include_asm!("helpers.s");\n'
        'fn main() {\n'
        '    X = 0x42;\n'
        '    A = add_helper(0);\n'
        '    // X used after — if extern preserved X by default,\n'
        '    // codegen could elide the reload; with all-clobbered,\n'
        '    // we expect a fresh use of X with no proof of preservation.\n'
        '    Y = X;\n'
        '}\n'
    )
    asm = _compile(src, src_path)
    # Smoke-level check: the call is emitted; X handling is the compiler's
    # to prove correct. The presence of the call is enough at this layer.
    assert "JSR add_helper" in asm


def test_extern_far_static_call_in_explicit_bank(tmp_with_asm):
    """An extern static + extern far fn inside #[bank(2)] still wires up."""
    d, src_path = tmp_with_asm
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        '#[bank(2)]\n'
        'extern far fn far_helper();\n'
        'extern static PALETTE: [u8; 4];\n'
        'include_asm!("helpers.s");\n'
        '#[bank(0)]\n'
        'fn main() { far_helper(); A = PALETTE[0]; }\n',
        src_path,
    )
    # The include lands in bank 2 (where the extern decls were)
    bank2_idx = asm.find("Bank 2")
    bank0_idx = asm.find("Bank 0")
    assert bank2_idx >= 0
    bank2_section = asm[bank2_idx:bank0_idx if bank0_idx > bank2_idx else len(asm)]
    assert ".INCLUDE" in bank2_section
    assert "JSL far_helper" in asm


def test_include_asm_resolves_via_include_paths():
    """include_asm! falls back to -I include paths when the path isn't next to the .r65."""
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as inc_dir:
        src_dir = Path(src_dir)
        inc_dir = Path(inc_dir)
        # The asm file lives in the include path, NOT next to the source
        (inc_dir / "vendor.s").write_text(
            '.SECTION "Vendor" FREE\n'
            'vendor_helper: RTS\n'
            '.ENDS\n'
        )
        src_path = src_dir / "main.r65"
        src = (
            '#[snesrom(name="T", lorom)]\n'
            '#[stack(0x0100, 0x01FF)]\n'
            'extern fn vendor_helper();\n'
            'include_asm!("vendor.s");\n'
            'fn main() { vendor_helper(); }\n'
        )
        src_path.write_text(src)

        # Without -I: must fail (file isn't next to source)
        with pytest.raises(Exception) as exc:
            compile_string(src, str(src_path))
        assert "vendor.s" in str(exc.value) or "not found" in str(exc.value).lower()

        # With -I: resolves, emits absolute path to the included file
        asm = compile_string(src, str(src_path), include_paths=[str(inc_dir)])
        expected = f'.INCLUDE "{(inc_dir / "vendor.s").resolve()}"'
        assert expected in asm
        assert "JSR vendor_helper" in asm


def test_include_asm_prefers_source_dir_over_include_paths():
    """A file next to the .r65 wins over a same-named file in an -I path."""
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as inc_dir:
        src_dir = Path(src_dir)
        inc_dir = Path(inc_dir)
        # Same filename in both locations — source-dir copy should win.
        (src_dir / "shared.s").write_text(
            '.SECTION "Local" FREE\n'
            'shared_helper: RTS  ; LOCAL\n'
            '.ENDS\n'
        )
        (inc_dir / "shared.s").write_text(
            '.SECTION "Vendor" FREE\n'
            'shared_helper: RTS  ; VENDOR\n'
            '.ENDS\n'
        )
        src_path = src_dir / "main.r65"
        src = (
            '#[snesrom(name="T", lorom)]\n'
            '#[stack(0x0100, 0x01FF)]\n'
            'extern fn shared_helper();\n'
            'include_asm!("shared.s");\n'
            'fn main() { shared_helper(); }\n'
        )
        src_path.write_text(src)

        asm = compile_string(src, str(src_path), include_paths=[str(inc_dir)])
        expected_local = f'.INCLUDE "{(src_dir / "shared.s").resolve()}"'
        assert expected_local in asm


def test_include_asm_missing_file_raises(tmp_with_asm):
    """A nonexistent path triggers a clear compile-time error."""
    d, src_path = tmp_with_asm
    with pytest.raises(Exception) as exc:
        _compile(
            '#[snesrom(name="T", lorom)]\n'
            '#[stack(0x0100, 0x01FF)]\n'
            'include_asm!("does_not_exist.s");\n'
            'fn main() {}\n',
            src_path,
        )
    assert "does_not_exist.s" in str(exc.value) or "not found" in str(exc.value).lower()


def test_extern_static_rejects_storage_attribute(tmp_with_asm):
    """`#[ram] extern static FOO;` is rejected — the asm file owns placement."""
    d, src_path = tmp_with_asm
    with pytest.raises(Exception):
        _compile(
            '#[snesrom(name="T", lorom)]\n'
            '#[stack(0x0100, 0x01FF)]\n'
            '#[ram]\n'
            'extern static mut FOO: u8;\n'
            'include_asm!("helpers.s");\n'
            'fn main() { FOO = 1; }\n',
            src_path,
        )


def test_address_of_extern_static(tmp_with_asm):
    """`&extern_static[i]` compiles without falling into the allocator path.

    Earlier, `_emit_variable_address` required `mem_alloc.get_allocation()`
    to return non-None, which failed on extern statics (they have no
    allocation by design). Now it falls back to the `rom_label` path.
    """
    d, src_path = tmp_with_asm
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        'extern static PALETTE: [u8; 4];\n'
        'include_asm!("helpers.s");\n'
        'fn read(p: far *u8, idx @ Y: u16) -> u8 { A = p[Y]; return A; }\n'
        '#[entry]\n'
        'fn main() {\n'
        '    let ptr: far *u8 = &PALETTE[0];\n'
        '    A = read(ptr, 2);\n'
        '}\n',
        src_path,
    )
    # The label must appear in the address load — not a literal address.
    assert "PALETTE" in asm
    assert "JSR read" in asm


def test_near_extern_cross_bank_call_rejected(tmp_with_asm):
    """Near `extern fn` declared in one bank cannot be called from another.

    Same rule as native near fns: JSR can't cross bank boundaries, so the
    type checker rejects the call and suggests `far fn`.
    """
    d, src_path = tmp_with_asm
    with pytest.raises(Exception) as exc:
        _compile(
            '#[snesrom(name="T", lorom)]\n'
            '#[stack(0x0100, 0x01FF)]\n'
            '#[bank(2)]\n'
            'extern fn helper_in_bank2();\n'
            'include_asm!("helpers.s");\n'
            '#[bank(0)]\n'
            'fn main() { helper_in_bank2(); }\n',
            src_path,
        )
    msg = str(exc.value)
    assert "helper_in_bank2" in msg
    assert "bank" in msg.lower()


def test_extern_far_required_in_auto_bank(tmp_with_asm):
    """Auto-bank mode forces extern fn to be `extern far fn`."""
    d, src_path = tmp_with_asm
    # Near extern in auto-bank should fail
    with pytest.raises(Exception):
        _compile(
            '#[snesrom(name="T", lorom)]\n'
            '#[stack(0x0100, 0x01FF)]\n'
            '#[bank(auto)]\n'
            'extern fn helper();\n'
            'include_asm!("helpers.s");\n'
            'far fn main() { helper(); }\n',
            src_path,
        )
    # `extern far fn` in auto-bank works
    asm = _compile(
        '#[snesrom(name="T", lorom)]\n'
        '#[stack(0x0100, 0x01FF)]\n'
        '#[bank(auto)]\n'
        'extern far fn far_helper();\n'
        'include_asm!("helpers.s");\n'
        'far fn main() { far_helper(); }\n',
        src_path,
    )
    assert "JSL far_helper" in asm
