# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Regression tests for local array/string literal initializer codegen.

Root cause (fixed): a local `let arr: [T; N] = [...]` initializer lowered to
an MVN block-copy whose ROM data table was routed through the *global*
trailing "ROM Data Sections" block (a different ROM bank than the function),
while `select_block_copy` hardcoded `MVN $00, dst` — source bank $00. The MVN
then read bank-$00 garbage instead of the init constants. In classickong.r65
this filled cutscene_level's stack_* arrays with junk and Kong vanished.

Fix: the data table is emitted adjacent to its function (same ROM bank) and
the MVN source bank is the data label's own bank via WLA-DX's `:label`
operator. Tiny single-bank test ROMs hid the bug (data happened to land in
bank 0), so these assert the codegen contract directly.
"""

from r65.compiler.main import compile_string


SOURCE = """
#[zeropage(0x30)]
static mut OUT: u8;

#[bank(1)]
far fn show() {
    let pal: [u8; 4] = [10, 20, 30, 40];
    OUT = pal[2];
}

#[entry]
fn main() {
    show();
}
"""


def _function_region(full_asm: str, func_name: str) -> str:
    """Asm from `func:` up to the next top-level `; ---` separator.

    With the fix, the function's local literal data is emitted right after
    its body (before that separator), so this region includes it.
    """
    lines = full_asm.split('\n')
    out, in_func = [], False
    for line in lines:
        if line.strip() == f'{func_name}:':
            in_func = True
        elif in_func:
            if line.startswith('; ---') and out:
                break
            out.append(line)
    return '\n'.join(out)


class TestLocalArrayInitBlockCopyBank:
    def test_mvn_source_bank_is_label_not_hardcoded_zero(self):
        """MVN source bank must be the data label (`:label`), not `$00`."""
        asm = compile_string(SOURCE)
        region = _function_region(asm, 'show')
        mvn_lines = [l.strip() for l in region.split('\n')
                     if l.strip().startswith('MVN')]
        assert mvn_lines, f"expected an MVN block-copy in show(), got:\n{region}"
        for mvn in mvn_lines:
            # Pre-fix bug emitted `MVN $00, $00` (hardcoded source bank).
            assert mvn != 'MVN $00, $00', (
                f"MVN still hardcodes source bank $00: {mvn}")
            assert ':' in mvn.split(',')[0], (
                f"MVN source bank should use the `:label` operator: {mvn}")

    def test_local_data_emitted_adjacent_to_function(self):
        """The _data table is emitted in the function's region, not only in
        the global trailing ROM Data Sections (which is a different bank)."""
        asm = compile_string(SOURCE)
        region = _function_region(asm, 'show')
        assert 'Local literal init data for show' in region, (
            f"local init data not emitted adjacent to show():\n{region}")
        # The data label + correct bytes must be in the per-function region.
        assert '_data:' in region
        assert '.db $0A, $14, $1E, $28' in region, (
            f"init bytes [10,20,30,40] missing from show() region:\n{region}")

    def test_local_data_not_in_global_rom_sections(self):
        """The local table must NOT also be dumped in the global section."""
        asm = compile_string(SOURCE)
        marker = 'ROM Data Sections (array literal init data)'
        if marker not in asm:
            return  # no global section at all — fine
        global_block = asm.split(marker, 1)[1]
        assert '_data:' not in global_block or 'show' not in global_block, (
            "local literal data leaked into the global ROM Data Sections")
