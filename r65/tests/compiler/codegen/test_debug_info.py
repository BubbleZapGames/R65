"""Tests for the cc65 debug-info collector."""

from r65.compiler.codegen.debug_info import DebugInfoCollector


def test_find_or_add_span_reuses_an_identical_span():
    info = DebugInfoCollector()
    first = info.find_or_add_span(0, 0x10, 3)
    assert info.find_or_add_span(0, 0x10, 3) is first
    assert info.find_or_add_span(1, 0x10, 3) is not first
    assert len(info.spans) == 2


def test_auto_bank_function_symbol_is_in_its_banks_segment(tmp_path):
    from r65.compiler.frontend import parse
    from r65.compiler.hir import HIRBuilder
    from r65.compiler.typeck import TypeChecker
    from r65.compiler.mir import MIRBuilder
    from r65.compiler.codegen import ProgramCodeGenerator
    hir = HIRBuilder().build_program(parse('''
        #[zeropage] static mut R: u8;
        #[bank(auto)]
        #[inline(never)]
        far fn f() { R = 1; }
        #[bank(0)]
        #[entry] fn main() { f(); }
    '''))
    TypeChecker(hir).check()
    gen = ProgramCodeGenerator()
    gen.generate(MIRBuilder().build_program(hir), opt_level=0,
                 output_file=str(tmp_path / "t.asm"), debug=True)
    seg = {s.id: s.name for s in gen.debug_info.segments}
    sym = next(s for s in gen.debug_info.symbols if s.name == "f")
    assert seg[sym.seg_id] == "BANK4"


def test_inline_asm_has_line_info(tmp_path):
    # A game idles on asm!("WAI"): pausing it in the debugger lands there,
    # so the .dbg has to say which line that is
    from r65.compiler.frontend import parse
    from r65.compiler.hir import HIRBuilder
    from r65.compiler.typeck import TypeChecker
    from r65.compiler.mir import MIRBuilder
    from r65.compiler.codegen import ProgramCodeGenerator
    hir = HIRBuilder().build_program(parse(
        '#[bank(0)]\n'
        '#[entry] fn main() -> ! {\n'
        '    loop {\n'
        '        asm!("WAI");\n'
        '    }\n'
        '}\n'))
    TypeChecker(hir).check()
    gen = ProgramCodeGenerator()
    gen.generate(MIRBuilder().build_program(hir), opt_level=0,
                 output_file=str(tmp_path / "t.asm"), debug=True)
    spans = {s.id: s for s in gen.debug_info.spans}
    line4 = [l for l in gen.debug_info.lines if l.line == 4]
    assert line4, "no line entry for the asm! line"
    # The loop's branches share the line; the 1-byte WAI must be there too
    assert 1 in [spans[i].size for l in line4 for i in l.span_ids]


def _compile_debug(tmp_path, source):
    from r65.compiler.frontend import parse
    from r65.compiler.hir import HIRBuilder
    from r65.compiler.typeck import TypeChecker
    from r65.compiler.mir import MIRBuilder
    from r65.compiler.codegen import ProgramCodeGenerator
    hir = HIRBuilder().build_program(parse(source))
    TypeChecker(hir).check()
    gen = ProgramCodeGenerator()
    gen.generate(MIRBuilder().build_program(hir), opt_level=0,
                 output_file=str(tmp_path / "t.asm"), debug=True)
    return gen.debug_info


def test_address_calculator_lays_out_banks_like_wlalink():
    # wlalink: .ORG sets the offset into a bank, and .BANK alone keeps the
    # running offset, even into a different bank
    from r65.compiler.codegen.address_calc import AddressCalculator
    from r65.compiler.codegen.asm_nodes import Instruction, Label, RawAsm
    from r65.compiler.codegen.opcodes import Opcode
    nop = lambda: Instruction(Opcode.NOP)
    nodes = [
        RawAsm(".BANK 0 SLOT 0"), RawAsm(".ORG 0"), Label("a"), nop(), nop(),
        RawAsm(".BANK 1 SLOT 0"), RawAsm(".ORG 0"), Label("b"), nop(), nop(), nop(),
        RawAsm(".BANK 0 SLOT 0"), Label("c"), nop(),
    ]
    calc = AddressCalculator(0x8000)
    _, labels = calc.calculate_with_labels(nodes)
    assert labels == {"a": 0x8000, "b": 0x8000, "c": 0x8003}
    assert calc.label_banks == {"a": 0, "b": 1, "c": 0}
    assert calc.bank_ends == {0: 0x8004, 1: 0x8003}


def test_address_calculator_follows_rep_sep():
    # WLA-DX sizes an immediate by the last REP/SEP too: an asm! block after
    # SEP #$20 with no .ACCU 8 is 8-bit
    from r65.compiler.codegen.address_calc import AddressCalculator
    from r65.compiler.codegen.asm_nodes import Instruction, Label, ModeChange, RawAsm, Immediate
    from r65.compiler.codegen.opcodes import Opcode
    nodes = [
        ModeChange("ACCU", 16),
        Instruction(Opcode.SEP_IMMEDIATE, Immediate(0x20)),
        RawAsm("LDA #0x00\nSTA $2100\nREP #$20\nLDA #$1234"),
        Label("end"),
    ]
    _, labels = AddressCalculator(0x8000).calculate_with_labels(nodes)
    # SEP 2, LDA #imm8 2, STA abs 3, REP 2, LDA #imm16 3
    assert labels["end"] == 0x8000 + 12


def test_far_bank_lines_are_in_their_banks_segment(tmp_path):
    info = _compile_debug(tmp_path,
        '#[zeropage] static mut R: u8;\n'
        '#[bank(1)]\n'
        '#[inline(never)]\n'
        'far fn f() {\n'
        '    R = 1;\n'
        '}\n'
        '#[bank(0)]\n'
        '#[entry] fn main() -> ! {\n'
        '    f();\n'
        '    loop {}\n'
        '}\n')
    seg = {s.id: s.name for s in info.segments}
    spans = {s.id: s for s in info.spans}
    f = next(s for s in info.symbols if s.name == "f")
    assert seg[f.seg_id] == "BANK1"
    line5 = [spans[i] for l in info.lines if l.line == 5 for i in l.span_ids]
    assert line5 and all(seg[s.seg_id] == "BANK1" for s in line5)
    # f's body starts at the top of bank 1, not after main's bytes in bank 0
    assert f.value == 0x8000


def test_hirom_segments_and_symbols_carry_their_bank(tmp_path):
    """HiROM banks run from $C0:0000, $C1:0000, ... as WLA-DX links them."""
    info = _compile_debug(tmp_path,
        '#[snesrom(name="T", hirom)]\n'
        '#[zeropage] static mut R: u8;\n'
        '#[bank(1)]\n'
        '#[inline(never)]\n'
        'far fn f() { R = 1; }\n'
        '#[bank(0)]\n'
        '#[entry] fn main() -> ! { f(); loop {} }\n')
    assert {s.name: s.start for s in info.segments} == {
        "BANK0": 0xC00000, "BANK1": 0xC10000}
    sym = {s.name: s.value for s in info.symbols}
    assert sym["main"] == 0xC00000 and sym["f"] == 0xC10000


def test_prologue_is_on_the_declaration_line(tmp_path):
    # Pausing in an interrupt handler's register saves shows the handler
    info = _compile_debug(tmp_path,
        '#[zeropage] static mut R: u8;\n'
        '#[interrupt(nmi)]\n'
        'fn nmi() {\n'
        '    R = 1;\n'
        '}\n'
        '#[bank(0)]\n'
        '#[entry] fn main() -> ! {\n'
        '    loop {}\n'
        '}\n')
    spans = {s.id: s for s in info.spans}
    nmi = next(s for s in info.symbols if s.name == "nmi")
    # The declaration starts at its attribute, line 2
    entry = [l.line for l in info.lines for i in l.span_ids
             if spans[i].start == nmi.value - 0x8000]
    assert entry == [2]


def test_fastrom_trampoline_is_on_the_handler(tmp_path):
    # The NMI vector lands on `__fast_nmi: JML nmi`; a pause there shows the
    # handler, and the last function's scope stops before the stubs
    info = _compile_debug(tmp_path,
        '#[snesrom(name="T", fastrom)]\n'
        '#[zeropage] static mut R: u8;\n'
        '#[interrupt(nmi)]\n'
        'fn nmi() {\n'
        '    R = 1;\n'
        '}\n'
        '#[bank(0)]\n'
        '#[entry] fn main() -> ! {\n'
        '    loop {}\n'
        '}\n')
    spans = {s.id: s for s in info.spans}
    decl = [spans[i] for l in info.lines if l.line == 3 for i in l.span_ids]
    assert 4 in [s.size for s in decl], "no line for the JML stub"
    jml = next(s for s in decl if s.size == 4)
    for scope in info.scopes:
        sym = next((s for s in info.symbols if s.id == scope.sym_id), None)
        if sym is not None:
            assert not (sym.value - 0x8000 <= jml.start < sym.value - 0x8000 + scope.size), scope.name
