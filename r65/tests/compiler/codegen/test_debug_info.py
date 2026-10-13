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
