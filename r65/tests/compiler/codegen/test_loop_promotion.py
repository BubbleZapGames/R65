# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Test loop register promotion and codegen quality for loop-heavy functions.

Verifies that the compiler produces near-optimal assembly for common loop
patterns: strlen (far pointer scan), memset (far pointer fill), and simple
counting loops. These tests assert the ABSENCE of unnecessary instructions
rather than exact instruction sequences, making them resilient to minor
codegen changes while catching regressions.
"""

import pytest
from r65.compiler.main import compile_string


def _get_function_asm(full_asm: str, func_name: str) -> str:
    """Extract assembly for a single function from full compiler output."""
    lines = full_asm.split('\n')
    in_func = False
    func_lines = []
    for line in lines:
        if line.strip() == f'{func_name}:':
            in_func = True
        elif in_func:
            # Stop at next function header comment or end
            if line.startswith('; ---') and func_lines:
                break
            func_lines.append(line)
    return '\n'.join(func_lines)


def _instruction_lines(asm: str) -> list[str]:
    """Extract only instruction lines (no labels, directives, comments, blanks)."""
    result = []
    for line in asm.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(';'):
            continue
        if stripped.startswith('.'):
            continue
        if stripped.endswith(':'):
            continue
        result.append(stripped)
    return result


class TestStrlenCodegen:
    """
    Verify strlen generates near-optimal assembly.

    Optimal strlen loop for far *u8 with null check:
        LDY #$00
      loop:
        LDA [dp],Y      ; load char through far pointer
        BEQ done         ; exit if null
        INY              ; len++
        BRA loop         ; next char
      done:
        TYA              ; return len in A (u16)
        RTL

    The loop body should be exactly: LDA [dp],Y / BEQ / INY / BRA
    (~14 cycles per character).
    """

    STRLEN_SOURCE = """
    far fn strlen(s: far *u8) -> u16 {
        let mut len: u16 = 0;
        loop {
            if s[len] == 0 {
                return len;
            }
            len++;
        }
    }
    """

    def test_loop_uses_iny(self):
        """Loop counter `len` should be promoted to Y, using INY."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        assert 'INY' in func, f"len++ should use INY, got:\n{func}"

    def test_no_stack_frame_for_len(self):
        """Promoted loop counter should not use stack-relative addressing."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        instrs = _instruction_lines(func)
        # No ADC (would indicate LDA/CLC/ADC/STA increment pattern)
        adc_instrs = [i for i in instrs if i.startswith('ADC') or i.startswith('CLC')]
        assert not adc_instrs, (
            f"Loop counter should use INY not ADC chain, found: {adc_instrs}\n{func}")

    def test_uses_indirect_long_y(self):
        """Far pointer dereference should use [dp],Y indirect long addressing."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        assert '],Y' in func, f"Should use [dp],Y addressing, got:\n{func}"

    def test_loop_body_instruction_count(self):
        """Loop body should be at most 4 instructions: LDA, BEQ, INY, BRA."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        instrs = _instruction_lines(func)

        # Find the loop: from the LDA [dp],Y to the BRA back
        loop_start = None
        loop_end = None
        for i, inst in enumerate(instrs):
            if 'LDA' in inst and '],Y' in inst:
                loop_start = i
            if loop_start is not None and inst.startswith('BRA'):
                loop_end = i
                break

        assert loop_start is not None, f"Could not find LDA [dp],Y in:\n{func}"
        assert loop_end is not None, f"Could not find BRA after LDA in:\n{func}"

        loop_body = instrs[loop_start:loop_end + 1]
        assert len(loop_body) <= 4, (
            f"Loop body should be <= 4 instructions, got {len(loop_body)}: "
            f"{loop_body}\n{func}")

    def test_no_mode_switch_in_loop(self):
        """Loop body should not contain REP/SEP mode switches."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')

        # Find loop body (between first label after LDY and the BRA back)
        lines = func.split('\n')
        in_loop = False
        loop_lines = []
        for line in lines:
            stripped = line.strip()
            if 'LDA' in stripped and '],Y' in stripped:
                in_loop = True
            if in_loop:
                loop_lines.append(stripped)
                if stripped.startswith('BRA'):
                    break

        loop_text = '\n'.join(loop_lines)
        assert 'REP' not in loop_text, f"Loop should not have REP, got:\n{loop_text}"
        assert 'SEP' not in loop_text, f"Loop should not have SEP, got:\n{loop_text}"

    def test_return_uses_tya(self):
        """Return path should transfer Y to A (TYA) for u16 return."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        assert 'TYA' in func, f"Should return via TYA, got:\n{func}"

    def test_minimal_prologue(self):
        """Prologue should only have PHD/TSC/TCD (far pointer setup), no frame."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        instrs = _instruction_lines(func)

        # Should NOT have PHX/PHY for frame allocation (len is in Y, no frame needed)
        prologue = instrs[:6]  # First few instructions
        frame_alloc = [i for i in prologue if i in ('PHX', 'PHY', 'PHA')]
        assert not frame_alloc, (
            f"Should not allocate stack frame, found: {frame_alloc}\n{func}")


class TestTypeConvertDestNotPromoted:
    """
    A vreg that is the DEST of a widening TypeConvert must not be promoted
    to X/Y: codegen writes the conversion result with STA, which cannot
    resolve a hardware register as memory operand (regression: raised
    "Cannot resolve hardware register X as memory operand").
    """

    SOURCE = """
    #[lowram]
    static mut DIGITS: [u8; 6];

    fn bump(pos: u8, n: u8) {
        let mut p: u16 = pos as u16;
        DIGITS[p] += n;
        while DIGITS[p] >= 10 {
            if p == 0 { return; }
            DIGITS[p] -= 10;
            p--;
            DIGITS[p] += 1;
        }
    }
    """

    def test_compiles(self):
        compile_string(self.SOURCE)


class TestCounterStoredToMemoryIsStillPromoted:
    """A loop counter that is written to memory used to be refused promotion.

    `_uses_compatible_with_hw` rejected any vreg used as the source of a Store,
    on the grounds that codegen would resolve that source as a memory operand.
    Both store paths test for a hardware source first and route it through A, so
    the rejection bought nothing and cost the whole loop body -- a frame slot
    reloaded on every iteration -- to save one transfer after the loop.

    The rejection was also self-defeating: an `as u8` cast interposes a Move, so
    the Store's source became the cast result rather than the counter, and the
    counter was promoted anyway.
    """

    ARR = "#[ram]\nstatic mut ARR: [u8; 64];\n"

    def source(self, ty: str, out: str) -> str:
        return (self.ARR + out +
                f"fn go() {{ let mut i: {ty} = 0;"
                f" while i < 8 {{ ARR[i] = 1; i = i + 1; }} OUT = i; }}\n"
                "#[entry]\nfn main() { go(); }")

    def instrs(self, ty: str, out: str) -> list[str]:
        return _instruction_lines(
            _get_function_asm(compile_string(self.source(ty, out)), "go"))

    @pytest.mark.parametrize("ty,out", [
        ("u16", "#[zeropage(0x10)]\nstatic mut OUT: u16;\n"),
        ("u8", "#[zeropage(0x10)]\nstatic mut OUT: u8;\n"),
    ])
    def test_counter_stays_in_an_index_register(self, ty, out):
        emitted = self.instrs(ty, out)
        assert any(i.startswith(('INX', 'INY')) for i in emitted), (
            f"counter should increment in place: {emitted}")
        assert any(i.startswith(('CPX', 'CPY')) for i in emitted), (
            f"comparison should use the index register: {emitted}")

    @pytest.mark.parametrize("ty,out", [
        ("u16", "#[zeropage(0x10)]\nstatic mut OUT: u16;\n"),
        ("u8", "#[zeropage(0x10)]\nstatic mut OUT: u8;\n"),
    ])
    def test_no_per_iteration_frame_traffic(self, ty, out):
        """The point of promoting: the counter is not reloaded every iteration."""
        emitted = self.instrs(ty, out)
        assert not any(i.startswith('LDA') and ',S' in i.split(';')[0] for i in emitted), (
            f"counter should not live in a frame slot: {emitted}")

    def test_one_byte_destination_still_stores_one_byte(self):
        """The interaction with the STX/STY width rule: now that this counter is
        promoted, its store must not become a two-byte STX."""
        emitted = self.instrs("u8", "#[zeropage(0x10)]\nstatic mut OUT: u8;\n")
        assert not any(i.startswith(('STX', 'STY')) for i in emitted), (
            f"a u8 destination must not be written with STX/STY: {emitted}")
        assert any(i.startswith(('TXA', 'TYA')) for i in emitted), emitted

    def test_two_byte_destination_stores_directly(self):
        emitted = self.instrs("u16", "#[zeropage(0x10)]\nstatic mut OUT: u16;\n")
        assert any(i.startswith(('STX', 'STY')) for i in emitted), (
            f"a u16 destination should store directly: {emitted}")


class TestCounterAsIndirectStoreSource:
    """The other half of the relaxed guard: `StoreIndirect` with the counter as
    source.

    Riskier than a plain Store because `(zp),Y` addressing wants Y for the index
    while a promoted counter also wants Y. When the index and the value are the
    same variable that is harmless -- Y legitimately serves both. When they
    differ, the contention has to be resolved rather than ignored.
    """

    DECL = ("#[zeropage(0x40)]\nstatic mut PTR: *u8;\n"
            "#[lowram]\nstatic mut BUF: [u8; 32];\n"
            "#[zeropage(0x50)]\nstatic mut K: u8;\n")

    def instrs(self, body: str) -> list[str]:
        src = (self.DECL + "#[entry]\nfn main() { PTR = &BUF; K = 0;"
               " let mut i: u8 = 0; " + body + " }")
        return _instruction_lines(_get_function_asm(compile_string(src), "main"))

    def test_index_and_value_are_the_same_counter(self):
        """`PTR[i] = i` — Y holds the counter and indexes with it, both correct."""
        emitted = self.instrs("while i < 8 { PTR[i] = i; i = i + 1; }")
        assert any(i.startswith('INY') for i in emitted), (
            f"counter should stay in Y: {emitted}")
        assert any(',Y' in i.split(';')[0] for i in emitted), (
            f"the store should index through Y: {emitted}")

    def test_index_and_value_differ(self):
        """`PTR[K] = i` — the index needs Y, so the counter cannot also hold it.

        Promotion may still set the hint; the register allocator is what resolves
        the contention. This pins the *outcome* rather than the mechanism: the
        emitted code must not use one Y for both roles.
        """
        emitted = self.instrs(
            "while i < 8 { PTR[K] = i; i = i + 1; K = K + 2; }")
        store = next(i for i, s in enumerate(emitted) if ',Y' in s.split(';')[0])
        # Whatever loads Y for the index must be the last write to Y before the
        # store -- an INY of a counter in between would corrupt the index.
        y_writes = [s for s in emitted[:store]
                    if s.startswith(('TAY', 'LDY', 'INY', 'DEY', 'PLY'))]
        assert y_writes and y_writes[-1].startswith(('TAY', 'LDY')), (
            f"Y must hold the index at the store, not a counter: {emitted}")


class TestNestedIfIsNotALoop:
    """Loop detection must not read a forward jump to a lower block ID as a loop.

    Block IDs follow lowering order, so an outer `if`'s merge block is numbered
    before the inner `if`'s blocks, and the inner one's jump to it goes to a
    lower ID. Treated as a back-edge, that made the whole function a "loop":
    `oy++` became a counter pinned to Y, and codegen failed on the load into
    it ("Cannot resolve hardware register Y as memory operand") — classickong's
    barrel update hit this.
    """

    SOURCE = """
    #[lowram] static mut XS: [u8; 16];
    #[lowram] static mut YS: [u8; 16];
    #[lowram] static mut DX: [i8; 16];
    #[lowram] static mut FRAME: u16;
    far fn probe(x: u8, y: u16) -> u8 { return x; }
    fn step(i: u16) {
        let ox: u8 = XS[i];
        let mut oy: u8 = YS[i];
        if (FRAME & 1) == 0 {
            oy++;
            let floor: u8 = probe(ox + 8, oy + 14);
            if (floor & 4) != 0 {
                DX[i] = 0 - DX[i];
            }
        }
        YS[i] = oy;
    }
    fn count(n: u16) {
        let mut i: u16 = 0;
        while i < n { YS[i] = 0; i = i + 1; }
    }
    """

    def _loops(self, name):
        from r65.compiler.frontend import parse, preprocess, expand_macros
        from r65.compiler.hir import HIRBuilder
        from r65.compiler.typeck import TypeChecker
        from r65.compiler.mir import MIRBuilder
        from r65.compiler.analysis.loop_analysis import find_loops
        program = expand_macros(preprocess(parse(self.SOURCE, 't.r65'), 't.r65'))
        hir = HIRBuilder(source_file='t.r65').build_program(program)
        TypeChecker(hir).check()
        mir = MIRBuilder().build_program(hir)
        return find_loops(next(f for f in mir.functions if f.name == name))

    def test_nested_if_has_no_loop(self):
        assert self._loops('step') == []

    def test_a_real_loop_is_still_found(self):
        assert len(self._loops('count')) == 1


class TestCounterAsCallArgument:
    """A counter passed to a call used to be refused promotion outright.

    Stack, scratch and variable-bound arguments are set up before any register
    argument and read a counter in X or Y in place. A register argument is
    safe too, unless another register argument of the same call targets X or
    Y: register arguments load in a fixed order that does not see what the
    others read, so that one could overwrite the counter first. Values:
    tests/e2e/test_counter_as_call_argument.py.
    """

    def instrs(self, callee: str, ty: str, call: str) -> list[str]:
        src = (f"#[zeropage(0x30)]\nstatic mut VB: u16;\n#[zeropage(0x32)]\nstatic mut M: u16;\n"
               f"#[inline(never)] {callee}\n"
               f"#[inline(never)] fn go() {{ let mut i: {ty} = 0; while i < 4 {{ {call} i = i + 1; }} }}\n"
               "#[entry]\nfn main() { go(); }")
        return _instruction_lines(_get_function_asm(compile_string(src), "go"))

    @pytest.mark.parametrize("callee,ty", [
        ("fn f(v: u16) { VB = v; }", "u16"),
        ("fn f(v: u8) { VB = v as u16; }", "u8"),
        ("fn f(v @ A: u8) { VB = v as u16; }", "u8"),
        ("fn f(v @ Y: u16) { VB = v; }", "u16"),
        ("fn f(v @ VB: u16) { M = v; }", "u16"),
    ], ids=["stack u16", "stack u8", "@A", "@Y", "variable-bound"])
    def test_counter_stays_in_an_index_register(self, callee, ty):
        emitted = self.instrs(callee, ty, "f(i);")
        assert any(s.startswith(('INX', 'INY')) for s in emitted), emitted

    def test_not_beside_another_index_register_argument(self):
        """`b` targets Y, so a counter fed to `a` could be overwritten first."""
        emitted = self.instrs("fn f(a @ X: u16, b @ Y: u16) { VB = a + b; }", "u16", "f(i, M);")
        assert not any(s.startswith(('INX', 'INY')) for s in emitted), emitted
