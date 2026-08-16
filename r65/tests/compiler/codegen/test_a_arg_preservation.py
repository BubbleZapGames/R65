# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""An A-resident value passed to an `@ A` parameter must survive argument setup.

When a call passes a value that already lives in A to an `@ A` parameter, and
another argument needs a different register, the other argument's setup routes
through A and destroys the value. Nothing reloads it, because an A-resident
value has no memory home to reload *from* — which is why the same call is
correct when the A-bound value happens to be a stack parameter.

    // stdlib/math.r65: far fn mul8(multA @ A: u8, multB @ B: u8) -> u16
    fn scaled(v @ A: u8, k: u8) -> u16 { return mul8(v, k); }

    LDA $03,S   ; k -- A held v, now destroyed
    XBA         ; B = k
    JSR mul8    ; A = garbage

The `@ A` argument itself emits nothing (`emit_register_argument` returns early
when the value is already allocated to A), so argument *ordering* cannot fix
this: the value has to be parked somewhere and put back.

Correctness is checked by symbolically executing the emitted prologue: A starts
holding the parameter, every instruction is applied to a tiny A/B/X/Y/stack
model, and at the call A must still hold it. A predicate over instruction names
is not good enough here — the motivating bug's `XBA` looks exactly like the
closing half of an `XBA`-pair save, and a correct reload of a stack-homed value
looks exactly like the `LDA` that destroys an A-homed one.
"""

import pytest
from r65.compiler.main import compile_string


def _get_function_asm(full_asm: str, func_name: str) -> str:
    lines = full_asm.split('\n')
    in_func = False
    func_lines: list[str] = []
    for line in lines:
        if line.strip() == f'{func_name}:':
            in_func = True
        elif in_func:
            if line.startswith('; ---') and func_lines:
                break
            func_lines.append(line)
    return '\n'.join(func_lines)


def _instruction_lines(asm: str) -> list[str]:
    result = []
    for line in asm.split('\n'):
        stripped = line.strip()
        if (not stripped
                or stripped.startswith(';')
                or stripped.startswith('.')
                or stripped.endswith(':')):
            continue
        result.append(stripped.split(';')[0].strip())
    return result


class Unknown:
    """A value the model cannot name — never equal to the tracked parameter."""
    def __repr__(self):
        return '<unknown>'


def simulate_to_call(instrs: list[str], entry_a='v') -> tuple:
    """Run the emitted prologue over an A/B/X/Y/stack model.

    Returns (A, B) at the first JSR/JSL. Anything the model does not understand
    poisons the destination with Unknown rather than being silently ignored, so
    an unmodelled instruction shows up as a test failure, not a false pass.
    """
    a, b, x, y = entry_a, Unknown(), Unknown(), Unknown()
    stack: list = []
    for ins in instrs:
        op = ins.split()[0]
        operand = ins[len(op):].strip()
        if op in ('JSR', 'JSL'):
            return a, b
        elif op == 'LDA':
            a = f'mem({operand})'
        elif op == 'XBA':
            a, b = b, a
        elif op == 'TAY':
            y = a
        elif op == 'TYA':
            a = y
        elif op == 'TAX':
            x = a
        elif op == 'TXA':
            a = x
        elif op == 'PHA':
            stack.append(a)
        elif op == 'PLA':
            a = stack.pop() if stack else Unknown()
        elif op in ('STA', 'STX', 'STY', 'STZ', 'LDX', 'LDY',
                    'REP', 'SEP', 'PHB', 'PLB', 'PHD', 'PLD',
                    'PHX', 'PLX', 'PHY', 'PLY', 'PHP', 'PLP',
                    'CLC', 'SEC', 'TSC', 'TCS', 'TCD', 'TDC',
                    'INX', 'INY', 'DEX', 'DEY', 'NOP'):
            if op == 'LDX':
                x = f'mem({operand})'
            elif op == 'LDY':
                y = f'mem({operand})'
            elif op == 'PLX':
                x = stack.pop() if stack else Unknown()
            elif op == 'PLY':
                y = stack.pop() if stack else Unknown()
            elif op in ('PHX', 'PHY'):
                stack.append(x if op == 'PHX' else y)
        else:
            a = Unknown()          # unmodelled: fail loudly rather than pass
    raise AssertionError(f"no call found in: {instrs}")


def caller_body(callee: str, call: str, extra: str = "",
                params: str = "v @ A: u8, k: u8", args: str = "12, 10") -> str:
    return f"""
#[zeropage(0x20)]
static mut OUT: u16;
{extra}
{callee}
fn caller({params}) -> u16 {{ return {call}; }}
#[entry]
fn main() {{ OUT = caller({args}); }}
"""


def instrs_for(source: str, func: str = "caller") -> list[str]:
    return _instruction_lines(_get_function_asm(compile_string(source), func))


def assert_a_survives(source: str, expect_b=None):
    instrs = instrs_for(source)
    a, b = simulate_to_call(instrs)
    assert a == 'v', f"A holds {a!r} at the call, expected 'v'\n  {instrs}"
    if expect_b is not None:
        assert b == expect_b, f"B holds {b!r}, expected {expect_b!r}\n  {instrs}"


MUL8 = 'far fn mul8(multA @ A: u8, multB @ B: u8) -> u16 { return 0; }'
TAKES_X = ('far fn takes_x(a @ A: u8, b @ X: u16) -> u16 '
           '{ return (a as u16) + b; }')
TAKES_Y = ('far fn takes_y(a @ A: u8, b @ Y: u16) -> u16 '
           '{ return (a as u16) + b; }')


class TestBTargetArgument:
    """A `@ B` argument's setup always routes through A — every branch of
    `_emit_b_register_argument` loads A and then XBAs."""

    def test_b_from_stack_param(self):
        assert_a_survives(caller_body(MUL8, "mul8(v, k)"))

    def test_b_from_immediate(self):
        assert_a_survives(caller_body(MUL8, "mul8(v, 9)"))

    def test_same_value_to_a_and_b(self):
        """`XBA` alone leaves B correct but A holding the *old* B."""
        assert_a_survives(caller_body(MUL8, "mul8(v, v)"), expect_b='v')


class TestIndexTargetArgument:
    """X/Y route through A only when the source is memory or stack — the
    65816 has no stack-relative LDX/LDY."""

    def test_x_from_stack_source(self):
        assert_a_survives(caller_body(TAKES_X, "takes_x(v, w)",
                                      params="v @ A: u8, w: u16"))

    def test_y_from_stack_source(self):
        assert_a_survives(caller_body(TAKES_Y, "takes_y(v, w)",
                                      params="v @ A: u8, w: u16"))

    def test_x_from_immediate_is_untouched(self):
        """`LDX #imm` never touches A, so no save may be emitted."""
        source = caller_body(TAKES_X, "takes_x(v, 7)")
        assert_a_survives(source)
        instrs = instrs_for(source)
        assert not any(s.startswith(('TAY', 'TAX', 'PHA', 'XBA'))
                       for s in instrs), f"needless save emitted: {instrs}"


class TestScratchParamArgument:
    """`Argument` is an unhashable dataclass, so the `needs_a_save` set raised
    `TypeError: unhashable type: 'Argument'` and failed the whole compilation
    rather than emitting the bracket it had already decided on."""

    SCRATCH = "#[zeropage(0x00, register)]\nstatic mut SCRATCH0: u8;"
    CALLEE = 'far fn f(a @ A: u8, b: u8) -> u16 { return b as u16; }'

    def test_compiles_at_all(self):
        assert instrs_for(caller_body(self.CALLEE, "f(v, k)", self.SCRATCH))

    def test_a_is_preserved(self):
        assert_a_survives(caller_body(self.CALLEE, "f(v, k)", self.SCRATCH))


class TestAlreadyCorrect:
    """Shapes that work today and must keep working."""

    def test_a_value_on_the_stack(self):
        """The A-bound value has a frame slot, so it can be reloaded after the
        clobber. This is why the bug is invisible in so much code."""
        instrs = instrs_for(caller_body(MUL8, "mul8(v, k)",
                                        params="v: u8, k: u8"))
        assert instrs[-1] != 'JSR mul8', instrs
        call = next(i for i, s in enumerate(instrs) if s.startswith('JSR'))
        assert instrs[call - 1].startswith('LDA'), (
            f"v must be reloaded immediately before the call: {instrs}")

    def test_no_other_register_argument(self):
        callee = 'far fn one(a @ A: u8) -> u16 { return a as u16; }'
        source = caller_body(callee, "one(v)")
        assert_a_survives(source)
        assert not any(s.startswith(('TAY', 'TAX', 'PHA', 'XBA'))
                       for s in instrs_for(source)), instrs_for(source)


class TestNoFreeIndexRegister:
    """Both X and Y are argument targets, so neither can hold the saved value."""

    CALLEE = ('far fn xy(a @ A: u8, p @ X: u16, q @ Y: u16) -> u16 '
              '{ return p + q; }')

    def test_falls_back_to_the_stack(self):
        source = caller_body(self.CALLEE, "xy(v, p, q)",
                             params="v @ A: u8, p: u16, q: u16", args="1, 2, 3")
        assert_a_survives(source)
        instrs = instrs_for(source)
        assert 'PHA' in instrs and 'PLA' in instrs, instrs

    def test_push_and_pull_widths_match(self):
        """The push happens in m8 and the index loads switch to m16, so the
        pull must be narrowed back or it takes two bytes off a one-byte push
        and unbalances the frame."""
        source = caller_body(self.CALLEE, "xy(v, p, q)",
                             params="v @ A: u8, p: u16, q: u16", args="1, 2, 3")
        instrs = instrs_for(source)
        pull = instrs.index('PLA')
        mode_before_pull = [s for s in instrs[:pull] if s.startswith(('REP', 'SEP'))]
        assert mode_before_pull[-1].startswith('SEP'), (
            f"PLA must run in m8 to match the PHA: {instrs}")


class TestSixteenBitAccumulator:
    """A u16 `@ A` argument. TAY/TYA move index-width bits under x16, so the
    whole value survives, but the restore still has to run in m16."""

    CALLEE = 'far fn wide(a @ A: u16, p @ X: u16) -> u16 { return a + p; }'

    def test_a_survives(self):
        assert_a_survives(caller_body(self.CALLEE, "wide(v, p)",
                                      params="v @ A: u16, p: u16",
                                      args="300, 4"))

    def test_restore_runs_in_m16(self):
        instrs = instrs_for(caller_body(self.CALLEE, "wide(v, p)",
                                        params="v @ A: u16, p: u16",
                                        args="300, 4"))
        restore = next(i for i, s in enumerate(instrs) if s in ('TYA', 'TXA'))
        modes = [s for s in instrs[:restore] if s.startswith(('REP', 'SEP'))]
        assert modes and modes[-1].startswith('REP'), (
            f"a 16-bit restore must run in m16: {instrs}")
