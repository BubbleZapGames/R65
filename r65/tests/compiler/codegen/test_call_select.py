# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Call selection: argument setup and cleanup around calls."""

from r65.compiler.codegen import (
    AssemblyEmitter,
    InstructionSelector,
    MemoryAllocator,
    RegisterAllocator,
    ScratchRegisterPool,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.nodes import (
    Argument, ArgumentMechanism, Call, HardwareRegister,
)


def _selector():
    emitter = AssemblyEmitter()
    selector = InstructionSelector(emitter, RegisterAllocator(scratch_pool=ScratchRegisterPool()),
                                   MemoryAllocator())
    return emitter, selector


def _lines(emitter) -> list[str]:
    return [line.split(';')[0].strip() for line in emitter.to_string().splitlines()
            if line.strip() and not line.strip().startswith(('.', ';'))]


def _select(instr) -> list[str]:
    emitter, selector = _selector()
    selector.select_instruction(instr)
    return _lines(emitter)


def test_runtime_builtin_moves_y_argument_with_tyx():
    """A TYA/TAX detour gives X a stale high byte in m8 (TAX takes B:A)."""
    u16 = BasicTypeInfo('u16')
    call = Call(function='shl', builtin_name='shl', args=[
        Argument(value=HardwareRegister('A'), mechanism=ArgumentMechanism.REGISTER,
                 location=HardwareRegister('A'), param_type=u16),
        Argument(value=HardwareRegister('Y'), mechanism=ArgumentMechanism.REGISTER,
                 location=HardwareRegister('X'), param_type=u16),
    ])
    assert _select(call) == ['TYX', 'JSR __builtin_shl']


def test_d_equals_s_restore_with_a_x_y_live_parks_two_bytes_in_m16():
    """PHA and PLA must agree in width, and D must step over what PHA pushed."""
    emitter, selector = _selector()
    emitter.emit_accu_mode(16)
    selector.call_selector._emit_d_equals_s_restore({'A', 'X', 'Y'})
    assert _lines(emitter) == ['PHD', 'PHA', 'TSC', 'INC A', 'INC A', 'TCD', 'PLA']


def test_d_equals_s_restore_keeps_results_captured_after_the_call():
    """`let a, b = f()` leaves a in A and b in B past the D=S restore."""
    from r65.compiler.main import compile_string
    source = """
    #[zeropage]
    static mut OUT: u8;
    fn two() -> u8, u8 { return 3, 4; }
    #[used]
    fn get(p: far *u8, q: far *u8) {
        let v: u8 = *p;
        let a, b = two();
        *q = v + a;
        OUT = b;
    }
    #[entry]
    fn main() -> ! { get(0x7E2000 as far *u8, 0x7E2001 as far *u8); loop {} }
    """
    asm = compile_string(source, "test.r65")
    after = asm.split('JSR two', 1)[1].split('CLC', 1)[0]
    body = [line.split(';')[0].strip() for line in after.splitlines()]
    body = [op for op in body if op and not op.startswith('.')]
    assert body == ['PHD', 'REP #$20', 'PHA', 'TSC', 'INC A', 'INC A', 'TCD',
                    'PLA', 'SEP #$20']
