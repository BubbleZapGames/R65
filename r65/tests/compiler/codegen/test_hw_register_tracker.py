# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""HardwareRegisterTracker: is a register-bound value still to be read?"""

from r65.compiler.codegen.function_gen import compute_block_order
from r65.compiler.codegen.hw_register_tracker import HardwareRegisterTracker
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.main import compile_string
from r65.compiler.mir.nodes import (
    BasicBlock, HardwareRegister, Immediate, Jump, MIRFunction, MemoryLocation,
    Move, Return, Store,
)
from r65.compiler.mir.register_tracker import RegisterAliasTracker


def test_emission_order_not_block_id_order():
    """Block 1 returns, so it is emitted after block 2; its read of A still
    follows block 2's store."""
    u8, u16 = BasicTypeInfo('u8'), BasicTypeInfo('u16')
    func = MIRFunction(name='f')
    func.entry_block_id = 0
    func.alias_tracker = RegisterAliasTracker()
    func.alias_tracker.add_alias(object(), HardwareRegister('A'), 0)
    func.blocks = {
        0: BasicBlock(0, [Move(HardwareRegister('A'), Immediate(9), u8), Jump(2)],
                      successors=[1, 2]),
        1: BasicBlock(1, [Store(HardwareRegister('A'),
                                MemoryLocation('zeropage', 0x10, None), u8),
                          Return()]),
        2: BasicBlock(2, [Move(HardwareRegister('X'), Immediate(1), u16),
                          Store(HardwareRegister('Y'), MemoryLocation('ram', None, None), u16),
                          Jump(1)], successors=[1]),
    }
    assert compute_block_order(func) == [0, 2, 1]

    tracker = HardwareRegisterTracker()
    tracker.initialize_from_parameters(func)
    tracker.advance_to(3)                 # block 2's Store Y
    assert not tracker.is_free('A')
    tracker.advance_to(6)                 # block 1's Return, after the read
    assert tracker.is_free('A')


def test_read_across_a_back_edge_keeps_a_saved():
    """`OUT = v` runs again on the next iteration, so the TYA for the
    store into RAM must not destroy A."""
    asm = compile_string('''
#[lowram] static mut OUT: u8;
static mut SUM: u16;
fn f() -> ! {
    let v @ A = 9;
    Y = 0;
    loop { OUT = v; SUM = Y; Y++; }
}
#[entry] fn main() -> ! { SUM = 0; f(); }
''')
    body = asm[asm.index('f__L2:'):]
    assert body.index('PHA') < body.index('TYA') < body.index('PLA')
