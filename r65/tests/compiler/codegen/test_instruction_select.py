#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Test instruction selection."""

from r65.compiler.codegen import (
    AssemblyEmitter,
    InstructionSelector,
    RegisterAllocator,
    ScratchRegisterPool,
    MemoryAllocator,
)
from r65.compiler.mir.nodes import (
    Load, Store, Move,
    BinaryOp, UnaryOp,
    Jump, CondBranch,
    SetMode,
    VirtualRegister, HardwareRegister,
    Immediate,
    MemoryLocation,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.symbol_table import Symbol, SymbolKind


def test_memory_operations():
    """Test Load, Store, Move instruction selection."""
    print("=" * 80)
    print("Test 1: Memory Operations (Load, Store, Move)")
    print("=" * 80)
    print()

    # Setup
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()

    # Create scratch pool
    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 1, "SCRATCH0")
    pool.add_scratch(0x17, 1, "SCRATCH1")

    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create virtual registers
    vreg0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="a")
    vreg1 = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="b")

    # Test Move with immediate
    print("Move #42 → %0:")
    move_instr = Move(
        dest=vreg0,
        source=Immediate(42),
        type_info=BasicTypeInfo('u8')
    )
    selector.select_instruction(move_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test Move register to register
    print("Move %0 → %1:")
    move_instr2 = Move(
        dest=vreg1,
        source=vreg0,
        type_info=BasicTypeInfo('u8')
    )
    selector.select_instruction(move_instr2)
    print(emitter.to_string())
    emitter.clear()
    print()



def test_arithmetic_operations():
    """Test arithmetic instruction selection."""
    print("=" * 80)
    print("Test 2: Arithmetic Operations (Add, Sub, AND, OR, XOR)")
    print("=" * 80)
    print()

    # Setup
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 1, "SCRATCH0")
    pool.add_scratch(0x17, 1, "SCRATCH1")
    pool.add_scratch(0x18, 1, "SCRATCH2")

    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Virtual registers
    vreg0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="a")
    vreg1 = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="b")
    vreg2 = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="result")

    # Test addition: result = a + b
    print("Add: %2 = %0 + %1")
    add_instr = BinaryOp(
        dest=vreg2,
        left=vreg0,
        right=vreg1,
        op='+',
        type_info=BasicTypeInfo('u8')
    )
    selector.select_instruction(add_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test addition with immediate: result = a + 5
    print("Add: %2 = %0 + #5")
    add_imm_instr = BinaryOp(
        dest=vreg2,
        left=vreg0,
        right=Immediate(5),
        op='+',
        type_info=BasicTypeInfo('u8')
    )
    selector.select_instruction(add_imm_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test subtraction
    print("Sub: %2 = %0 - #10")
    sub_instr = BinaryOp(
        dest=vreg2,
        left=vreg0,
        right=Immediate(10),
        op='-',
        type_info=BasicTypeInfo('u8')
    )
    selector.select_instruction(sub_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test bitwise AND
    print("AND: %2 = %0 & #$0F")
    and_instr = BinaryOp(
        dest=vreg2,
        left=vreg0,
        right=Immediate(0x0F),
        op='&',
        type_info=BasicTypeInfo('u8')
    )
    selector.select_instruction(and_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test shift left
    print("Shift Left: %2 = %0 << 3")
    shl_instr = BinaryOp(
        dest=vreg2,
        left=vreg0,
        right=Immediate(3),
        op='<<',
        type_info=BasicTypeInfo('u8')
    )
    selector.select_instruction(shl_instr)
    print(emitter.to_string())
    emitter.clear()
    print()



def test_control_flow():
    """Test control flow instruction selection."""
    print("=" * 80)
    print("Test 3: Control Flow (Jump, Branch)")
    print("=" * 80)
    print()

    # Setup
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 1, "SCRATCH0")
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Test Jump
    print("Jump to block 5:")
    jump_instr = Jump(target=5)
    selector.select_instruction(jump_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test conditional branch
    print("Conditional branch: if %0 != 0 goto block 10 else block 20:")
    vreg0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="cond")
    branch_instr = CondBranch(
        condition=vreg0,
        true_target=10,
        false_target=20,
        comparison='!='
    )
    selector.select_instruction(branch_instr)
    print(emitter.to_string())
    emitter.clear()
    print()




def test_mode_control():
    """Test mode control instruction selection."""
    print("=" * 80)
    print("Test 4: Mode Control (SEP, REP)")
    print("=" * 80)
    print()

    # Setup
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    reg_alloc = RegisterAllocator()
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Test SEP
    print("SEP #$30 (set M and X flags - 8-bit mode):")
    sep_instr = SetMode(mask=0x30, is_set=True)
    selector.select_instruction(sep_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test REP
    print("REP #$30 (reset M and X flags - 16-bit mode):")
    rep_instr = SetMode(mask=0x30, is_set=False)
    selector.select_instruction(rep_instr)
    print(emitter.to_string())
    emitter.clear()
    print()



def test_16bit_operations():
    """Test 16-bit operations."""
    print("=" * 80)
    print("Test 6: 16-bit Operations")
    print("=" * 80)
    print()

    # Setup
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 2, "SCRATCH0")  # 2-byte scratch
    pool.add_scratch(0x18, 2, "SCRATCH1")  # 2-byte scratch

    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create 16-bit virtual registers
    vreg0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint="word")
    vreg1 = VirtualRegister(id=1, type_info=BasicTypeInfo('u16'), hint="result")

    # Test 16-bit move
    print("Move (16-bit): %1 = #$1234")
    move_instr = Move(
        dest=vreg1,
        source=Immediate(0x1234),
        type_info=BasicTypeInfo('u16')
    )
    selector.select_instruction(move_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test 16-bit addition
    print("Add (16-bit): %1 = %0 + #$0100")
    add_instr = BinaryOp(
        dest=vreg1,
        left=vreg0,
        right=Immediate(0x0100),
        op='+',
        type_info=BasicTypeInfo('u16')
    )
    selector.select_instruction(add_instr)
    print(emitter.to_string())
    emitter.clear()
    print()



if __name__ == "__main__":
    print("=" * 80)
    print("Instruction Selection Tests")
    print("=" * 80)
    print()

    # Run tests
    test_memory_operations()
    test_arithmetic_operations()
    test_control_flow()
    test_mode_control()
    test_16bit_operations()

    print("🎉 All tests passed!")


# =============================================================================
# select_set_mode emitter-mode tracking
# =============================================================================
# Regression tests for the bug where select_set_mode emitted SEP/REP but
# left the emitter's tracked accu_mode stale, causing a subsequent
# _ensure_mX_mode check to skip a needed SEP/REP. Surfaced when the
# inliner started inserting MIR-level SetMode at inline boundaries: a
# u16 STA after an exit-boundary SEP silently dropped the high byte
# because the codegen still believed it was in m16.

def _fresh_selector():
    emitter = AssemblyEmitter()
    selector = InstructionSelector(emitter, RegisterAllocator(), MemoryAllocator())
    return emitter, selector


def test_select_set_mode_sep_m_flag_updates_accu_to_m8():
    emitter, selector = _fresh_selector()
    emitter.emit_accu_mode(16)
    selector.select_instruction(SetMode(mask=0x20, is_set=True))
    assert emitter.get_accu_mode() == 8


def test_select_set_mode_rep_m_flag_updates_accu_to_m16():
    emitter, selector = _fresh_selector()
    emitter.emit_accu_mode(8)
    selector.select_instruction(SetMode(mask=0x20, is_set=False))
    assert emitter.get_accu_mode() == 16


def test_select_set_mode_x_only_does_not_change_accu():
    """SEP/REP touching only the X flag (mask=0x10) must not affect the
    emitter's accu (M-flag) tracking. R65 keeps X always at x16, so the
    M tracker should stay where it was."""
    emitter, selector = _fresh_selector()
    emitter.emit_accu_mode(16)
    selector.select_instruction(SetMode(mask=0x10, is_set=True))
    assert emitter.get_accu_mode() == 16
    selector.select_instruction(SetMode(mask=0x10, is_set=False))
    assert emitter.get_accu_mode() == 16


def test_select_set_mode_combined_mx_mask_updates_accu():
    """SEP #$30 sets both M and X — the M-flag effect must reach the
    accu tracker even when the X bit is also present."""
    emitter, selector = _fresh_selector()
    emitter.emit_accu_mode(16)
    selector.select_instruction(SetMode(mask=0x30, is_set=True))
    assert emitter.get_accu_mode() == 8


def test_status_flag_read_uses_immediates_in_m8():
    """`STATUS.Decimal as u8` after u16 work: PHP pushes one byte, so the
    PLA runs in m8, and the mask and the normalized 1 are immediates."""
    from r65.compiler.main import compile_string
    asm = compile_string('''
        #[zeropage(0x10)] static mut R: u8;
        #[zeropage(0x12)] static mut W: u16;
        #[entry] fn main() { W = W + 1; R = STATUS.Decimal as u8; }
    ''', cfg_options=["snes"])
    body = asm.split("; StatusFlagRead Decimal", 1)[1]
    lines = [l.split(";")[0].strip() for l in body.splitlines()]
    lines = [l for l in lines if l and not l.startswith(".")]
    assert lines[:4] == ["SEP #$20", "PHP", "PLA", "AND #$08"]
    assert "LDA #$01" in lines


def test_emit_instr_rejects_bare_int_operand():
    """A bare int would print as an address (`AND 8`), so it is refused."""
    import pytest
    from r65.compiler.codegen.opcodes import Opcode
    from r65.compiler.errors import InternalCompilerError
    with pytest.raises(InternalCompilerError):
        AssemblyEmitter().emit_instr(Opcode.AND_IMMEDIATE, 8)


def test_memory_fill_stores_wram_long_and_bytes_in_m8():
    """A #[ram] fill stores through long,X (4 bytes, as WLA-DX assembles a
    24-bit operand), and a byte fill after u16 work runs in m8."""
    from r65.compiler.frontend import parse
    from r65.compiler.hir import HIRBuilder
    from r65.compiler.typeck import TypeChecker
    from r65.compiler.mir import MIRBuilder
    from r65.compiler.codegen import ProgramCodeGenerator
    from r65.compiler.codegen.asm_nodes import Address, Instruction
    from r65.compiler.codegen.opcodes import Opcode, instruction_size
    hir = HIRBuilder().build_program(parse('''
        #[ram] static mut BUF: [u8; 16] = [5; 16];
        #[lowram] static mut W: u16;
        #[zeropage(0x12)] static mut Q: u16;
        #[zeropage(0x10)] static mut R: u8;
        fn work(i @ X: u16) -> u8 {
            Q = W + 1;
            let mut buf: [u8; 8] = [7; 8];
            buf[i] = 3;
            return buf[1] + buf[i];
        }
        #[entry] fn main() { R = BUF[3] + work(2); }
    '''))
    TypeChecker(hir).check()
    gen = ProgramCodeGenerator()
    asm = gen.generate(MIRBuilder().build_program(hir), opt_level=0)
    store = next(n for n in gen._last_final_nodes if isinstance(n, Instruction)
                 and n.operand == Address(0x7E2000))
    assert store.opcode == Opcode.STA_LONG_X
    assert instruction_size(store.opcode) == 4
    fill = asm.split("; Fill 8 x 1B elements with #7", 1)[1].splitlines()
    assert fill[1].split(";")[0].strip() == "SEP #$20"


def test_pointer_spill_skips_a_scratch_holding_a_live_value(codegen):
    """A pointer spilled from a register borrows a free scratch, not the first."""
    from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind
    codegen.pool.add_scratch(0x10, 2, "SCRATCH0")
    codegen.pool.add_scratch(0x12, 2, "SCRATCH1")
    codegen.pool.allocate(VirtualRegister(id=0, type_info=BasicTypeInfo('u16')))
    spilled = codegen.selector.memory_selector._spill_pointer_to_scratch(
        PhysicalLocation(kind=LocationKind.HARDWARE, hw_register='X', size=2))
    assert spilled.scratch_addr == 0x12
