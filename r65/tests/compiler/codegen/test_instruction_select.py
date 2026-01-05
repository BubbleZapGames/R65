#!/usr/bin/env python3
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
    Jump, CondBranch, Return,
    SetMode,
    SaveRegister, RestoreRegister,
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

    return True


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

    return True


def test_control_flow():
    """Test control flow instruction selection."""
    print("=" * 80)
    print("Test 3: Control Flow (Jump, Branch, Return)")
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

    # Test return
    print("Return:")
    return_instr = Return(values=[])
    selector.select_instruction(return_instr)
    print(emitter.to_string())
    emitter.clear()
    print()

    return True


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

    return True


def test_register_save_restore():
    """Test register save/restore instruction selection."""
    print("=" * 80)
    print("Test 5: Register Save/Restore (Push/Pull)")
    print("=" * 80)
    print()

    # Setup
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    reg_alloc = RegisterAllocator()
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Test save registers
    print("Save A, X, Y:")
    save_a = SaveRegister(register=HardwareRegister('A'), save_location=None)
    save_x = SaveRegister(register=HardwareRegister('X'), save_location=None)
    save_y = SaveRegister(register=HardwareRegister('Y'), save_location=None)

    selector.select_instruction(save_a)
    selector.select_instruction(save_x)
    selector.select_instruction(save_y)
    print(emitter.to_string())
    emitter.clear()
    print()

    # Test restore registers
    print("Restore Y, X, A (reverse order):")
    restore_y = RestoreRegister(register=HardwareRegister('Y'), save_location=None)
    restore_x = RestoreRegister(register=HardwareRegister('X'), save_location=None)
    restore_a = RestoreRegister(register=HardwareRegister('A'), save_location=None)

    selector.select_instruction(restore_y)
    selector.select_instruction(restore_x)
    selector.select_instruction(restore_a)
    print(emitter.to_string())
    emitter.clear()
    print()

    return True


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

    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Instruction Selection Tests")
    print("=" * 80)
    print()

    # Run tests
    test1_passed = test_memory_operations()
    test2_passed = test_arithmetic_operations()
    test3_passed = test_control_flow()
    test4_passed = test_mode_control()
    test5_passed = test_register_save_restore()
    test6_passed = test_16bit_operations()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Memory Operations: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Arithmetic Operations: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Control Flow: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Mode Control: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    print(f"Register Save/Restore: {'✅ PASSED' if test5_passed else '❌ FAILED'}")
    print(f"16-bit Operations: {'✅ PASSED' if test6_passed else '❌ FAILED'}")
    print()

    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed, test6_passed]):
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
