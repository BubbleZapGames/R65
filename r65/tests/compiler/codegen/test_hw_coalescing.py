#!/usr/bin/env python3
"""Test hardware register coalescing optimization.

This tests the optimization that allows trivial functions to skip stack frame
allocation when a virtual register's only definition is a Move from a hardware
register and its only use is in a Return instruction.
"""

import pytest
from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.codegen.slot_allocator import StackSlotAllocator, SlotAllocation
from r65.compiler.mir.nodes import (
    MIRProgram,
    MIRFunction,
    BasicBlock,
    Move, Return, BinaryOp,
    VirtualRegister,
    Immediate,
    HardwareRegister,
)
from r65.compiler.hir.attributes import BankAttribute
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


class TestHwCoalescingSlotAllocator:
    """Test the slot allocator's hw-coalesceable detection."""

    def test_identity_function_coalesceable(self):
        """Test that identity function vreg is hw-coalesceable."""
        # Create: far fn identity(a @ A: u8) -> u8 { return a; }
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            # Move from A to vreg (parameter save)
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            # Return vreg
            Return(values=[vreg_a])
        ]

        func = MIRFunction(
            name="identity",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc
        )

        # Run slot allocator
        allocator = StackSlotAllocator(func)
        allocation = allocator.allocate()

        # Verify vreg is hw-coalesceable
        assert vreg_a in allocation.hw_coalesceable, \
            "Identity function vreg should be hw-coalesceable"
        assert allocation.hw_coalesceable[vreg_a] == 'A', \
            "Should coalesce to A register"
        assert allocation.total_slots == 0, \
            "Identity function should need 0 stack slots"

    def test_vreg_with_computation_not_coalesceable(self):
        """Test that vreg used in computation is not hw-coalesceable."""
        # Create: far fn add_one(a @ A: u8) -> u8 { return a + 1; }
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")
        vreg_result = vreg_alloc.alloc(BasicTypeInfo('u8'), "result")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            # Move from A to vreg (parameter save)
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            # a + 1 -> result
            BinaryOp(
                dest=vreg_result,
                left=vreg_a,
                op='+',
                right=Immediate(1),
                type_info=BasicTypeInfo('u8')
            ),
            # Return result
            Return(values=[vreg_result])
        ]

        func = MIRFunction(
            name="add_one",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc
        )

        # Run slot allocator
        allocator = StackSlotAllocator(func)
        allocation = allocator.allocate()

        # vreg_a is used in BinaryOp, not just Return - not coalesceable
        assert vreg_a not in allocation.hw_coalesceable, \
            "Vreg used in computation should not be hw-coalesceable"

    def test_multiple_uses_not_coalesceable(self):
        """Test that vreg with multiple uses is not hw-coalesceable."""
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            # Use vreg_a in a move (not just return)
            Move(
                dest=HardwareRegister('X'),
                source=vreg_a,
                type_info=BasicTypeInfo('u8')
            ),
            Return(values=[vreg_a])
        ]

        func = MIRFunction(
            name="multi_use",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc
        )

        allocator = StackSlotAllocator(func)
        allocation = allocator.allocate()

        # vreg_a used in Move to X AND Return - not coalesceable
        assert vreg_a not in allocation.hw_coalesceable, \
            "Vreg with multiple uses should not be hw-coalesceable"

    def test_vreg_from_immediate_not_coalesceable(self):
        """Test that vreg defined from immediate is not hw-coalesceable."""
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            # Move from immediate (not hw register)
            Move(
                dest=vreg_a,
                source=Immediate(42),
                type_info=BasicTypeInfo('u8')
            ),
            Return(values=[vreg_a])
        ]

        func = MIRFunction(
            name="const_return",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc
        )

        allocator = StackSlotAllocator(func)
        allocation = allocator.allocate()

        # vreg_a defined from immediate, not hw register
        assert vreg_a not in allocation.hw_coalesceable, \
            "Vreg from immediate should not be hw-coalesceable"


class TestHwCoalescingCodeGen:
    """Test code generation with hw coalescing."""

    def test_identity_function_no_frame(self):
        """Test that identity function generates no frame allocation."""
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            Return(values=[vreg_a])
        ]

        func = MIRFunction(
            name="identity",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            bank_attr=BankAttribute(name='bank', bank_number=1),
            vreg_allocator=vreg_alloc
        )

        program = MIRProgram(functions=[func])
        codegen = ProgramCodeGenerator()
        asm_output = codegen.generate(program)

        # Find the identity function section
        identity_start = asm_output.find('identity:')
        assert identity_start != -1, "identity function not found"

        # Get the function body (up to next label or end)
        func_section = asm_output[identity_start:identity_start + 200]
        lines = [l.strip() for l in func_section.split('\n') if l.strip()]

        # Should NOT have PHA (frame allocation)
        has_pha = any('PHA' in line for line in lines[:10])
        assert not has_pha, \
            f"Identity function should not allocate frame (no PHA)\nOutput:\n{func_section}"

        # Should have RTL
        has_rtl = any('RTL' in line for line in lines)
        assert has_rtl, "Far function should have RTL"

    def test_function_with_locals_has_frame(self):
        """Test that function with real locals allocates frame."""
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")
        vreg_temp = vreg_alloc.alloc(BasicTypeInfo('u8'), "temp")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            # Use vreg_a in computation
            BinaryOp(
                dest=vreg_temp,
                left=vreg_a,
                op='+',
                right=Immediate(1),
                type_info=BasicTypeInfo('u8')
            ),
            Return(values=[vreg_temp])
        ]

        func = MIRFunction(
            name="add_one",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            bank_attr=BankAttribute(name='bank', bank_number=1),
            vreg_allocator=vreg_alloc
        )

        program = MIRProgram(functions=[func])
        codegen = ProgramCodeGenerator()
        asm_output = codegen.generate(program)

        # Find the function section
        func_start = asm_output.find('add_one:')
        assert func_start != -1, "add_one function not found"

        func_section = asm_output[func_start:func_start + 300]

        # Should have frame allocation (PHA) since vreg_a is used in BinaryOp
        has_pha = 'PHA' in func_section
        assert has_pha, \
            f"Function with locals should allocate frame\nOutput:\n{func_section}"


def test_hw_coalescing_summary():
    """Summary test showing the optimization in action."""
    print("\n=== Hardware Register Coalescing Tests ===\n")

    # Test 1: Identity function
    vreg_alloc = VirtualRegisterAllocator()
    vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

    entry_block = BasicBlock(block_id=0)
    entry_block.instructions = [
        Move(dest=vreg_a, source=HardwareRegister('A'), type_info=BasicTypeInfo('u8')),
        Return(values=[vreg_a])
    ]

    func = MIRFunction(
        name="identity",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: entry_block},
        entry_block_id=0,
        is_far=True,
        vreg_allocator=vreg_alloc
    )

    allocator = StackSlotAllocator(func)
    allocation = allocator.allocate()

    print(f"Identity function:")
    print(f"  Total vregs: 1")
    print(f"  HW-coalesceable: {len(allocation.hw_coalesceable)}")
    print(f"  Stack slots needed: {allocation.total_slots}")

    assert allocation.total_slots == 0
    print("  Result: No frame allocation needed")
    print()
    print("=== All Tests Passed ===")


if __name__ == '__main__':
    test_hw_coalescing_summary()
