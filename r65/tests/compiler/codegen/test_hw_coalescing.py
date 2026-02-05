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
    Move, Return, BinaryOp, Load,
    VirtualRegister,
    Immediate,
    HardwareRegister,
    MemoryLocation,
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

    def test_vreg_with_adjacent_computation_coalesceable(self):
        """Test that vreg used in immediately-following computation IS hw-coalesceable.

        When vreg_a is defined by Move from A and used in the very next instruction
        (BinaryOp), there are no intervening instructions that could clobber A,
        so the vreg can safely stay in A.
        """
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
            # a + 1 -> result (uses vreg_a immediately, no clobber between def and use)
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

        # vreg_a is used in BinaryOp immediately after def - no intervening clobbers,
        # so it IS coalesceable with the extended clobber analysis
        assert vreg_a in allocation.hw_coalesceable, \
            "Vreg used in adjacent computation should be hw-coalesceable (no clobber)"
        assert allocation.hw_coalesceable[vreg_a] == 'A'

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

        # Should NOT have PHB (frame allocation)
        has_phb = any('PHB' in line for line in lines[:10])
        assert not has_phb, \
            f"Identity function should not allocate frame (no PHB)\nOutput:\n{func_section}"

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

        # Should have frame allocation (PHB) since vreg_a is used in BinaryOp
        has_phb = 'PHB' in func_section
        assert has_phb, \
            f"Function with locals should allocate frame\nOutput:\n{func_section}"


class TestBRegisterReturns:
    """Test that PLB frame cleanup doesn't clobber any return registers."""

    def test_return_b_uses_plb_frame_cleanup(self):
        """Test that returning B uses PLB for frame cleanup (doesn't clobber B)."""
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            # Save parameter to create a frame
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            # Use vreg in a computation (forces frame allocation)
            BinaryOp(
                dest=vreg_a,
                left=vreg_a,
                op='+',
                right=Immediate(1),
                type_info=BasicTypeInfo('u8')
            ),
            # Move result to B and return B
            Move(
                dest=HardwareRegister('B'),
                source=vreg_a,
                type_info=BasicTypeInfo('u8')
            ),
            # Return B register directly
            Return(values=[HardwareRegister('B')])
        ]

        func = MIRFunction(
            name="return_b",
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
        func_start = asm_output.find('return_b:')
        assert func_start != -1, "return_b function not found"

        func_section = asm_output[func_start:func_start + 500]

        # Should use PLB for frame cleanup (doesn't clobber A, X, Y, or B)
        has_plb = 'PLB' in func_section
        assert has_plb, \
            f"Function returning B should use PLB for frame cleanup\nOutput:\n{func_section}"

        # Should NOT need XBA workaround or TSC/TCS for small frame
        has_xba = 'XBA' in func_section and 'Save A in B' in func_section
        assert not has_xba, \
            f"Function returning B should NOT use XBA workaround\nOutput:\n{func_section}"

    def test_return_a_no_frame_with_coalescence(self):
        """Test that returning A with coalesceable vreg needs no frame at all.

        When vreg_a is defined from A and used in the immediately-following
        BinaryOp, it coalesces to A, eliminating the need for stack storage.
        With no stack slots needed, no frame is allocated (no PHB/PLB).
        """
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            BinaryOp(
                dest=vreg_a,
                left=vreg_a,
                op='+',
                right=Immediate(1),
                type_info=BasicTypeInfo('u8')
            ),
            # Return A register (not B)
            Return(values=[HardwareRegister('A')])
        ]

        func = MIRFunction(
            name="return_a",
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

        func_start = asm_output.find('return_a:')
        assert func_start != -1, "return_a function not found"

        func_section = asm_output[func_start:func_start + 400]

        # vreg_a coalesces to A, so no frame needed at all
        has_phb = 'PHB' in func_section
        assert not has_phb, \
            f"Function with coalesceable vreg should not allocate frame\nOutput:\n{func_section}"

        # Should have RTL (far function)
        has_rtl = 'RTL' in func_section
        assert has_rtl, f"Far function should have RTL\nOutput:\n{func_section}"

    def test_return_a_and_b_no_frame_with_coalescence(self):
        """Test that returning both A and B with coalesceable vreg needs no frame.

        Same as test_return_a_no_frame_with_coalescence but with both A and B
        in the return values. The vreg still coalesces, so no frame is needed.
        """
        vreg_alloc = VirtualRegisterAllocator()
        vreg_a = vreg_alloc.alloc(BasicTypeInfo('u8'), "a")

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            Move(
                dest=vreg_a,
                source=HardwareRegister('A'),
                type_info=BasicTypeInfo('u8')
            ),
            BinaryOp(
                dest=vreg_a,
                left=vreg_a,
                op='+',
                right=Immediate(1),
                type_info=BasicTypeInfo('u8')
            ),
            # Return both A and B
            Return(values=[HardwareRegister('A'), HardwareRegister('B')])
        ]

        func = MIRFunction(
            name="return_a_b",
            parameters=[],
            return_type=BasicTypeInfo('u8'),  # Simplified - real tuple would differ
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            bank_attr=BankAttribute(name='bank', bank_number=1),
            vreg_allocator=vreg_alloc
        )

        program = MIRProgram(functions=[func])
        codegen = ProgramCodeGenerator()
        asm_output = codegen.generate(program)

        func_start = asm_output.find('return_a_b:')
        assert func_start != -1, "return_a_b function not found"

        func_section = asm_output[func_start:func_start + 500]

        # vreg_a coalesces to A, so no frame needed at all
        has_phb = 'PHB' in func_section
        assert not has_phb, \
            f"Function with coalesceable vreg should not allocate frame\nOutput:\n{func_section}"

        # Should have RTL (far function)
        has_rtl = 'RTL' in func_section
        assert has_rtl, f"Far function should have RTL\nOutput:\n{func_section}"


class TestReturnSinkable:
    """Test return-sinkable vreg detection and optimization."""

    def _make_hw_symbol(self, name, address):
        """Create a minimal symbol object for hw memory locations."""
        class FakeSymbol:
            def __init__(self, name, addr):
                self.name = name
                self.address = addr
        return FakeSymbol(name, address)

    def test_load_return_sinkable(self):
        """Test that Load from MemoryLocation used only in Return is sinkable."""
        vreg_alloc = VirtualRegisterAllocator()
        vreg_low = vreg_alloc.alloc(BasicTypeInfo('u8'), "low")
        vreg_high = vreg_alloc.alloc(BasicTypeInfo('u8'), "high")

        sym_low = self._make_hw_symbol("RDMPYL", 0x4216)
        sym_high = self._make_hw_symbol("RDMPYH", 0x4217)

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            Load(
                dest=vreg_low,
                source=MemoryLocation(
                    storage_type='hw', address=0x4216,
                    symbol=sym_low, is_volatile=True
                ),
                type_info=BasicTypeInfo('u8')
            ),
            Load(
                dest=vreg_high,
                source=MemoryLocation(
                    storage_type='hw', address=0x4217,
                    symbol=sym_high, is_volatile=True
                ),
                type_info=BasicTypeInfo('u8')
            ),
            Return(values=[vreg_low, vreg_high])
        ]

        func = MIRFunction(
            name="mul8_return",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc
        )

        allocator = StackSlotAllocator(func)
        allocation = allocator.allocate()

        # Both vregs should be return-sinkable
        assert vreg_low in allocation.return_sinkable, \
            "Load from hw used only in Return should be return-sinkable"
        assert vreg_high in allocation.return_sinkable, \
            "Load from hw used only in Return should be return-sinkable"
        assert allocation.return_sinkable[vreg_low].address == 0x4216
        assert allocation.return_sinkable[vreg_high].address == 0x4217
        assert allocation.total_slots == 0, \
            "Return-sinkable vregs should not need stack slots"

    def test_load_with_non_return_use_not_sinkable(self):
        """Test that Load used in non-Return instruction is not sinkable."""
        vreg_alloc = VirtualRegisterAllocator()
        vreg_val = vreg_alloc.alloc(BasicTypeInfo('u8'), "val")
        vreg_result = vreg_alloc.alloc(BasicTypeInfo('u8'), "result")

        sym = self._make_hw_symbol("RDMPYL", 0x4216)

        entry_block = BasicBlock(block_id=0)
        entry_block.instructions = [
            Load(
                dest=vreg_val,
                source=MemoryLocation(
                    storage_type='hw', address=0x4216,
                    symbol=sym, is_volatile=True
                ),
                type_info=BasicTypeInfo('u8')
            ),
            # vreg_val used in BinaryOp, not just Return
            BinaryOp(
                dest=vreg_result,
                left=vreg_val,
                op='+',
                right=Immediate(1),
                type_info=BasicTypeInfo('u8')
            ),
            Return(values=[vreg_result])
        ]

        func = MIRFunction(
            name="not_sinkable",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc
        )

        allocator = StackSlotAllocator(func)
        allocation = allocator.allocate()

        assert vreg_val not in allocation.return_sinkable, \
            "Load used in non-Return instruction should NOT be return-sinkable"

    def test_load_from_move_not_sinkable(self):
        """Test that Move (not Load) is not detected as return-sinkable."""
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
            name="move_not_load",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={0: entry_block},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc
        )

        allocator = StackSlotAllocator(func)
        allocation = allocator.allocate()

        # Move from HardwareRegister should be hw-coalesceable, not return-sinkable
        assert vreg_a not in allocation.return_sinkable, \
            "Move instructions should not be return-sinkable (handled by hw-coalescence)"
        assert vreg_a in allocation.hw_coalesceable, \
            "Move from HW register should be hw-coalesceable"


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
