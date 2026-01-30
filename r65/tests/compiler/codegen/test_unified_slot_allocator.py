"""
Tests for unified slot allocator.
"""

import pytest
from r65.compiler.codegen.unified_slot_allocator import (
    UnifiedSlotAllocator, PreassignedSlot, UnifiedSlotAllocation
)
from r65.compiler.mir.nodes import (
    MIRFunction, BasicBlock, VirtualRegister, Move, Return,
    HardwareRegister, BinaryOp
)
from r65.compiler.hir.types import BasicTypeInfo


U8 = BasicTypeInfo('u8')


def make_vreg(id: int, type_info=None) -> VirtualRegister:
    """Create a virtual register for testing."""
    return VirtualRegister(id=id, type_info=type_info or U8)


def test_unified_allocator_basic():
    """Test basic allocation without params."""
    # Create a simple function with two locals
    mir_func = MIRFunction(name="test", is_entry=False)
    block = BasicBlock(block_id=0)
    mir_func.blocks[0] = block
    mir_func.entry_block_id = 0

    vreg1 = make_vreg(1)
    vreg2 = make_vreg(2)
    vreg3 = make_vreg(3)

    # v1 = 10
    # v2 = 20
    # return v1 + v2
    block.instructions = [
        Move(dest=vreg1, source=10, type_info=U8),
        Move(dest=vreg2, source=20, type_info=U8),
        BinaryOp(dest=vreg3, left=vreg1, right=vreg2, op='+', type_info=U8),
        Return(values=[vreg3])
    ]

    allocator = UnifiedSlotAllocator(mir_func, preassigned=[], prologue_stack_bytes=2)
    result = allocator.allocate()

    # Should have allocated slots for locals
    assert result.frame_size >= 0
    assert len(result.param_offsets) == 0


def test_unified_allocator_with_params():
    """Test allocation with preassigned stack params."""
    mir_func = MIRFunction(name="test", is_entry=False)
    block = BasicBlock(block_id=0)
    mir_func.blocks[0] = block
    mir_func.entry_block_id = 0

    # Simulate: fn test(a: u8, b: u8) { let x = a + b; return x; }
    param_a = make_vreg(1)
    param_b = make_vreg(2)
    local_x = make_vreg(3)
    result_vreg = make_vreg(4)

    block.instructions = [
        # x = a + b
        BinaryOp(dest=local_x, left=param_a, right=param_b, op='+', type_info=U8),
        # return x
        Move(dest=result_vreg, source=local_x, type_info=U8),
        Return(values=[result_vreg])
    ]

    # Params at offsets 3 and 4 (after return addr at 1-2)
    preassigned = [
        PreassignedSlot(vreg=param_a, base_offset=3, size=1),
        PreassignedSlot(vreg=param_b, base_offset=4, size=1),
    ]

    prologue_bytes = 2  # return address
    allocator = UnifiedSlotAllocator(mir_func, preassigned=preassigned, prologue_stack_bytes=prologue_bytes)
    result = allocator.allocate()

    # Should have param offsets (adjusted for prologue + frame)
    assert param_a in result.param_offsets
    assert param_b in result.param_offsets

    # Frame size should only count local_x (and result_vreg if not coalesced)
    # Params don't count toward frame_size
    assert result.frame_size >= 0

    # Param offsets should be: base_offset + prologue_bytes + frame_size
    expected_a_offset = 3 + prologue_bytes + result.frame_size
    expected_b_offset = 4 + prologue_bytes + result.frame_size
    assert result.param_offsets[param_a] == expected_a_offset
    assert result.param_offsets[param_b] == expected_b_offset


def test_unified_allocator_slot_reuse():
    """Test that locals with non-overlapping lifetimes reuse slots."""
    mir_func = MIRFunction(name="test", is_entry=False)
    block = BasicBlock(block_id=0)
    mir_func.blocks[0] = block
    mir_func.entry_block_id = 0

    # Simulate: { let a = 1; } { let b = 2; } return b;
    # a and b don't overlap, so they can share a slot
    vreg_a = make_vreg(1)
    vreg_b = make_vreg(2)

    block.instructions = [
        Move(dest=vreg_a, source=1, type_info=U8),
        # a is dead after this point
        Move(dest=vreg_b, source=2, type_info=U8),
        Return(values=[vreg_b])
    ]

    allocator = UnifiedSlotAllocator(mir_func, preassigned=[], prologue_stack_bytes=2)
    result = allocator.allocate()

    # With reuse, both should fit in 1 slot (if liveness analysis works)
    # At minimum, we shouldn't use more than 2 slots
    assert result.frame_size <= 2

    # Check that some reuse happened (or at least it didn't explode)
    if result.local_count > 0:
        assert result.frame_size <= result.local_count


def test_param_offset_calculation():
    """Test that param offsets are calculated correctly."""
    mir_func = MIRFunction(name="test", is_entry=False)
    block = BasicBlock(block_id=0)
    mir_func.blocks[0] = block
    mir_func.entry_block_id = 0

    param_a = make_vreg(1)
    local_x = make_vreg(2)

    block.instructions = [
        Move(dest=local_x, source=param_a, type_info=U8),
        Return(values=[local_x])
    ]

    preassigned = [
        PreassignedSlot(vreg=param_a, base_offset=3, size=1),
    ]

    # Test with different prologue sizes
    for prologue_bytes in [0, 2, 4]:
        allocator = UnifiedSlotAllocator(
            mir_func,
            preassigned=preassigned,
            prologue_stack_bytes=prologue_bytes
        )
        result = allocator.allocate()

        # param_offset = base_offset + prologue + frame_size
        expected = 3 + prologue_bytes + result.frame_size
        assert result.param_offsets[param_a] == expected, \
            f"With prologue={prologue_bytes}, frame={result.frame_size}: " \
            f"expected {expected}, got {result.param_offsets[param_a]}"


def test_get_offset_unified():
    """Test the unified get_offset method works for both params and locals."""
    mir_func = MIRFunction(name="test", is_entry=False)
    block = BasicBlock(block_id=0)
    mir_func.blocks[0] = block
    mir_func.entry_block_id = 0

    param_a = make_vreg(1)
    local_x = make_vreg(2)

    block.instructions = [
        Move(dest=local_x, source=param_a, type_info=U8),
        Return(values=[local_x])
    ]

    preassigned = [
        PreassignedSlot(vreg=param_a, base_offset=3, size=1),
    ]

    allocator = UnifiedSlotAllocator(mir_func, preassigned=preassigned, prologue_stack_bytes=2)
    result = allocator.allocate()

    # Should be able to look up both via get_offset
    param_offset = result.get_offset(param_a)
    assert param_offset is not None
    assert result.is_param(param_a)

    # local_x might be hw-coalesced or allocated
    local_offset = result.get_offset(local_x)
    # It's ok if local_x is hw-coalesced and returns None
    if local_x not in result.hw_coalesceable:
        assert local_offset is not None
        assert not result.is_param(local_x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
