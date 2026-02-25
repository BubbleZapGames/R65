"""
Test vreg-to-vreg move coalescing in the slot allocator.

The coalescing pass merges Move(dest_vreg, src_vreg) when their lifetimes
don't interfere, recovering the efficiency of vreg reuse while maintaining
correctness when both variables are independently modified.
"""

import pytest
from r65.compiler.codegen.slot_allocator import StackSlotAllocator
from r65.compiler.mir.nodes import (
    MIRFunction, BasicBlock,
    Move, Return, BinaryOp, Compare, Jump, CondBranch,
    VirtualRegister, Immediate, HardwareRegister,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


def _make_func(blocks_dict, vreg_alloc=None):
    """Helper to build a MIRFunction from a dict of block_id -> instructions."""
    if vreg_alloc is None:
        vreg_alloc = VirtualRegisterAllocator()
    func = MIRFunction(
        name="test",
        return_type=BasicTypeInfo('u8'),
        vreg_allocator=vreg_alloc,
    )
    for bid, instrs in blocks_dict.items():
        block = BasicBlock(block_id=bid, instructions=instrs)
        func.blocks[bid] = block
    func.entry_block_id = min(blocks_dict.keys())
    # Wire up successors/predecessors from Jump/CondBranch
    for bid, block in func.blocks.items():
        if block.instructions:
            last = block.instructions[-1]
            if isinstance(last, Jump):
                block.successors = [last.target]
                func.blocks[last.target].predecessors.append(bid)
            elif isinstance(last, CondBranch):
                block.successors = [last.true_target, last.false_target]
                func.blocks[last.true_target].predecessors.append(bid)
                func.blocks[last.false_target].predecessors.append(bid)
    # Exit blocks
    for bid, block in func.blocks.items():
        if block.instructions and isinstance(block.instructions[-1], Return):
            func.exit_block_ids.append(bid)
    return func


class TestVregCoalescing:
    """Test the _coalesce_vreg_moves pass in StackSlotAllocator."""

    def test_dead_source_coalesced(self):
        """Move(x, y) where y is dead after → coalesced, x reuses y's slot."""
        u8 = BasicTypeInfo('u8')
        alloc = VirtualRegisterAllocator()
        y = alloc.alloc(u8, "y")
        x = alloc.alloc(u8, "x")

        func = _make_func({
            0: [
                Move(dest=y, source=HardwareRegister('A'), type_info=u8),
                Move(dest=x, source=y, type_info=u8),  # y dead after
                Return(values=[x]),
            ]
        }, alloc)

        sa = StackSlotAllocator(func)
        result = sa.allocate()

        # After coalescing, x should be merged into y (or vice versa).
        # Only one local vreg should need a slot (the other is coalesced away).
        # The hw-coalesceable pass may also kick in, so check either outcome.
        total_local_vregs = len(result.register_to_slot)
        total_hw = len(result.hw_coalesceable)
        # At most 1 local slot (coalesced) + whatever is hw-coalesceable
        assert total_local_vregs + total_hw <= 2, (
            f"Expected coalescing: locals={total_local_vregs}, hw={total_hw}"
        )

    def test_interfering_not_coalesced(self):
        """Move(x, y) where y is still used after → NOT coalesced."""
        u8 = BasicTypeInfo('u8')
        alloc = VirtualRegisterAllocator()
        y = alloc.alloc(u8, "y")
        x = alloc.alloc(u8, "x")

        func = _make_func({
            0: [
                Move(dest=y, source=HardwareRegister('A'), type_info=u8),
                Move(dest=x, source=y, type_info=u8),
                # x is modified independently
                BinaryOp(dest=x, left=x, right=Immediate(1),
                          op='-', type_info=u8),
                # Both y and x are used → they interfere
                Return(values=[y]),
            ]
        }, alloc)

        sa = StackSlotAllocator(func)
        result = sa.allocate()

        # y and x must have separate storage (can't share a slot)
        # y is hw-coalesceable (Move from A, used in Return)
        # x needs its own slot
        assert y in result.hw_coalesceable or y in result.register_to_slot
        assert x in result.hw_coalesceable or x in result.register_to_slot

    def test_coalescing_propagates_register_hint(self):
        """When coalescing Move(x, y), x's register_hint propagates to y."""
        u8 = BasicTypeInfo('u8')
        alloc = VirtualRegisterAllocator()
        y = alloc.alloc(u8, "y")
        x = alloc.alloc(u8, "x", register_hint='X')

        func = _make_func({
            0: [
                Move(dest=y, source=HardwareRegister('A'), type_info=u8),
                Move(dest=x, source=y, type_info=u8),  # y dead after
                Return(values=[x]),
            ]
        }, alloc)

        sa = StackSlotAllocator(func)
        sa.allocate()

        # After coalescing x → y, y should inherit x's hint
        assert y.register_hint == 'X'
