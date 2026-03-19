# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for loop unrolling optimization pass."""

import pytest

from r65.compiler.optimize.loop_unroll import (
    LoopUnroller, _detect_for_loop, _find_loops, _count_body_ops,
    _body_is_safe, MIN_BODY_OPS, MAX_BODY_OPS, MAX_UNROLLED_OPS,
)
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock,
    Move, BinaryOp, Compare, CondBranch, Jump, Return, Call,
    Store, Load, MemoryLocation, InlineAsm,
    VirtualRegister, HardwareRegister, Immediate,
    Argument, ArgumentMechanism,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


U8 = BasicTypeInfo('u8')
U16 = BasicTypeInfo('u16')


class MockSymbol:
    def __init__(self, name: str):
        self.name = name


def _make_func(blocks_dict, entry=0, exit_ids=None):
    """Build a MIRFunction from a dict of {block_id: BasicBlock}."""
    func = MIRFunction(
        name="test_func",
        parameters=[],
        return_type=U8,
        blocks=blocks_dict,
        entry_block_id=entry,
        exit_block_ids=exit_ids or [],
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )
    # Ensure vreg allocator counter is past existing vregs
    max_id = 0
    for block in blocks_dict.values():
        for instr in block.instructions:
            for attr in ('dest', 'source', 'left', 'right', 'operand',
                         'condition'):
                v = getattr(instr, attr, None)
                if isinstance(v, VirtualRegister) and v.id > max_id:
                    max_id = v.id
    func.vreg_allocator._next_id = max_id + 1
    return func


def _make_for_loop(trip_count, body_op_count, start=0):
    """
    Build a canonical for-loop MIR:

      Block 0 (pre-header): Move(i, start); Jump → Block 1
      Block 1 (header):     Compare(i, end, '<'); CondBranch(→2, →3)
      Block 2 (body):       [body_op_count Store instructions]
                             BinaryOp(i, i, '+', 1); Jump → Block 1
      Block 3 (exit):       Return
    """
    end = start + trip_count
    i_vreg = VirtualRegister(id=0, type_info=U16, hint="i")
    mem = MemoryLocation(storage_type='zeropage', address=0x10, symbol=MockSymbol("BUF"))

    # Body: body_op_count Store instructions
    body_instrs = []
    for n in range(body_op_count):
        tmp = VirtualRegister(id=10 + n, type_info=U8, hint=f"tmp{n}")
        body_instrs.append(
            Store(dest=mem, source=tmp, type_info=U8)
        )

    pre_header = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=i_vreg, source=Immediate(start), type_info=U16),
            Jump(target=1),
        ],
        predecessors=[],
        successors=[1],
    )
    header = BasicBlock(
        block_id=1,
        instructions=[
            Compare(left=i_vreg, right=Immediate(end),
                    comparison='<', type_info=U16),
            CondBranch(condition=i_vreg, true_target=2,
                       false_target=3, comparison='<'),
        ],
        predecessors=[0, 2],
        successors=[2, 3],
    )
    body = BasicBlock(
        block_id=2,
        instructions=body_instrs + [
            BinaryOp(dest=i_vreg, left=i_vreg, right=Immediate(1),
                     op='+', type_info=U16),
            Jump(target=1),
        ],
        predecessors=[1],
        successors=[1],
    )
    exit_block = BasicBlock(
        block_id=3,
        instructions=[Return(values=[])],
        predecessors=[1],
        successors=[],
    )
    return _make_func({0: pre_header, 1: header, 2: body, 3: exit_block},
                      entry=0, exit_ids=[3])


class TestLoopDetection:
    def test_find_loops_simple(self):
        func = _make_for_loop(trip_count=4, body_op_count=5)
        loops = _find_loops(func)
        assert len(loops) >= 1
        header_id, body_ids = loops[0]
        assert header_id == 1
        assert 2 in body_ids

    def test_detect_for_loop_pattern(self):
        func = _make_for_loop(trip_count=4, body_op_count=5)
        loops = _find_loops(func)
        header_id, body_ids = loops[0]
        result = _detect_for_loop(func, header_id, body_ids)
        assert result is not None
        counter_vreg, start, end, step, comp, body_bids, exit_id, inc_id = result
        assert start == 0
        assert end == 4
        assert step == 1
        assert exit_id == 3
        assert inc_id == 2


class TestUnrollEligibility:
    def test_body_too_small_rejected(self):
        """Loops with < 4 body ops are rejected."""
        func = _make_for_loop(trip_count=4, body_op_count=3)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 0

    def test_body_too_large_rejected(self):
        """Loops with > 20 body ops are rejected."""
        func = _make_for_loop(trip_count=2, body_op_count=21)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 0

    def test_total_size_exceeded_rejected(self):
        """Loops where trip_count * body_ops >= 255 are rejected."""
        # 20 ops * 13 iterations = 260 >= 255
        func = _make_for_loop(trip_count=13, body_op_count=20)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 0

    def test_eligible_loop_accepted(self):
        """A loop within all thresholds is unrolled."""
        # 5 ops * 4 iterations = 20 < 255
        func = _make_for_loop(trip_count=4, body_op_count=5)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 1

    def test_body_with_return_rejected(self):
        """Loops containing Return in body are rejected."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        # Insert a Return into the body block
        func.blocks[2].instructions.insert(0, Return(values=[]))
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 0

    def test_body_with_inline_asm_rejected(self):
        """Loops containing asm!() in body are rejected."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        func.blocks[2].instructions.insert(0, InlineAsm(instructions=["NOP"]))
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 0

    def test_body_with_call_rejected(self):
        """Loops containing function calls in body are rejected."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        func.blocks[2].instructions.insert(
            0, Call(function="some_func", args=[], returns=[]))
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 0


class TestUnrollCorrectness:
    def test_old_loop_blocks_removed(self):
        """After unrolling, the original loop header and body are removed."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)

        # Original header (1) and body (2) should be gone
        assert 1 not in func.blocks
        assert 2 not in func.blocks

    def test_exit_block_preserved(self):
        """The exit block remains after unrolling."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)
        assert 3 in func.blocks

    def test_correct_number_of_new_blocks(self):
        """Unrolling creates trip_count copies of body blocks."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        original_block_count = len(func.blocks)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)

        # Original: 4 blocks (pre-header, header, body, exit)
        # After: pre-header (0) + exit (3) + 4 unrolled body copies = 6
        # Header (1) and body (2) are removed
        new_blocks = {bid for bid in func.blocks.keys() if bid >= 4}
        assert len(new_blocks) == 4  # 4 unrolled copies

    def test_no_back_edges_in_unrolled_code(self):
        """Unrolled code should have no backward jumps to loop headers."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)

        # Original header (1) should be gone — no block should jump to it
        for bid, block in func.blocks.items():
            for instr in block.instructions:
                if isinstance(instr, Jump):
                    assert instr.target != 1, (
                        f"Block {bid} still jumps to old header")

    def test_counter_replaced_with_constants(self):
        """The loop counter should be replaced with Immediate values."""
        i_vreg = VirtualRegister(id=0, type_info=U16, hint="i")
        func = _make_for_loop(trip_count=3, body_op_count=5)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)

        # No instruction in unrolled blocks should reference vreg 0 (the counter)
        for bid, block in func.blocks.items():
            if bid <= 3:
                continue  # Skip pre-header and exit
            for instr in block.instructions:
                for attr in ('source', 'left', 'right', 'operand'):
                    val = getattr(instr, attr, None)
                    if isinstance(val, VirtualRegister):
                        assert val.id != 0, (
                            f"Block {bid} still references counter vreg: {instr}")

    def test_last_unrolled_block_reaches_exit(self):
        """The last unrolled block should jump to the exit block."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)

        # Find the highest-numbered block (last unrolled copy)
        max_bid = max(bid for bid in func.blocks.keys() if bid >= 4)
        last_block = func.blocks[max_bid]
        assert 3 in last_block.successors

    def test_pre_header_reaches_first_unrolled(self):
        """The pre-header should jump to the first unrolled block."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)

        pre_header = func.blocks[0]
        # Should have a Jump to block 4 (first unrolled)
        last_instr = pre_header.instructions[-1]
        assert isinstance(last_instr, Jump)
        assert last_instr.target == 4

    def test_no_counter_increment_in_unrolled(self):
        """Counter increment instructions should be stripped from unrolled body."""
        func = _make_for_loop(trip_count=3, body_op_count=5)
        unroller = LoopUnroller()
        unroller._unroll_in_function(func)

        for bid, block in func.blocks.items():
            if bid <= 3:
                continue
            for instr in block.instructions:
                if isinstance(instr, BinaryOp):
                    # No BinaryOp should be i = i + 1
                    assert not (isinstance(instr.dest, VirtualRegister) and
                                instr.dest.id == 0 and instr.op == '+'), \
                        f"Counter increment found in block {bid}"


class TestUnrollProgram:
    def test_unroll_via_program(self):
        """Test the top-level unroll() method on MIRProgram."""
        func = _make_for_loop(trip_count=4, body_op_count=5)
        program = MIRProgram(functions=[func])
        unroller = LoopUnroller()
        count = unroller.unroll(program)
        assert count == 1

    def test_no_eligible_loops(self):
        """Program with no loops returns 0."""
        func = _make_func({
            0: BasicBlock(block_id=0,
                          instructions=[Return(values=[])],
                          predecessors=[], successors=[]),
        }, entry=0, exit_ids=[0])
        program = MIRProgram(functions=[func])
        unroller = LoopUnroller()
        count = unroller.unroll(program)
        assert count == 0


class TestEdgeCases:
    def test_nonzero_start(self):
        """for i in 3..7 should unroll with constants 3, 4, 5, 6."""
        func = _make_for_loop(trip_count=4, body_op_count=5, start=3)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 1

    def test_boundary_4_ops(self):
        """Exactly MIN_BODY_OPS (4) should be accepted."""
        func = _make_for_loop(trip_count=4, body_op_count=4)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 1

    def test_boundary_20_ops(self):
        """Exactly MAX_BODY_OPS (20) should be accepted."""
        # 20 ops * 12 = 240 < 255
        func = _make_for_loop(trip_count=12, body_op_count=20)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 1

    def test_boundary_254_total(self):
        """trip_count * body_ops = 254 should be accepted (< 255)."""
        # 127 * 2 = 254 ... but 2 < MIN_BODY_OPS. Use 127 trips * doesn't work
        # Use: 6 ops * 42 = 252 < 255
        func = _make_for_loop(trip_count=42, body_op_count=6)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 1

    def test_boundary_255_total_rejected(self):
        """trip_count * body_ops = 255 should be rejected (not < 255)."""
        # 5 * 51 = 255, not < 255
        func = _make_for_loop(trip_count=51, body_op_count=5)
        unroller = LoopUnroller()
        count = unroller._unroll_in_function(func)
        assert count == 0
