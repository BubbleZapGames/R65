"""
Tests for hardware register liveness analysis.

Tests the tracking of direct X/Y hardware register usage across instructions
and basic blocks.
"""

import pytest
from r65.compiler.mir.nodes import (
    MIRFunction, BasicBlock, VirtualRegister, HardwareRegister, Immediate,
    Move, BinaryOp, Call, Return, Jump, CondBranch, Argument, ArgumentMechanism
)
from r65.compiler.mir.liveness import LivenessAnalyzer, InstructionLivenessAnalyzer
from r65.compiler.hir.types import BasicTypeInfo


def make_vreg(id: int, type_name: str = 'u8') -> VirtualRegister:
    """Helper to create a virtual register."""
    return VirtualRegister(id=id, type_info=BasicTypeInfo(type_name))


def make_hw_reg(name: str) -> HardwareRegister:
    """Helper to create a hardware register."""
    return HardwareRegister(name=name)


class TestHardwareRegisterLiveness:
    """Tests for hardware register liveness tracking in LivenessAnalyzer."""

    def test_hw_reg_use_def_tracking(self):
        """Test that X/Y uses and defs are tracked correctly."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block with X = 0; Y = X + 1
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            BinaryOp(dest=make_hw_reg('Y'), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        analyzer = LivenessAnalyzer(mir_func)
        liveness = analyzer.analyze()

        info = liveness[0]
        # X is defined, then used
        assert 'X' in info.hw_define
        # Y is defined
        assert 'Y' in info.hw_define
        # X is used (in BinaryOp) but defined first, so not in hw_use
        assert 'X' not in info.hw_use

    def test_hw_reg_use_before_def(self):
        """Test that hw_use tracks registers used before being defined in block."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block that uses X before defining it (X comes from outside)
        block = BasicBlock(block_id=0, instructions=[
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        analyzer = LivenessAnalyzer(mir_func)
        liveness = analyzer.analyze()

        info = liveness[0]
        # X is used before being defined in this block
        assert 'X' in info.hw_use

    def test_hw_reg_live_in_propagation(self):
        """Test that hw_live_in propagates through the CFG."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block 0: X = 0; jump to block 1
        block0 = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            Jump(target=1)
        ])

        # Block 1: use X; return
        block1 = BasicBlock(block_id=1, instructions=[
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])

        mir_func.blocks[0] = block0
        mir_func.blocks[1] = block1

        analyzer = LivenessAnalyzer(mir_func)
        liveness = analyzer.analyze()

        # Block 1 uses X, so X should be in hw_use and hw_live_in
        assert 'X' in liveness[1].hw_use
        assert 'X' in liveness[1].hw_live_in

        # Block 0's hw_live_out should include X (from block 1's hw_live_in)
        assert 'X' in liveness[0].hw_live_out

    def test_a_register_not_tracked(self):
        """Test that A register is NOT tracked (only X/Y are tracked)."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block with A = 0; use A
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('A'), source=Immediate(0), type_info=BasicTypeInfo('u8')),
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('A'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u8')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        analyzer = LivenessAnalyzer(mir_func)
        liveness = analyzer.analyze()

        info = liveness[0]
        # A should NOT be tracked
        assert 'A' not in info.hw_define
        assert 'A' not in info.hw_use


class TestInstructionLevelHwLiveness:
    """Tests for per-instruction hardware register liveness."""

    def test_is_hw_reg_live_after_simple(self):
        """Test is_hw_reg_live_after for simple case."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; call; use X
        call_instr = Call(function="clobber", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            call_instr,
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        analyzer = InstructionLivenessAnalyzer(mir_func)

        # X should be live after the call (instruction index 1)
        assert analyzer.is_hw_reg_live_after('X', 0, 1) is True

        # X should be live after the Move (instruction index 0)
        assert analyzer.is_hw_reg_live_after('X', 0, 0) is True

        # X should NOT be live after the BinaryOp (instruction index 2) - last use
        assert analyzer.is_hw_reg_live_after('X', 0, 2) is False

    def test_is_hw_reg_live_after_not_used(self):
        """Test is_hw_reg_live_after when register is not used after."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; call; (X not used after)
        call_instr = Call(function="clobber", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            call_instr,
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        analyzer = InstructionLivenessAnalyzer(mir_func)

        # X should NOT be live after the call - it's not used afterward
        assert analyzer.is_hw_reg_live_after('X', 0, 1) is False

    def test_is_hw_reg_live_after_y_register(self):
        """Test is_hw_reg_live_after for Y register."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block: Y = 0; call; use Y
        call_instr = Call(function="clobber", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('Y'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            call_instr,
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('Y'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        analyzer = InstructionLivenessAnalyzer(mir_func)

        # Y should be live after the call
        assert analyzer.is_hw_reg_live_after('Y', 0, 1) is True

    def test_is_hw_reg_live_after_a_returns_false(self):
        """Test that is_hw_reg_live_after returns False for A register."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block: A = 0; call; use A (but A is not tracked)
        call_instr = Call(function="clobber", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('A'), source=Immediate(0), type_info=BasicTypeInfo('u8')),
            call_instr,
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('A'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u8')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        analyzer = InstructionLivenessAnalyzer(mir_func)

        # A is not tracked, so should always return False
        assert analyzer.is_hw_reg_live_after('A', 0, 1) is False

    def test_hw_liveness_across_blocks(self):
        """Test hardware register liveness across multiple blocks."""
        mir_func = MIRFunction(name="test", blocks={})

        # Block 0: X = 0; jump to block 1
        block0 = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            Jump(target=1)
        ])

        # Block 1: call; jump to block 2
        call_instr = Call(function="clobber", args=[], returns=[])
        block1 = BasicBlock(block_id=1, instructions=[
            call_instr,
            Jump(target=2)
        ])

        # Block 2: use X; return
        block2 = BasicBlock(block_id=2, instructions=[
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])

        mir_func.blocks[0] = block0
        mir_func.blocks[1] = block1
        mir_func.blocks[2] = block2

        analyzer = InstructionLivenessAnalyzer(mir_func)

        # X should be live after the call in block 1
        # because it's used in block 2
        assert analyzer.is_hw_reg_live_after('X', 1, 0) is True
