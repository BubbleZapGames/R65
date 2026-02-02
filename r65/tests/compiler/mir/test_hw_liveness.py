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


class TestClobberRegionAnalyzer:
    """Tests for ClobberRegionAnalyzer - region-based spilling optimization."""

    def test_single_call_creates_region(self):
        """Test that a single clobbering call creates a region."""
        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; call; use X; return
        call_instr = Call(function="clobber", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            call_instr,
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        instr_liveness = InstructionLivenessAnalyzer(mir_func)
        analyzer = ClobberRegionAnalyzer(instr_liveness)
        regions = analyzer.analyze_block(0)

        # Should have one region for X
        assert len(regions['X']) == 1
        region = regions['X'][0]
        assert region.hw_reg == 'X'
        assert region.save_before_idx == 1  # Before the call
        assert region.restore_before_idx == 2  # Before the use
        assert region.clobbering_calls == [1]  # The call at index 1

    def test_multi_call_merged_region(self):
        """Test that consecutive calls without intervening use create one region."""
        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; call1; call2; use X; return
        call1 = Call(function="clobber1", args=[], returns=[])
        call2 = Call(function="clobber2", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            call1,  # idx 1
            call2,  # idx 2
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),  # idx 3
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        instr_liveness = InstructionLivenessAnalyzer(mir_func)
        analyzer = ClobberRegionAnalyzer(instr_liveness)
        regions = analyzer.analyze_block(0)

        # Should have ONE region for X (merged)
        assert len(regions['X']) == 1
        region = regions['X'][0]
        assert region.save_before_idx == 1  # Before first call
        assert region.restore_before_idx == 3  # Before use
        assert region.clobbering_calls == [1, 2]  # Both calls

    def test_intervening_use_splits_regions(self):
        """Test that a use between calls creates separate regions."""
        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; call1; use X; call2; use X; return
        call1 = Call(function="clobber1", args=[], returns=[])
        call2 = Call(function="clobber2", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),  # idx 0
            call1,  # idx 1
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),  # idx 2 - use X
            call2,  # idx 3
            BinaryOp(dest=make_vreg(1), left=make_hw_reg('X'), right=Immediate(2),
                     op='+', type_info=BasicTypeInfo('u16')),  # idx 4 - use X again
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        instr_liveness = InstructionLivenessAnalyzer(mir_func)
        analyzer = ClobberRegionAnalyzer(instr_liveness)
        regions = analyzer.analyze_block(0)

        # Should have TWO regions for X (split by intervening use)
        assert len(regions['X']) == 2

        region1 = regions['X'][0]
        assert region1.save_before_idx == 1
        assert region1.restore_before_idx == 2
        assert region1.clobbering_calls == [1]

        region2 = regions['X'][1]
        assert region2.save_before_idx == 3
        assert region2.restore_before_idx == 4
        assert region2.clobbering_calls == [3]

    def test_mixed_x_y_regions(self):
        """Test that X and Y can have independent regions."""
        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; Y = 1; call_both; call_y_only; use X; use Y; return
        call_both = Call(function="clobber_both", args=[], returns=[])
        call_y = Call(function="clobber_y", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),  # idx 0
            Move(dest=make_hw_reg('Y'), source=Immediate(1), type_info=BasicTypeInfo('u16')),  # idx 1
            call_both,  # idx 2 - clobbers both
            call_y,     # idx 3 - clobbers Y only (preserves X)
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),  # idx 4 - use X
            BinaryOp(dest=make_vreg(1), left=make_hw_reg('Y'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),  # idx 5 - use Y
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        instr_liveness = InstructionLivenessAnalyzer(mir_func)
        analyzer = ClobberRegionAnalyzer(instr_liveness)

        # Provide preserves map: call_y preserves X
        preserves_map = {'clobber_y': {'X'}}
        regions = analyzer.analyze_block(0, preserves_map)

        # X region: only call_both clobbers X (call_y preserves it)
        assert len(regions['X']) == 1
        x_region = regions['X'][0]
        assert x_region.clobbering_calls == [2]  # Only call_both
        assert x_region.restore_before_idx == 4  # Before use X

        # Y region: both calls clobber Y
        assert len(regions['Y']) == 1
        y_region = regions['Y'][0]
        assert y_region.clobbering_calls == [2, 3]  # Both calls
        assert y_region.restore_before_idx == 5  # Before use Y

    def test_no_region_when_not_live_after(self):
        """Test that no region is created if register isn't used after call."""
        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; call; return (X not used after call)
        call_instr = Call(function="clobber", args=[], returns=[])
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            call_instr,
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        instr_liveness = InstructionLivenessAnalyzer(mir_func)
        analyzer = ClobberRegionAnalyzer(instr_liveness)
        regions = analyzer.analyze_block(0)

        # Should have no regions for X (not live after call)
        assert len(regions['X']) == 0

    def test_no_region_when_callee_preserves(self):
        """Test that no region is created if callee preserves the register."""
        from r65.compiler.mir.liveness import ClobberRegionAnalyzer
        from r65.compiler.hir.attributes import PreservesAttribute

        mir_func = MIRFunction(name="test", blocks={})

        # Block: X = 0; call (preserves X); use X; return
        call_instr = Call(function="safe", args=[], returns=[],
                         preserves_attr=PreservesAttribute(name='preserves', registers=['X']))
        block = BasicBlock(block_id=0, instructions=[
            Move(dest=make_hw_reg('X'), source=Immediate(0), type_info=BasicTypeInfo('u16')),
            call_instr,
            BinaryOp(dest=make_vreg(0), left=make_hw_reg('X'), right=Immediate(1),
                     op='+', type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ])
        mir_func.blocks[0] = block

        instr_liveness = InstructionLivenessAnalyzer(mir_func)
        analyzer = ClobberRegionAnalyzer(instr_liveness)
        regions = analyzer.analyze_block(0)

        # Should have no regions for X (callee preserves it via preserves_attr)
        assert len(regions['X']) == 0
