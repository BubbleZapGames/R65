"""Tests for ABI, stack frame layout, and stack state tracker abstractions."""

import pytest
from r65.compiler.codegen.abi import ABIInfo, StackFrameLayout, StackStateTracker
from r65.compiler.typeck.processor_mode import ModeState
from r65.compiler.hir.attributes import DataBankMode


# ========================================================================
# ABIInfo Tests
# ========================================================================

class TestABIInfo:
    def test_return_addr_size_near(self):
        abi = ABIInfo(is_far=False)
        assert abi.return_addr_size == 2

    def test_return_addr_size_far(self):
        abi = ABIInfo(is_far=True)
        assert abi.return_addr_size == 3

    def test_accu_size_m8(self):
        abi = ABIInfo(entry_m_mode=ModeState.M8)
        assert abi.accu_size == 1

    def test_accu_size_m16(self):
        abi = ABIInfo(entry_m_mode=ModeState.M16)
        assert abi.accu_size == 2

    def test_accu_size_default_none(self):
        abi = ABIInfo()
        assert abi.accu_size == 1  # Default: m8

    def test_push_size_a_m8(self):
        abi = ABIInfo(entry_m_mode=ModeState.M8)
        assert abi.push_size('A') == 1

    def test_push_size_a_m16(self):
        abi = ABIInfo(entry_m_mode=ModeState.M16)
        assert abi.push_size('A') == 2

    def test_push_size_x(self):
        abi = ABIInfo()
        assert abi.push_size('X') == 2

    def test_push_size_y(self):
        abi = ABIInfo()
        assert abi.push_size('Y') == 2

    def test_push_size_d(self):
        abi = ABIInfo()
        assert abi.push_size('D') == 2

    def test_push_size_status(self):
        abi = ABIInfo()
        assert abi.push_size('STATUS') == 1

    def test_push_size_dbr(self):
        abi = ABIInfo()
        assert abi.push_size('DBR') == 1

    def test_push_size_b(self):
        abi = ABIInfo()
        assert abi.push_size('B') == 1

    def test_push_size_unknown(self):
        abi = ABIInfo()
        with pytest.raises(ValueError):
            abi.push_size('INVALID')

    def test_prologue_stack_bytes_no_preserves(self):
        abi = ABIInfo()
        assert abi.prologue_stack_bytes == 0

    def test_prologue_stack_bytes_preserves_xy(self):
        abi = ABIInfo(preserves=('X', 'Y'))
        assert abi.prologue_stack_bytes == 4  # 2 + 2

    def test_prologue_stack_bytes_preserves_a_m8(self):
        abi = ABIInfo(preserves=('A',), entry_m_mode=ModeState.M8)
        assert abi.prologue_stack_bytes == 1

    def test_prologue_stack_bytes_preserves_a_m16(self):
        abi = ABIInfo(preserves=('A',), entry_m_mode=ModeState.M16)
        assert abi.prologue_stack_bytes == 2

    def test_prologue_stack_bytes_preserves_status(self):
        abi = ABIInfo(preserves=('STATUS',))
        assert abi.prologue_stack_bytes == 1

    def test_prologue_stack_bytes_far_inline_dbr(self):
        abi = ABIInfo(is_far=True, databank_mode=DataBankMode.INLINE)
        assert abi.prologue_stack_bytes == 1  # PHB

    def test_prologue_stack_bytes_far_no_dbr(self):
        abi = ABIInfo(is_far=True, databank_mode=DataBankMode.NONE)
        assert abi.prologue_stack_bytes == 0

    def test_prologue_stack_bytes_far_caller_dbr(self):
        abi = ABIInfo(is_far=True, databank_mode=DataBankMode.CALLER)
        assert abi.prologue_stack_bytes == 0

    def test_prologue_stack_bytes_near_inline_dbr(self):
        # Near functions with INLINE databank mode don't push PHB
        abi = ABIInfo(is_far=False, databank_mode=DataBankMode.INLINE)
        assert abi.prologue_stack_bytes == 0

    def test_prologue_stack_bytes_far_ptr_params(self):
        abi = ABIInfo(has_far_ptr_stack_params=True)
        assert abi.prologue_stack_bytes == 2  # PHD

    def test_prologue_stack_bytes_combined(self):
        abi = ABIInfo(
            is_far=True,
            databank_mode=DataBankMode.INLINE,
            preserves=('X', 'Y', 'STATUS'),
            has_far_ptr_stack_params=True,
            entry_m_mode=ModeState.M8,
        )
        # PHB(1) + PHX(2) + PHY(2) + PHP(1) + PHD(2) = 8
        assert abi.prologue_stack_bytes == 8


# ========================================================================
# StackFrameLayout Tests
# ========================================================================

class TestStackFrameLayout:
    def test_total_frame_size(self):
        abi = ABIInfo()
        layout = StackFrameLayout(abi=abi, local_frame_size=10, outgoing_arg_bytes=4)
        assert layout.total_frame_size == 14

    def test_total_frame_size_zero(self):
        abi = ABIInfo()
        layout = StackFrameLayout(abi=abi)
        assert layout.total_frame_size == 0

    def test_stack_base_offset_entry(self):
        abi = ABIInfo(is_entry=True)
        layout = StackFrameLayout(abi=abi)
        assert layout.stack_base_offset == 1

    def test_stack_base_offset_regular(self):
        abi = ABIInfo(preserves=('X', 'Y'))  # 4 prologue bytes
        layout = StackFrameLayout(abi=abi)
        assert layout.stack_base_offset == 5  # 4 + 1

    def test_stack_base_offset_no_preserves(self):
        abi = ABIInfo()
        layout = StackFrameLayout(abi=abi)
        assert layout.stack_base_offset == 1  # 0 + 1

    def test_local_offset(self):
        abi = ABIInfo(preserves=('X',))  # 2 prologue bytes
        layout = StackFrameLayout(abi=abi, outgoing_arg_bytes=4)
        # stack_base_offset = 3, + outgoing=4, + slot_num
        assert layout.local_offset(0) == 7
        assert layout.local_offset(1) == 8
        assert layout.local_offset(5) == 12

    def test_param_offset(self):
        abi = ABIInfo(preserves=('X',))  # 2 prologue bytes
        layout = StackFrameLayout(abi=abi, local_frame_size=6, outgoing_arg_bytes=2)
        # param_offset = base_offset + prologue(2) + total_frame(8) = base + 10
        assert layout.param_offset(3) == 13
        assert layout.param_offset(5) == 15

    def test_outgoing_arg_offset(self):
        abi = ABIInfo()
        layout = StackFrameLayout(abi=abi)
        assert layout.outgoing_arg_offset(0) == 1
        assert layout.outgoing_arg_offset(2) == 3

    def test_has_frame_true(self):
        abi = ABIInfo()
        layout = StackFrameLayout(abi=abi, local_frame_size=4)
        assert layout.has_frame is True

    def test_has_frame_false(self):
        abi = ABIInfo()
        layout = StackFrameLayout(abi=abi)
        assert layout.has_frame is False

    def test_has_frame_outgoing_only(self):
        abi = ABIInfo()
        layout = StackFrameLayout(abi=abi, outgoing_arg_bytes=2)
        assert layout.has_frame is True


# ========================================================================
# StackStateTracker Tests
# ========================================================================

class TestStackStateTracker:
    def test_initial_displacement(self):
        tracker = StackStateTracker()
        assert tracker.displacement == 0

    def test_push(self):
        tracker = StackStateTracker()
        tracker.push(2)
        assert tracker.displacement == 2

    def test_multiple_pushes(self):
        tracker = StackStateTracker()
        tracker.push(2)
        tracker.push(1)
        assert tracker.displacement == 3

    def test_pop(self):
        tracker = StackStateTracker()
        tracker.push(4)
        tracker.pop(2)
        assert tracker.displacement == 2

    def test_push_pop_balanced(self):
        tracker = StackStateTracker()
        tracker.push(2)
        tracker.push(2)
        tracker.pop(2)
        tracker.pop(2)
        assert tracker.displacement == 0

    def test_pop_allows_negative_displacement(self):
        """Negative displacement is valid for PLD/PHD dance in far-ptr functions."""
        tracker = StackStateTracker()
        tracker.pop(2)  # Simulates PLD popping saved D
        assert tracker.displacement == -2
        tracker.push(3)  # Simulates 3 PHA bytes for far ptr arg
        assert tracker.displacement == 1

    def test_adjust_offset(self):
        tracker = StackStateTracker()
        tracker.push(3)
        assert tracker.adjust_offset(5) == 8

    def test_adjust_offset_zero_displacement(self):
        tracker = StackStateTracker()
        assert tracker.adjust_offset(5) == 5

    def test_reset(self):
        tracker = StackStateTracker()
        tracker.push(5)
        tracker.reset()
        assert tracker.displacement == 0

    def test_typical_spill_sequence(self):
        """Simulate a typical X/Y spill + reload sequence."""
        tracker = StackStateTracker()
        # Spill X (2 bytes)
        tracker.push(2)
        assert tracker.displacement == 2
        # Spill Y (2 bytes)
        tracker.push(2)
        assert tracker.displacement == 4
        # Adjust a stack local at offset 3
        assert tracker.adjust_offset(3) == 7
        # Reload Y
        tracker.pop(2)
        assert tracker.displacement == 2
        # Reload X
        tracker.pop(2)
        assert tracker.displacement == 0
