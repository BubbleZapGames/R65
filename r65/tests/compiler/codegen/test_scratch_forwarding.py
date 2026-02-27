#!/usr/bin/env python3
"""Test scratch parameter forwarding identity skip optimization.

When caller and callee both use scratch param promotion for the same parameter
at the same DP address, the LDA/STA pair should be skipped.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind
from r65.compiler.codegen.call_select import CallInstructionSelector
from r65.compiler.mir.nodes import Argument, ArgumentMechanism, VirtualRegister, Immediate as MIRImmediate
from r65.compiler.hir.types import BasicTypeInfo, PointerTypeInfo


class TestScratchParamIdentitySkip:
    """Test that emit_scratch_param_argument skips identity copies."""

    def _make_selector(self):
        """Create a minimally mocked CallInstructionSelector."""
        parent = MagicMock()
        parent.emitter.get_accu_mode.return_value = 8
        selector = CallInstructionSelector.__new__(CallInstructionSelector)
        selector.parent = parent
        selector.region_state = MagicMock()
        selector._function_regions = None
        selector._a_bound_to_vreg = False
        return selector

    def _make_arg(self, scratch_addr, param_type=None):
        """Create a scratch param Argument."""
        vreg = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="param")
        arg = Argument(
            value=vreg,
            mechanism=ArgumentMechanism.SCRATCH_PARAM,
            scratch_addr=scratch_addr,
            param_type=param_type,
        )
        return arg

    def test_identity_skip_u8(self):
        """u8 scratch param at same address should skip LDA/STA."""
        selector = self._make_selector()
        arg = self._make_arg(scratch_addr=0x10, param_type=BasicTypeInfo('u8'))
        arg_loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x10, size=1)

        selector.emit_scratch_param_argument(arg, arg_loc)

        # Should not emit any instructions (identity skip)
        selector.emitter.emit_instr.assert_not_called()
        selector.parent._emit_load.assert_not_called()

    def test_identity_skip_u16(self):
        """u16 scratch param at same address should skip LDA/STA."""
        selector = self._make_selector()
        arg = self._make_arg(scratch_addr=0x10, param_type=BasicTypeInfo('u16'))
        arg_loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x10, size=2)

        selector.emit_scratch_param_argument(arg, arg_loc)

        # Should not emit any instructions
        selector.emitter.emit_instr.assert_not_called()
        selector.parent._emit_load.assert_not_called()

    def test_identity_skip_far_pointer(self):
        """3-byte far pointer scratch param at same address should skip."""
        selector = self._make_selector()
        far_ptr_type = PointerTypeInfo(is_far=True, pointee_type=BasicTypeInfo('u8'))
        arg = self._make_arg(scratch_addr=0x10, param_type=far_ptr_type)
        arg_loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x10, size=3)

        selector.emit_scratch_param_argument(arg, arg_loc)

        # Should not emit any instructions
        selector.emitter.emit_instr.assert_not_called()
        selector.parent._emit_load.assert_not_called()

    def test_different_address_not_skipped(self):
        """Different scratch addresses should NOT skip."""
        selector = self._make_selector()
        arg = self._make_arg(scratch_addr=0x12, param_type=BasicTypeInfo('u8'))
        arg_loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x10, size=1)

        selector.emit_scratch_param_argument(arg, arg_loc)

        # Should emit instructions (no identity skip)
        selector.emitter.emit_instr.assert_called()

    def test_non_scratch_location_not_skipped(self):
        """Stack-allocated source should NOT skip even if address matches."""
        selector = self._make_selector()
        arg = self._make_arg(scratch_addr=0x10, param_type=BasicTypeInfo('u8'))
        arg_loc = PhysicalLocation(kind=LocationKind.STACK, stack_offset=3, size=1)

        selector.emit_scratch_param_argument(arg, arg_loc)

        # Should emit instructions
        selector.emitter.emit_instr.assert_called()

    def test_size_mismatch_not_skipped(self):
        """u8 source for u16 param should NOT skip (high byte missing)."""
        selector = self._make_selector()
        arg = self._make_arg(scratch_addr=0x10, param_type=BasicTypeInfo('u16'))
        arg_loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x10, size=1)

        selector.emit_scratch_param_argument(arg, arg_loc)

        # Should emit instructions (size mismatch)
        selector.emitter.emit_instr.assert_called()
