"""Tests for ABIModel abstraction."""

import sys
import pytest
from r65.compiler.codegen.abi_model import (
    ABIModel, ABIKind, ABIDefault, ABIFixedStack,
    ABI_DEFAULT, ABI_FIXED_STACK, abi_model_from_string,
)


class TestABIKind:
    def test_default_value(self):
        assert ABIKind.DEFAULT.value == "Default"

    def test_fixed_stack_value(self):
        assert ABIKind.FIXED_STACK.value == "FixedStack"


class TestABIDefaultFrameAlloc:
    """ABIDefault.emit_frame_alloc delegates to PHB or TSC depending on size."""

    def test_small_frame_emits_phb(self):
        """frame_size=2 → 2 PHB opcodes."""
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_DEFAULT.emit_frame_alloc(fake_emit, frame_size=2, force_direct_stack=False)
        assert collected == ['PHB', 'PHB']

    def test_large_frame_emits_tsc(self):
        """frame_size=8 → TSC/SBC/TCS sequence."""
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_DEFAULT.emit_frame_alloc(fake_emit, frame_size=8, force_direct_stack=False)
        names = [n for n in collected]
        assert 'TSC' in names
        assert 'SBC_IMMEDIATE' in names
        assert 'TCS' in names
        assert 'PHB' not in names

    def test_zero_frame_noop(self):
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_DEFAULT.emit_frame_alloc(fake_emit, frame_size=0, force_direct_stack=False)
        assert collected == []


class TestABIFixedStackFrameAlloc:
    """ABIFixedStack.emit_frame_alloc always uses PHB (unless force_direct_stack)."""

    def test_large_frame_emits_phb(self):
        """frame_size=8 → 8 PHB opcodes (not TSC)."""
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_FIXED_STACK.emit_frame_alloc(fake_emit, frame_size=8, force_direct_stack=False)
        assert collected == ['PHB'] * 8

    def test_force_direct_stack_emits_tsc(self):
        """force_direct_stack=True → TSC even for FixedStack."""
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_FIXED_STACK.emit_frame_alloc(fake_emit, frame_size=8, force_direct_stack=True)
        names = [n for n in collected]
        assert 'TSC' in names
        assert 'PHB' not in names


class TestMaxPlaDealloc:
    def test_default_is_4(self):
        assert ABI_DEFAULT.max_pla_dealloc_size == 4

    def test_fixed_stack_is_large(self):
        assert ABI_FIXED_STACK.max_pla_dealloc_size > 1000


class TestComputeOutgoingArgs:
    def test_default_calls_compute_fn(self):
        called = []
        def compute_fn(prog):
            called.append(prog)
        ABI_DEFAULT.compute_outgoing_args("mir_prog", compute_fn)
        assert called == ["mir_prog"]

    def test_fixed_stack_skips_compute_fn(self):
        called = []
        def compute_fn(prog):
            called.append(prog)
        ABI_FIXED_STACK.compute_outgoing_args("mir_prog", compute_fn)
        assert called == []


class TestAbiModelFromString:
    def test_default(self):
        model = abi_model_from_string("Default")
        assert model.kind == ABIKind.DEFAULT

    def test_fixed_stack(self):
        model = abi_model_from_string("FixedStack")
        assert model.kind == ABIKind.FIXED_STACK

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown ABI model"):
            abi_model_from_string("invalid")

    def test_singleton_default(self):
        assert abi_model_from_string("Default") is ABI_DEFAULT

    def test_singleton_fixed_stack(self):
        assert abi_model_from_string("FixedStack") is ABI_FIXED_STACK


class TestRepr:
    def test_default_repr(self):
        assert "Default" in repr(ABI_DEFAULT)

    def test_fixed_stack_repr(self):
        assert "FixedStack" in repr(ABI_FIXED_STACK)


# ---------------------------------------------------------------------------
# emit_call_args / emit_trait_dispatch_args
# ---------------------------------------------------------------------------

class _FakeTypeInfo:
    """Minimal type_info stub for testing."""
    def __init__(self, name='u8'):
        self.name = name


class _FakeValue:
    """Minimal value stub for Argument."""
    def __init__(self, type_info=None):
        self.type_info = type_info


class _FakeLocation:
    """Minimal location stub for REGISTER args."""
    def __init__(self, name):
        self.name = name


class _FakeStackTracker:
    def __init__(self):
        self.displacement = 0
        self.pushes = []
    def push(self, n):
        self.displacement += n
        self.pushes.append(n)
    def pop(self, n):
        self.displacement -= n


class _FakeRegionState:
    def __init__(self):
        self.stack_tracker = _FakeStackTracker()


class _FakePhysicalLocation:
    def __init__(self, kind='STACK'):
        from r65.compiler.codegen.register_alloc import LocationKind
        self.kind = LocationKind[kind]


class _FakeParent:
    """Minimal parent stub (InstructionSelector)."""
    def __init__(self):
        self._operand_locations = {}

    def _get_operand_location(self, value):
        return self._operand_locations.get(id(value), _FakePhysicalLocation('MEMORY'))

    def _offset_location(self, loc, adj):
        return loc


class _MockSelector:
    """Records which emission methods were called."""
    def __init__(self, spill_offset=0):
        self.calls = []
        self.parent = _FakeParent()
        self.region_state = _FakeRegionState()
        self._spill_offset = spill_offset

    def get_current_spill_offset(self):
        return self._spill_offset

    def emit_pha_stack_argument(self, arg, arg_loc, arg_size):
        self.calls.append(('pha_stack', arg.mechanism.value, arg_size))

    def emit_outgoing_stack_argument(self, arg, arg_loc, outgoing_offset):
        self.calls.append(('outgoing_stack', arg.mechanism.value, outgoing_offset))

    def emit_register_argument(self, arg, arg_loc):
        self.calls.append(('register', arg.location.name))

    def emit_variable_argument(self, arg, arg_loc):
        self.calls.append(('variable',))

    def emit_scratch_param_argument(self, arg, arg_loc):
        self.calls.append(('scratch_param', arg.scratch_addr))

    def load_y_with_self(self, arg, stack_bytes_pushed):
        self.calls.append(('load_y_self',))

    def arg_sort_key(self, arg):
        from r65.compiler.mir.nodes import ArgumentMechanism
        order = {
            ArgumentMechanism.STACK: 0,
            ArgumentMechanism.SCRATCH_PARAM: 1,
            ArgumentMechanism.VARIABLE: 2,
            ArgumentMechanism.REGISTER: 3,
        }
        return order.get(arg.mechanism, 5)


def _make_arg(mechanism_str, **kwargs):
    """Create an Argument dataclass for testing."""
    from r65.compiler.mir.nodes import Argument, ArgumentMechanism
    mechanism = ArgumentMechanism(mechanism_str)
    value = kwargs.pop('value', _FakeValue())
    return Argument(
        value=value,
        mechanism=mechanism,
        **kwargs,
    )


class TestEmitCallArgs:
    """Tests for ABIModel.emit_call_args."""

    def test_no_args_returns_zero(self):
        """Call with no args returns 0 bytes pushed."""
        from r65.compiler.mir.nodes import Call
        instr = Call(function='foo', args=[], returns=[], is_far=False)
        selector = _MockSelector()
        result = ABI_DEFAULT.emit_call_args(selector, instr)
        assert result == 0
        assert selector.calls == []

    def test_sta_mode_stack_args(self):
        """STA mode (no spills): stack args emit outgoing_stack calls."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector(spill_offset=0)
        result = ABI_DEFAULT.emit_call_args(selector, instr)
        assert result == 0  # STA mode pushes nothing
        assert len([c for c in selector.calls if c[0] == 'outgoing_stack']) == 2
        # First at offset 1, second at offset 2
        assert selector.calls[0] == ('outgoing_stack', 'stack', 1)
        assert selector.calls[1] == ('outgoing_stack', 'stack', 2)

    def test_pha_mode_stack_args(self):
        """PHA mode (spills active): stack args emit pha_stack calls."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector(spill_offset=2)  # Spills active
        result = ABI_DEFAULT.emit_call_args(selector, instr)
        assert result == 2  # 2 bytes pushed
        assert len([c for c in selector.calls if c[0] == 'pha_stack']) == 2

    def test_register_args(self):
        """Register args emit register calls."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('register', location=_FakeLocation('A')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_DEFAULT.emit_call_args(selector, instr)
        assert ('register', 'A') in selector.calls

    def test_variable_args(self):
        """Variable-bound args emit variable calls."""
        from r65.compiler.mir.nodes import Call
        args = [_make_arg('variable', location=_FakeLocation('TEMP'))]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_DEFAULT.emit_call_args(selector, instr)
        assert ('variable',) in selector.calls

    def test_scratch_param_args(self):
        """Scratch param args emit scratch_param calls."""
        from r65.compiler.mir.nodes import Call
        args = [_make_arg('scratch_param', scratch_addr=0x10)]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_DEFAULT.emit_call_args(selector, instr)
        assert ('scratch_param', 0x10) in selector.calls

    def test_mixed_args_order(self):
        """Mixed args are processed: stack first, then others sorted."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('register', location=_FakeLocation('A')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('variable', location=_FakeLocation('VAR')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_DEFAULT.emit_call_args(selector, instr)
        # Stack first via outgoing, then variable before register (sort key)
        assert selector.calls[0][0] == 'outgoing_stack'
        assert selector.calls[1][0] == 'variable'
        assert selector.calls[2][0] == 'register'

    def test_fixed_stack_abi_same_behavior(self):
        """FixedStack ABI uses the same concrete method."""
        from r65.compiler.mir.nodes import Call
        args = [_make_arg('register', location=_FakeLocation('X'))]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_FIXED_STACK.emit_call_args(selector, instr)
        assert ('register', 'X') in selector.calls


class TestEmitTraitDispatchArgs:
    """Tests for ABIModel.emit_trait_dispatch_args."""

    def test_no_args_returns_zero(self):
        from r65.compiler.mir.nodes import TraitDispatch
        instr = TraitDispatch(
            trait_name='MyTrait', method_name='do_thing',
            args=[], returns=[], is_far=False,
        )
        selector = _MockSelector()
        result = ABI_DEFAULT.emit_trait_dispatch_args(selector, instr)
        assert result == 0
        assert selector.calls == []

    def test_self_y_arg(self):
        """SELF_Y arg triggers load_y_with_self."""
        from r65.compiler.mir.nodes import TraitDispatch
        args = [_make_arg('self_y')]
        instr = TraitDispatch(
            trait_name='MyTrait', method_name='do_thing',
            args=args, returns=[], is_far=False,
        )
        selector = _MockSelector()
        ABI_DEFAULT.emit_trait_dispatch_args(selector, instr)
        assert ('load_y_self',) in selector.calls

    def test_stack_args_sta_mode(self):
        """Stack args in STA mode emit outgoing_stack calls."""
        from r65.compiler.mir.nodes import TraitDispatch
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('self_y'),
        ]
        instr = TraitDispatch(
            trait_name='T', method_name='m',
            args=args, returns=[], is_far=False,
        )
        selector = _MockSelector(spill_offset=0)
        result = ABI_DEFAULT.emit_trait_dispatch_args(selector, instr)
        assert result == 0
        assert any(c[0] == 'outgoing_stack' for c in selector.calls)
        assert ('load_y_self',) in selector.calls

    def test_stack_args_pha_mode(self):
        """Stack args in PHA mode (spills active) emit pha_stack calls."""
        from r65.compiler.mir.nodes import TraitDispatch
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = TraitDispatch(
            trait_name='T', method_name='m',
            args=args, returns=[], is_far=False,
        )
        selector = _MockSelector(spill_offset=2)
        result = ABI_DEFAULT.emit_trait_dispatch_args(selector, instr)
        assert result == 1
        assert any(c[0] == 'pha_stack' for c in selector.calls)

    def test_fixed_stack_abi_same_behavior(self):
        """FixedStack ABI uses the same concrete method."""
        from r65.compiler.mir.nodes import TraitDispatch
        args = [_make_arg('self_y')]
        instr = TraitDispatch(
            trait_name='T', method_name='m',
            args=args, returns=[], is_far=False,
        )
        selector = _MockSelector()
        ABI_FIXED_STACK.emit_trait_dispatch_args(selector, instr)
        assert ('load_y_self',) in selector.calls


# ---------------------------------------------------------------------------
# emit_frame_dealloc
# ---------------------------------------------------------------------------

class _FakeEmitter:
    """Minimal emitter stub for accu_mode tracking."""
    def __init__(self, accu_mode=8):
        self._accu_mode = accu_mode
    def get_accu_mode(self):
        return self._accu_mode


class _FakeParentWithEmitter:
    """Parent stub with an emitter for frame dealloc tests."""
    def __init__(self, accu_mode=8):
        self.emitter = _FakeEmitter(accu_mode)


class _DeallocSelector:
    """Records which frame-dealloc path was taken."""
    def __init__(self, accu_mode=8):
        self.parent = _FakeParentWithEmitter(accu_mode)
        self.path = None
        self.args = None

    def emit_pla_frame_dealloc(self, frame_size, return_count):
        self.path = 'pla'
        self.args = (frame_size, return_count)

    def emit_sp_adjust_preserving_a(self, adjust_bytes, return_count, current_mode):
        self.path = 'sp_adjust'
        self.args = (adjust_bytes, return_count, current_mode)


class TestEmitFrameDealloc:
    """Tests for ABIModel.emit_frame_dealloc."""

    def test_zero_frame_is_noop(self):
        sel = _DeallocSelector()
        ABI_DEFAULT.emit_frame_dealloc(sel, frame_size=0, return_count=0)
        assert sel.path is None

    def test_default_small_frame_uses_pla(self):
        sel = _DeallocSelector()
        ABI_DEFAULT.emit_frame_dealloc(sel, frame_size=2, return_count=1)
        assert sel.path == 'pla'
        assert sel.args == (2, 1)

    def test_default_large_frame_uses_sp_adjust(self):
        sel = _DeallocSelector()
        ABI_DEFAULT.emit_frame_dealloc(sel, frame_size=8, return_count=1)
        assert sel.path == 'sp_adjust'
        assert sel.args == (8, 1, 8)  # current_mode=8

    def test_fixed_stack_large_frame_uses_pla(self):
        """FixedStack always uses PLA (max_pla_dealloc_size is huge)."""
        sel = _DeallocSelector()
        ABI_FIXED_STACK.emit_frame_dealloc(sel, frame_size=8, return_count=0)
        assert sel.path == 'pla'
        assert sel.args == (8, 0)

    def test_default_boundary_frame_uses_pla(self):
        """frame_size=4 is exactly at the Default threshold — should PLA."""
        sel = _DeallocSelector()
        ABI_DEFAULT.emit_frame_dealloc(sel, frame_size=4, return_count=0)
        assert sel.path == 'pla'

    def test_default_boundary_plus_one_uses_sp_adjust(self):
        """frame_size=5 exceeds Default threshold — should SP adjust."""
        sel = _DeallocSelector()
        ABI_DEFAULT.emit_frame_dealloc(sel, frame_size=5, return_count=0)
        assert sel.path == 'sp_adjust'

    def test_sp_adjust_passes_current_mode(self):
        """SP-adjust path passes the emitter's current accu mode."""
        sel = _DeallocSelector(accu_mode=16)
        ABI_DEFAULT.emit_frame_dealloc(sel, frame_size=8, return_count=2)
        assert sel.path == 'sp_adjust'
        assert sel.args == (8, 2, 16)
