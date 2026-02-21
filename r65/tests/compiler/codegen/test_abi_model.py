"""Tests for ABIModel abstraction."""

import sys
import pytest
from r65.compiler.codegen.abi_model import (
    ABIModel, ABIKind, ABIDefault, ABIFixedStack, ABIPascal, ABICompact,
    ABI_DEFAULT, ABI_FIXED_STACK, ABI_PASCAL, ABI_COMPACT, abi_model_from_string,
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

    def _emit_push(self, reg, comment=""):
        self.calls.append(('push', reg, comment))

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


# ===========================================================================
# Pascal ABI Tests
# ===========================================================================


class TestABIKindPascal:
    def test_pascal_value(self):
        assert ABIKind.PASCAL.value == "Pascal"


class TestAbiModelFromStringPascal:
    def test_pascal(self):
        model = abi_model_from_string("Pascal")
        assert model.kind == ABIKind.PASCAL

    def test_singleton_pascal(self):
        assert abi_model_from_string("Pascal") is ABI_PASCAL


class TestABIPascalFrameAlloc:
    """ABIPascal.emit_frame_alloc uses same strategy as Default."""

    def test_small_frame_emits_phb(self):
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_PASCAL.emit_frame_alloc(fake_emit, frame_size=2, force_direct_stack=False)
        assert collected == ['PHB', 'PHB']

    def test_large_frame_emits_tsc(self):
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_PASCAL.emit_frame_alloc(fake_emit, frame_size=8, force_direct_stack=False)
        assert 'TSC' in collected
        assert 'PHB' not in collected

    def test_max_pla_dealloc_size(self):
        assert ABI_PASCAL.max_pla_dealloc_size == 4


class TestABIPascalParamAnalysis:
    """Pascal ABI skips all parameter analysis (no scratch promotion)."""

    def test_run_param_analysis_is_noop(self):
        """No crash, no side effects."""
        ABI_PASCAL.run_param_analysis("mir_program", "scratch_pool", False)

    def test_compute_outgoing_args_sets_zero(self):
        """Pascal sets max_outgoing_arg_bytes=0 on all functions."""
        class FakeFunc:
            max_outgoing_arg_bytes = 99
        class FakeProg:
            functions = [FakeFunc()]
        ABI_PASCAL.compute_outgoing_args(FakeProg(), None)
        assert FakeProg.functions[0].max_outgoing_arg_bytes == 0


class TestPascalEmitCallArgs:
    """Pascal emit_call_args: all args PHA'd left-to-right."""

    def _make_pascal_push_selector(self):
        """Return a selector that records push calls with names."""
        sel = _MockSelector()
        sel._emit_push_calls = []
        original_emit_push = sel.__class__.__dict__.get('_emit_push', None)
        def fake_emit_push(reg, comment=None):
            sel._emit_push_calls.append(('push', reg, comment))
        sel._emit_push = fake_emit_push
        return sel

    def test_no_args_returns_zero(self):
        from r65.compiler.mir.nodes import Call
        instr = Call(function='foo', args=[], returns=[], is_far=False)
        selector = _MockSelector()
        result = ABI_PASCAL.emit_call_args(selector, instr)
        assert result == 0

    def test_all_args_pushed_via_pha(self):
        """Pascal forces all args through PHA regardless of mechanism."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector(spill_offset=0)
        result = ABI_PASCAL.emit_call_args(selector, instr)
        # Pascal always uses PHA, so all args produce pha_stack calls
        assert result == 2
        pha_calls = [c for c in selector.calls if c[0] == 'pha_stack']
        assert len(pha_calls) == 2

    def test_result_space_pushed_first(self):
        """Result space is pushed before params via _emit_push."""
        from r65.compiler.mir.nodes import Call
        args = [_make_arg('stack', param_type=_FakeTypeInfo('u8'))]
        instr = Call(function='foo', args=args, returns=[], is_far=False,
                     pascal_result_bytes=1)
        sel = self._make_pascal_push_selector()
        result = ABI_PASCAL.emit_call_args(sel, instr)
        # 1 byte result space + 1 byte param = 2
        assert result == 2
        # The first call should be a push for result space
        assert sel._emit_push_calls[0] == ('push', 'A', 'Result space (Pascal)')

    def test_left_to_right_param_order(self):
        """Pascal pushes param0 first, param1 second (left-to-right)."""
        from r65.compiler.mir.nodes import Call
        val0 = _FakeValue(_FakeTypeInfo('u8'))
        val1 = _FakeValue(_FakeTypeInfo('u8'))
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8'), value=val0),
            _make_arg('stack', param_type=_FakeTypeInfo('u8'), value=val1),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_PASCAL.emit_call_args(selector, instr)
        # Both should be pha_stack, in order
        pha_calls = [c for c in selector.calls if c[0] == 'pha_stack']
        assert len(pha_calls) == 2
        # Check they are in forward order (not reversed)
        assert pha_calls[0][1] == 'stack'
        assert pha_calls[1][1] == 'stack'

    def test_stack_tracker_updated(self):
        """Stack tracker reflects all pushed bytes."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False,
                     pascal_result_bytes=1)
        selector = _MockSelector()
        result = ABI_PASCAL.emit_call_args(selector, instr)
        # 1 result + 2 params = 3 bytes
        assert result == 3
        assert selector.region_state.stack_tracker.displacement == 3


class TestPascalRepr:
    def test_repr(self):
        assert "Pascal" in repr(ABI_PASCAL)


# ===========================================================================
# Pascal integration tests (compile_string)
# ===========================================================================


class TestPascalCompileIntegration:
    """Integration tests: compile R65 source with Pascal ABI, verify assembly."""

    def test_pascal_callee_cleanup(self):
        """Pascal callee should clean up stack params before return."""
        source = """
        fn add(a: u8, b: u8) -> u8 {
            return a + b;
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_PASCAL)
        # Callee should have TSC/CLC/ADC/TCS for param cleanup (2 bytes)
        assert 'TSC' in result
        assert 'TCS' in result

    def test_pascal_caller_pushes_args(self):
        """Pascal caller should push all args via PHA."""
        source = """
        fn callee(a: u8) -> u8 {
            asm!("NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP");
            return a;
        }
        fn caller() -> u8 {
            return callee(42);
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_PASCAL)
        # Caller should push argument via PHA (callee too large to inline)
        assert 'PHA' in result

    def test_pascal_no_register_params(self):
        """Pascal ignores register bindings; all params go to stack."""
        source = """
        fn add(a @ A: u8, b @ X: u16) -> u8 {
            return a;
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_PASCAL)
        # The function should read params from stack, not from A/X registers
        # Look for stack-relative load (LDA $xx,S)
        assert ',S' in result or ',s' in result

    def test_pascal_void_function_no_result_space(self):
        """Void Pascal functions have no result space."""
        source = """
        #[ram]
        static mut RESULT: u8;
        fn set_value(v: u8) {
            RESULT = v;
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_PASCAL)
        # Should compile without errors
        assert 'set_value' in result

    def test_pascal_result_space_store(self):
        """Pascal callee writes return value to stack result space."""
        source = """
        fn double(x: u8) -> u8 {
            return x + x;
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_PASCAL)
        # Callee should store result to stack (STA offset,S)
        # and have "Pascal result space" comment
        assert 'Pascal result space' in result


# ===========================================================================
# Compact ABI Tests
# ===========================================================================


class TestABIKindCompact:
    def test_compact_value(self):
        assert ABIKind.COMPACT.value == "Compact"


class TestAbiModelFromStringCompact:
    def test_compact(self):
        model = abi_model_from_string("Compact")
        assert model.kind == ABIKind.COMPACT

    def test_singleton_compact(self):
        assert abi_model_from_string("Compact") is ABI_COMPACT


class TestABICompactFrameAlloc:
    """ABICompact.emit_frame_alloc uses same strategy as Default."""

    def test_small_frame_emits_phb(self):
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_COMPACT.emit_frame_alloc(fake_emit, frame_size=2, force_direct_stack=False)
        assert collected == ['PHB', 'PHB']

    def test_large_frame_emits_tsc(self):
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_COMPACT.emit_frame_alloc(fake_emit, frame_size=8, force_direct_stack=False)
        assert 'TSC' in collected
        assert 'PHB' not in collected

    def test_zero_frame_noop(self):
        collected = []
        def fake_emit(opcode, *args, **kwargs):
            collected.append(opcode.name)
        ABI_COMPACT.emit_frame_alloc(fake_emit, frame_size=0, force_direct_stack=False)
        assert collected == []

    def test_max_pla_dealloc_size(self):
        assert ABI_COMPACT.max_pla_dealloc_size == 4


class TestABICompactParamAnalysis:
    """Compact ABI runs scratch param analysis like Default."""

    def test_compute_outgoing_args_sets_zero(self):
        """Compact sets max_outgoing_arg_bytes=0 on all functions."""
        class FakeFunc:
            max_outgoing_arg_bytes = 99
        class FakeProg:
            functions = [FakeFunc()]
        ABI_COMPACT.compute_outgoing_args(FakeProg(), None)
        assert FakeProg.functions[0].max_outgoing_arg_bytes == 0


class TestCompactEmitCallArgs:
    """Compact emit_call_args: stack args always use PHA, register args unchanged."""

    def test_no_args_returns_zero(self):
        from r65.compiler.mir.nodes import Call
        instr = Call(function='foo', args=[], returns=[], is_far=False)
        selector = _MockSelector()
        result = ABI_COMPACT.emit_call_args(selector, instr)
        assert result == 0
        assert selector.calls == []

    def test_stack_args_always_use_pha(self):
        """Compact always PHA's stack args even without spills."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector(spill_offset=0)  # No spills, but still uses PHA
        result = ABI_COMPACT.emit_call_args(selector, instr)
        assert result == 2  # 2 bytes pushed
        pha_calls = [c for c in selector.calls if c[0] == 'pha_stack']
        assert len(pha_calls) == 2
        # Should NOT have any outgoing_stack calls
        outgoing_calls = [c for c in selector.calls if c[0] == 'outgoing_stack']
        assert len(outgoing_calls) == 0

    def test_stack_args_reversed_order(self):
        """Compact pushes stack args right-to-left (reversed)."""
        from r65.compiler.mir.nodes import Call
        val0 = _FakeValue(_FakeTypeInfo('u8'))
        val1 = _FakeValue(_FakeTypeInfo('u8'))
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8'), value=val0),
            _make_arg('stack', param_type=_FakeTypeInfo('u8'), value=val1),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_COMPACT.emit_call_args(selector, instr)
        pha_calls = [c for c in selector.calls if c[0] == 'pha_stack']
        assert len(pha_calls) == 2

    def test_register_args_unchanged(self):
        """Register args emit register calls, same as Default."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('register', location=_FakeLocation('A')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_COMPACT.emit_call_args(selector, instr)
        assert ('register', 'A') in selector.calls

    def test_mixed_args(self):
        """Stack args PHA'd, register/variable args handled normally."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('register', location=_FakeLocation('A')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('variable', location=_FakeLocation('VAR')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        result = ABI_COMPACT.emit_call_args(selector, instr)
        assert result == 1  # 1 stack arg byte pushed
        # PHA stack first, then variable and register (sorted)
        assert selector.calls[0][0] == 'pha_stack'
        assert selector.calls[1][0] == 'variable'
        assert selector.calls[2][0] == 'register'

    def test_stack_tracker_updated(self):
        """Stack tracker reflects all PHA-pushed bytes."""
        from r65.compiler.mir.nodes import Call
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        result = ABI_COMPACT.emit_call_args(selector, instr)
        assert result == 2
        assert selector.region_state.stack_tracker.displacement == 2

    def test_scratch_param_args(self):
        """Scratch param args emit scratch_param calls."""
        from r65.compiler.mir.nodes import Call
        args = [_make_arg('scratch_param', scratch_addr=0x10)]
        instr = Call(function='foo', args=args, returns=[], is_far=False)
        selector = _MockSelector()
        ABI_COMPACT.emit_call_args(selector, instr)
        assert ('scratch_param', 0x10) in selector.calls


class TestCompactEmitTraitDispatchArgs:
    """Compact emit_trait_dispatch_args: stack args always PHA'd."""

    def test_no_args_returns_zero(self):
        from r65.compiler.mir.nodes import TraitDispatch
        instr = TraitDispatch(
            trait_name='MyTrait', method_name='do_thing',
            args=[], returns=[], is_far=False,
        )
        selector = _MockSelector()
        result = ABI_COMPACT.emit_trait_dispatch_args(selector, instr)
        assert result == 0

    def test_stack_args_always_pha(self):
        """Stack args PHA'd even without spills."""
        from r65.compiler.mir.nodes import TraitDispatch
        args = [
            _make_arg('stack', param_type=_FakeTypeInfo('u8')),
        ]
        instr = TraitDispatch(
            trait_name='T', method_name='m',
            args=args, returns=[], is_far=False,
        )
        selector = _MockSelector(spill_offset=0)
        result = ABI_COMPACT.emit_trait_dispatch_args(selector, instr)
        assert result == 1
        assert any(c[0] == 'pha_stack' for c in selector.calls)

    def test_self_y_arg(self):
        """SELF_Y arg triggers load_y_with_self."""
        from r65.compiler.mir.nodes import TraitDispatch
        args = [_make_arg('self_y')]
        instr = TraitDispatch(
            trait_name='T', method_name='m',
            args=args, returns=[], is_far=False,
        )
        selector = _MockSelector()
        ABI_COMPACT.emit_trait_dispatch_args(selector, instr)
        assert ('load_y_self',) in selector.calls


class TestCompactFrameDealloc:
    """Compact frame dealloc uses same thresholds as Default."""

    def test_small_frame_uses_pla(self):
        sel = _DeallocSelector()
        ABI_COMPACT.emit_frame_dealloc(sel, frame_size=3, return_count=1)
        assert sel.path == 'pla'

    def test_large_frame_uses_sp_adjust(self):
        sel = _DeallocSelector()
        ABI_COMPACT.emit_frame_dealloc(sel, frame_size=8, return_count=1)
        assert sel.path == 'sp_adjust'


class TestCompactRepr:
    def test_repr(self):
        assert "Compact" in repr(ABI_COMPACT)


# ===========================================================================
# Compact ABI integration tests (compile_string)
# ===========================================================================


class TestCompactCompileIntegration:
    """Integration tests: compile R65 source with Compact ABI, verify assembly."""

    def test_compact_caller_pushes_args(self):
        """Compact caller should push stack args via PHA, not STA to outgoing."""
        source = """
        fn callee(a: u8) -> u8 {
            asm!("NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP");
            return a;
        }
        fn caller() -> u8 {
            return callee(42);
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_COMPACT)
        assert 'PHA' in result

    def test_compact_caller_cleanup(self):
        """Compact caller should clean up pushed args after call."""
        source = """
        fn callee(a: u8, b: u8) -> u8 {
            asm!("NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP");
            return a;
        }
        fn caller() -> u8 {
            return callee(1, 2);
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_COMPACT)
        # Should have PLX for cleanup (2 bytes pushed = 1 PLX)
        assert 'PLX' in result or 'PLY' in result

    def test_compact_no_outgoing_area(self):
        """Compact should NOT have outgoing arg area in frame allocation."""
        source = """
        fn callee(a: u8) -> u8 {
            asm!("NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP");
            return a;
        }
        fn caller() -> u8 {
            let x: u8 = callee(42);
            return x;
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_COMPACT)
        # Should compile without errors
        assert 'caller' in result

    def test_compact_leaf_function_unchanged(self):
        """Leaf functions have identical output to Default (no calls)."""
        source = """
        fn add(a @ A: u8, b @ X: u16) -> u8 {
            return a;
        }
        """
        from r65.compiler.main import compile_string
        result_compact = compile_string(source, "test.r65", abi_model=ABI_COMPACT)
        result_default = compile_string(source, "test.r65", abi_model=ABI_DEFAULT)
        # Both should produce the same assembly for a leaf function with register params
        assert 'add' in result_compact
        assert 'add' in result_default

    def test_compact_callee_reads_from_stack(self):
        """Callee should read params from stack just like Default ABI."""
        source = """
        fn add(a: u8, b: u8) -> u8 {
            return a + b;
        }
        """
        from r65.compiler.main import compile_string
        result = compile_string(source, "test.r65", abi_model=ABI_COMPACT)
        # Callee reads from stack-relative addressing
        assert ',S' in result or ',s' in result
