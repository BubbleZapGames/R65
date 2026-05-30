# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for the static stack-usage analyzer (r65.compiler.analysis.stack_usage).

Distinct from test_stack_depth.py — the new analyzer:
  - Raises CodegenError on overflow rather than emitting warnings.
  - Uses MIRFunction.codegen_max_region_spill_bytes (the recorded peak) for
    spill cost rather than the legacy constant MAX_SPILL_BYTES=6.
  - Includes MIRFunction.max_outgoing_arg_bytes in own_frame.
  - Resolves trait dispatch via CallGraph.trait_impls and indirect calls via
    CallGraph.address_taken.
"""

import pytest

from r65.compiler.mir.nodes import (
    BasicBlock, Call, FunctionPointer, MIRFunction, MIRProgram, Move,
    Return, TraitDispatch, VirtualRegister,
)
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.attributes import (
    InterruptAttribute, InterruptVector, StackAttribute,
)
from r65.compiler.errors import CodegenError
from r65.compiler.analysis.stack_usage import (
    StackUsageAnalyzer, analyze_stack_usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_func(name, *,
               is_entry=False, is_far=False,
               calls=None, far_calls=None, trait_calls=None, indirect_calls=None,
               address_taken=None,
               interrupt_attr=None, interrupt_modified_regs=None,
               frame_size=0, prologue_bytes=0,
               region_spill=0, outgoing_args=0):
    """Build a minimal MIRFunction with optional Call / TraitDispatch nodes."""
    instructions = []
    if calls:
        for callee in calls:
            instructions.append(Call(function=callee, is_far=False))
    if far_calls:
        for callee in far_calls:
            instructions.append(Call(function=callee, is_far=True))
    if trait_calls:
        for trait_name, method_name, is_far_call in trait_calls:
            instructions.append(TraitDispatch(
                trait_name=trait_name,
                method_name=method_name,
                is_far=is_far_call,
            ))
    if indirect_calls:
        for is_far_call in indirect_calls:
            vreg = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'))
            instructions.append(Call(function=vreg, is_far=is_far_call))
    if address_taken:
        for taken_name in address_taken:
            instructions.append(Move(
                dest=VirtualRegister(id=0, type_info=BasicTypeInfo('u16')),
                source=FunctionPointer(function_name=taken_name),
                type_info=BasicTypeInfo('u16'),
            ))
    instructions.append(Return())

    block = BasicBlock(
        block_id=0,
        instructions=instructions,
        predecessors=[],
        successors=[],
    )

    func = MIRFunction(
        name=name,
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: block},
        entry_block_id=0,
        exit_block_ids=[0],
        is_entry=is_entry,
        is_far=is_far,
        interrupt_attr=interrupt_attr,
        vreg_allocator=VirtualRegisterAllocator(),
    )
    func.codegen_frame_size = frame_size
    func.codegen_prologue_bytes = prologue_bytes
    func.codegen_max_region_spill_bytes = region_spill
    func.max_outgoing_arg_bytes = outgoing_args
    if interrupt_modified_regs is not None:
        func.interrupt_modified_regs = interrupt_modified_regs
    return func


def _program(*funcs, stack_lower=0x0100, stack_upper=0x01FF,
             trait_dispatch_info=None):
    prog = MIRProgram(functions=list(funcs))
    prog.stack_attr = StackAttribute(
        name="stack", lower=stack_lower, upper=stack_upper, source_loc=None,
    )
    if trait_dispatch_info is not None:
        prog.trait_dispatch_info = trait_dispatch_info
    return prog


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOwnFrameComposition:
    """own_frame(f) sums prologue + frame + region_spill + outgoing_args."""

    def test_single_entry_no_calls(self):
        main = _make_func("main", is_entry=True,
                          frame_size=4, prologue_bytes=2,
                          region_spill=0, outgoing_args=0)
        budget = analyze_stack_usage(_program(main))
        assert budget.entry_use == 6  # 2 + 4 + 0 + 0
        assert budget.total == 6
        assert budget.capacity == 256

    def test_own_frame_includes_outgoing_args(self):
        leaf = _make_func("leaf")
        main = _make_func("main", is_entry=True, calls=["leaf"],
                          frame_size=4, outgoing_args=6)
        budget = analyze_stack_usage(_program(main, leaf))
        # main: 0 + 4 + 0 + 6 = 10; call leaf (near = +2); leaf: 0 → 12.
        assert budget.entry_use == 12

    def test_own_frame_includes_region_spill(self):
        leaf = _make_func("leaf")
        main = _make_func("main", is_entry=True, calls=["leaf"],
                          region_spill=4)
        budget = analyze_stack_usage(_program(main, leaf))
        assert budget.entry_use == 6  # 0+0+4+0 + 2 (near call) + 0


class TestCallChain:
    """Bottom-up DFS picks the deepest chain."""

    def test_diamond_picks_deeper_branch(self):
        c = _make_func("c", frame_size=4)
        a = _make_func("a", calls=["c"], frame_size=50)
        b = _make_func("b", calls=["c"], frame_size=2)
        main = _make_func("main", is_entry=True, calls=["a", "b"], frame_size=2)
        budget = analyze_stack_usage(_program(main, a, b, c))
        # main(2) + [a edge 2 + a(50) + c edge 2 + c(4)] = 60
        assert budget.entry_use == 60
        chain_names = [name for name, _ in budget.deepest_chain]
        assert chain_names == ["main", "a", "c"]

    def test_far_call_costs_three(self):
        leaf = _make_func("leaf", is_far=True)
        main = _make_func("main", is_entry=True, far_calls=["leaf"])
        budget = analyze_stack_usage(_program(main, leaf))
        # 0 + 3 (JSL) + 0 = 3
        assert budget.entry_use == 3


class TestOverflow:
    """CodegenError on overflow with descriptive chain."""

    def test_overflow_raises_codegen_error(self):
        b = _make_func("b", frame_size=10)
        main = _make_func("main", is_entry=True, calls=["b"], frame_size=10)
        # main(10) + 2 (near) + b(10) = 22; declare 16-byte stack.
        with pytest.raises(CodegenError) as exc:
            analyze_stack_usage(_program(main, b, stack_upper=0x010F))
        msg = str(exc.value)
        assert "Stack overflow" in msg
        assert "main" in msg
        assert "b" in msg

    def test_in_budget_no_raise(self):
        b = _make_func("b", frame_size=4)
        main = _make_func("main", is_entry=True, calls=["b"], frame_size=4)
        # Total = 4 + 2 + 4 = 10 bytes
        budget = analyze_stack_usage(_program(main, b, stack_upper=0x010F))
        assert budget.total == 10
        assert budget.capacity == 16


class TestTraitDispatch:
    """Trait dispatch resolves to the impl set via CallGraph."""

    def test_picks_largest_impl(self):
        impl_small = _make_func("Trait__small", frame_size=2)
        impl_large = _make_func("Trait__large", frame_size=50)
        main = _make_func("main", is_entry=True,
                          trait_calls=[("Trait", "method", False)])
        info = {"Trait": {
            "is_far": False,
            "methods": ["method"],
            "implementors": [
                {"struct": "S1", "type_id": 1, "mangled": ["Trait__small"]},
                {"struct": "S2", "type_id": 2, "mangled": ["Trait__large"]},
            ],
        }}
        budget = analyze_stack_usage(_program(
            main, impl_small, impl_large, trait_dispatch_info=info,
        ))
        # main(0) + 2 (near trait) + Trait__large(50) = 52
        assert budget.entry_use == 52


class TestIndirectCalls:
    """Indirect (fn-pointer) call widens to the address-taken set."""

    def test_picks_largest_address_taken(self):
        small = _make_func("small", frame_size=2)
        large = _make_func("large", frame_size=50)
        # main takes addresses of both, then makes an indirect call.
        main = _make_func("main", is_entry=True,
                          address_taken=["small", "large"],
                          indirect_calls=[False])
        budget = analyze_stack_usage(_program(main, small, large))
        # main(0) + 2 (near indirect) + large(50) = 52
        assert budget.entry_use == 52
        assert budget.has_indirect_calls is True


class TestInterruptHandlers:
    """Interrupt overhead is added on top of the sync high-water mark."""

    def test_interrupt_adds_cpu_push_and_register_saves(self):
        # NMI handler: A + X + Y modified ⇒ 6 bytes of PHA/PHX/PHY,
        # plus 4 bytes the CPU pushes on entry (P+PB+PCH+PCL).
        nmi = _make_func(
            "nmi_handler", frame_size=2,
            interrupt_attr=InterruptAttribute(
                name="interrupt", vector=InterruptVector.NMI, preserve=True,
            ),
            interrupt_modified_regs={'A', 'X', 'Y'},
        )
        main = _make_func("main", is_entry=True, frame_size=4)
        budget = analyze_stack_usage(_program(main, nmi))
        assert budget.entry_use == 4
        # nmi: own=2, prologue=4(cpu) + 6(regs) = 10, total=12.
        assert budget.interrupt_extra == 12
        assert budget.total == 16
        assert budget.worst_handler == "nmi_handler"

    def test_no_interrupt_handler_is_zero(self):
        main = _make_func("main", is_entry=True, frame_size=4)
        budget = analyze_stack_usage(_program(main))
        assert budget.interrupt_extra == 0
        assert budget.worst_handler is None


class TestCapacityDefault:
    """Without #[stack(...)], capacity defaults to 256 bytes ($0100..$01FF)."""

    def test_default_capacity(self):
        main = _make_func("main", is_entry=True)
        prog = MIRProgram(functions=[main])  # no stack_attr
        budget = analyze_stack_usage(prog)
        assert budget.capacity == 256
