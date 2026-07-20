#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for function inlining optimization pass."""

import pytest

from r65.compiler.optimize.inline import (
    FunctionInliner,
    InlinabilityChecker,
    BlockCloner,
    INLINE_COST_WITH_ATTR,
    INLINE_COST_NO_ATTR,
)
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock,
    Move, BinaryOp, UnaryOp, Compare, Jump, CondBranch, Return, Call,
    Load, Store, LoadIndirect, StoreIndirect, MemoryLocation,
    VirtualRegister, HardwareRegister, Immediate,
    InlineAsm, Argument, ArgumentMechanism, SetMode,
    BitTest, Rotate, Push, Pull, SaveRegister, RestoreRegister,
    TraitDispatch, FarPtrStrategy,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.attributes import (
    InlineAttribute, InlineMode, InterruptAttribute, InterruptVector,
    BankAttribute, PreservesAttribute, ModeAttribute, DataBankMode,
)
from r65.compiler.hir.nodes import HIRParameter, RegisterBinding
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.typeck.processor_mode import ModeState, ProcessorMode, XModeState
from r65.tests.language.common import make_mir_function


class MockSymbol:
    """Simple mock symbol for testing MemoryLocation."""
    def __init__(self, name: str):
        self.name = name


INLINE = InlineAttribute(name='inline')


def create_simple_callee():
    """Create a simple function to be inlined: add_one(x) -> x + 1"""
    func = make_mir_function("add_one", inline_attr=INLINE)

    vreg_x = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="x")
    vreg_result = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="result")

    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_x, source=Immediate(5), type_info=BasicTypeInfo('u8')),
            BinaryOp(dest=vreg_result, left=vreg_x, right=Immediate(1), op='+',
                     type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_caller_with_call(callee_name: str):
    """Create a caller function that calls the given function."""
    func = make_mir_function("main", is_entry=True)

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")

    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Call(function=callee_name, args=[], returns=[vreg_result], is_far=False),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_recursive_function():
    """Create a recursive function that should not be inlined."""
    func = make_mir_function("factorial", inline_attr=INLINE)
    func.exit_block_ids = [1, 2]

    vreg_n = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="n")
    vreg_cond = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="cond")
    vreg_result = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="result")

    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_n, source=Immediate(5), type_info=BasicTypeInfo('u8')),
            CondBranch(condition=vreg_cond, true_target=1, false_target=2)
        ],
        predecessors=[], successors=[1, 2]
    )
    func.blocks[1] = BasicBlock(
        block_id=1,
        instructions=[
            Move(dest=vreg_result, source=Immediate(1), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[0], successors=[]
    )
    func.blocks[2] = BasicBlock(
        block_id=2,
        instructions=[
            Call(function="factorial", args=[], returns=[vreg_result], is_far=False),
            Return(values=[vreg_result])
        ],
        predecessors=[0], successors=[]
    )
    return func


def create_far_function():
    """Create a far function that should not be inlined."""
    func = make_mir_function("far_helper", is_far=True, inline_attr=INLINE)

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_result, source=Immediate(42), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_function_with_asm():
    """Create a function with inline assembly that should not be inlined."""
    func = make_mir_function("asm_helper", inline_attr=INLINE)

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            InlineAsm(instructions=["NOP"]),
            Move(dest=vreg_result, source=Immediate(42), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_interrupt_handler():
    """Create an interrupt handler that should not be inlined."""
    func = make_mir_function("nmi_handler", return_type=None,
                             interrupt_attr=InterruptAttribute(name='interrupt', vector=InterruptVector.NMI),
                             inline_attr=INLINE)
    func.blocks[0] = BasicBlock(
        block_id=0, instructions=[Return(values=[])],
        predecessors=[], successors=[]
    )
    return func


def create_function_with_multiple_returns():
    """Create a function with multiple return paths."""
    func = make_mir_function("multi_return", inline_attr=INLINE)
    func.exit_block_ids = [1, 2]

    vreg_cond = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="cond")
    vreg_result1 = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="result1")
    vreg_result2 = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="result2")

    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_cond, source=Immediate(1), type_info=BasicTypeInfo('u8')),
            CondBranch(condition=vreg_cond, true_target=1, false_target=2)
        ],
        predecessors=[], successors=[1, 2]
    )
    func.blocks[1] = BasicBlock(
        block_id=1,
        instructions=[
            Move(dest=vreg_result1, source=Immediate(10), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result1])
        ],
        predecessors=[0], successors=[]
    )
    func.blocks[2] = BasicBlock(
        block_id=2,
        instructions=[
            Move(dest=vreg_result2, source=Immediate(20), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result2])
        ],
        predecessors=[0], successors=[]
    )
    return func


def create_large_function(num_instructions: int):
    """Create a function with many instructions."""
    func = make_mir_function("large_func", inline_attr=INLINE)

    vreg_acc = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="acc")
    instructions = [
        Move(dest=vreg_acc, source=Immediate(0), type_info=BasicTypeInfo('u8'))
    ]
    for i in range(num_instructions - 2):
        instructions.append(
            BinaryOp(dest=vreg_acc, left=vreg_acc, right=Immediate(1), op='+',
                     type_info=BasicTypeInfo('u8'))
        )
    instructions.append(Return(values=[vreg_acc]))

    func.blocks[0] = BasicBlock(
        block_id=0, instructions=instructions,
        predecessors=[], successors=[]
    )
    return func


def create_getter_function():
    """Create a simple getter function: fn get_value() -> u8 { return 15; }"""
    func = make_mir_function("get_value", inline_attr=INLINE)

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_result, source=Immediate(15), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_getter_with_load():
    """Create a getter that loads from memory: fn get_static() -> u8 { return STATIC; }"""
    func = make_mir_function("get_static", inline_attr=INLINE)

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    mem_loc = MemoryLocation(storage_type="zeropage", address=0x10, symbol=MockSymbol("STATIC"))
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Load(dest=vreg_result, source=mem_loc, type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_setter_function():
    """Create a simple setter function: fn set_value(v @ A: u8) { STATIC = v; }"""
    func = make_mir_function("set_value", return_type=None, inline_attr=INLINE)

    vreg_value = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="value")
    mem_loc = MemoryLocation(storage_type="zeropage", address=0x10, symbol=MockSymbol("STATIC"))
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Store(dest=mem_loc, source=vreg_value, type_info=BasicTypeInfo('u8')),
            Return(values=[])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_pointer_getter_function():
    """Create a pointer-based getter: fn get_damage(*self) -> u8 { return self.damage; }"""
    func = make_mir_function("get_damage", inline_attr=INLINE)

    vreg_self = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint="self")
    vreg_result = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="result")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            LoadIndirect(dest=vreg_result, pointer=vreg_self, is_far=False,
                        index_register='Y', type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_pointer_setter_function():
    """Create a pointer-based setter: fn set_damage(*self, v @ A: u8) { self.damage = v; }"""
    func = make_mir_function("set_damage", return_type=None, inline_attr=INLINE)

    vreg_self = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint="self")
    vreg_value = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="value")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            StoreIndirect(source=vreg_value, pointer=vreg_self, is_far=False,
                         index_register='Y', type_info=BasicTypeInfo('u8')),
            Return(values=[])
        ],
        predecessors=[], successors=[]
    )
    return func


class TestInlinabilityChecker:
    """Tests for InlinabilityChecker."""

    def test_can_inline_simple_function(self):
        """Simple function should be inlinable."""
        callee = create_simple_callee()
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        assert checker.can_inline("add_one") is True

    def test_cannot_inline_recursive_function(self):
        """Recursive functions should not be inlinable."""
        recursive_func = create_recursive_function()
        caller = create_caller_with_call("factorial")
        program = MIRProgram(functions=[caller, recursive_func])

        checker = InlinabilityChecker(program)
        assert checker.can_inline("factorial") is False

    def test_bank_safe_far_function_inlinable(self):
        """Bank-safe far functions (no nested near calls, no databank attr,
        all memory accesses bank-independent) ARE inlinable — the body
        produces identical bytes regardless of caller's bank. Updated
        from the older blanket-reject rule."""
        far_func = create_far_function()  # trivial Move + Return body
        caller = create_caller_with_call("far_helper")
        program = MIRProgram(functions=[caller, far_func])

        checker = InlinabilityChecker(program)
        assert checker.can_inline("far_helper") is True

    def test_cannot_inline_interrupt_handler(self):
        """Interrupt handlers should not be inlinable."""
        handler = create_interrupt_handler()
        caller = create_caller_with_call("nmi_handler")
        program = MIRProgram(functions=[caller, handler])

        checker = InlinabilityChecker(program)
        assert checker.can_inline("nmi_handler") is False

    def test_cannot_inline_entry_point(self):
        """Entry point should not be inlinable."""
        caller = create_caller_with_call("add_one")
        callee = create_simple_callee()
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        # main is the entry point
        assert checker.can_inline("main") is False

    def test_cannot_inline_function_with_asm(self):
        """Functions with inline assembly should not be inlinable."""
        asm_func = create_function_with_asm()
        caller = create_caller_with_call("asm_helper")
        program = MIRProgram(functions=[caller, asm_func])

        checker = InlinabilityChecker(program)
        assert checker.can_inline("asm_helper") is False

    def test_should_inline_called_once(self):
        """Functions called exactly once should be inlined."""
        callee = create_simple_callee()
        callee.inline_attr = None  # No inline attribute
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        assert checker.should_inline("add_one") is True

    def test_should_inline_with_attribute(self):
        """Functions marked #[inline] should be inlined if under threshold."""
        callee = create_simple_callee()
        callee.inline_attr = InlineAttribute(name='inline')
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        assert checker.should_inline("add_one") is True

    def test_should_not_inline_large_function(self):
        """Large functions should not be inlined even with #[inline]."""
        # Create a function with many instructions
        large_func = create_large_function(INLINE_COST_WITH_ATTR + 10)
        large_func.inline_attr = InlineAttribute(name='inline')
        caller = create_caller_with_call("large_func")

        # Create another caller so the function is called twice
        caller2 = MIRFunction(
            name="caller2",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
        caller2.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="large_func", args=[], returns=[vreg_result], is_far=False),
                Return(values=[vreg_result])
            ],
            predecessors=[],
            successors=[]
        )

        program = MIRProgram(functions=[caller, caller2, large_func])

        checker = InlinabilityChecker(program)
        # Large function should not be inlined when called more than once
        assert checker.should_inline("large_func") is False

    def test_should_inline_small_function_no_attribute(self):
        """Small functions should be inlined even without #[inline]."""
        callee = create_simple_callee()
        callee.inline_attr = None  # No inline attribute
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        # Called once, so should inline
        assert checker.should_inline("add_one") is True

    def test_should_inline_getter_function(self):
        """Getter functions should be auto-inlined even without #[inline]."""
        getter = create_getter_function()
        caller = create_caller_with_call("get_value")

        # Create a second caller so the function is called twice
        caller2 = MIRFunction(
            name="caller2",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
        caller2.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="get_value", args=[], returns=[vreg_result], is_far=False),
                Return(values=[vreg_result])
            ],
            predecessors=[],
            successors=[]
        )

        program = MIRProgram(functions=[caller, caller2, getter])

        checker = InlinabilityChecker(program)
        # Getter should be inlined even though called twice and no #[inline]
        assert checker.should_inline("get_value") is True

    def test_should_inline_getter_with_load(self):
        """Getter that loads from static should be auto-inlined."""
        getter = create_getter_with_load()
        caller = create_caller_with_call("get_static")

        # Create a second caller
        caller2 = MIRFunction(
            name="caller2",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
        caller2.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="get_static", args=[], returns=[vreg_result], is_far=False),
                Return(values=[vreg_result])
            ],
            predecessors=[],
            successors=[]
        )

        program = MIRProgram(functions=[caller, caller2, getter])

        checker = InlinabilityChecker(program)
        # Getter with Load should also be inlined
        assert checker.should_inline("get_static") is True

    def test_should_inline_setter_function(self):
        """Setter functions should be auto-inlined even without #[inline]."""
        setter = create_setter_function()

        # Create caller that calls the setter
        caller = MIRFunction(
            name="main",
            parameters=[],
            return_type=None,
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=True,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="set_value", args=[], returns=[], is_far=False),
                Return(values=[])
            ],
            predecessors=[],
            successors=[]
        )

        # Create second caller
        caller2 = MIRFunction(
            name="caller2",
            parameters=[],
            return_type=None,
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        caller2.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="set_value", args=[], returns=[], is_far=False),
                Return(values=[])
            ],
            predecessors=[],
            successors=[]
        )

        program = MIRProgram(functions=[caller, caller2, setter])

        checker = InlinabilityChecker(program)
        # Setter should be inlined even though called twice and no #[inline]
        assert checker.should_inline("set_value") is True

    def test_should_inline_pointer_getter(self):
        """Pointer-based getter (LoadIndirect) should be auto-inlined."""
        # Tests: fn get_damage(*self) -> u8 { return self.damage; }
        getter = create_pointer_getter_function()

        # Create caller that calls get_damage multiple times
        caller = MIRFunction(
            name="main",
            parameters=[],
            return_type=None,
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=True,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="get_damage", args=[], returns=[vreg_result], is_far=False),
                Call(function="get_damage", args=[], returns=[vreg_result], is_far=False),
                Return(values=[])
            ],
            predecessors=[],
            successors=[]
        )

        program = MIRProgram(functions=[caller, getter])

        checker = InlinabilityChecker(program)
        # Pointer getter should be inlined (has inline_attr from HIR auto-detection)
        assert checker.should_inline("get_damage") is True

    def test_should_inline_pointer_setter(self):
        """Pointer-based setter (StoreIndirect) should be auto-inlined."""
        # Tests: fn set_damage(*self, v @ A: u8) { self.damage = v; }
        setter = create_pointer_setter_function()

        # Create caller that calls set_damage multiple times
        caller = MIRFunction(
            name="main",
            parameters=[],
            return_type=None,
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=True,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="set_damage", args=[], returns=[], is_far=False),
                Call(function="set_damage", args=[], returns=[], is_far=False),
                Return(values=[])
            ],
            predecessors=[],
            successors=[]
        )

        program = MIRProgram(functions=[caller, setter])

        checker = InlinabilityChecker(program)
        # Pointer setter should be inlined (has inline_attr from HIR auto-detection)
        assert checker.should_inline("set_damage") is True


class TestBlockCloner:
    """Tests for BlockCloner."""

    def test_clone_preserves_instruction_count(self):
        """Cloned blocks should have the same number of instructions."""
        callee = create_simple_callee()
        caller = create_caller_with_call("add_one")

        cloner = BlockCloner(caller, callee)
        cloned_blocks = cloner.clone_blocks()

        original_instr_count = sum(len(b.instructions) for b in callee.blocks.values())
        cloned_instr_count = sum(len(b.instructions) for b in cloned_blocks.values())

        assert cloned_instr_count == original_instr_count

    def test_clone_remaps_vreg_ids(self):
        """Cloned blocks should have new vreg IDs based on caller's allocator."""
        callee = create_simple_callee()
        caller = create_caller_with_call("add_one")

        # Pre-allocate some vregs in the caller so cloned vregs get different IDs
        caller.vreg_allocator.alloc(BasicTypeInfo('u8'), "pre_existing1")
        caller.vreg_allocator.alloc(BasicTypeInfo('u8'), "pre_existing2")

        cloner = BlockCloner(caller, callee)
        cloned_blocks = cloner.clone_blocks()

        # Collect vreg IDs from cloned blocks
        cloned_vreg_ids = set()
        for block in cloned_blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Move):
                    if isinstance(instr.dest, VirtualRegister):
                        cloned_vreg_ids.add(instr.dest.id)

        # Collect vreg IDs from original blocks
        original_vreg_ids = set()
        for block in callee.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Move):
                    if isinstance(instr.dest, VirtualRegister):
                        original_vreg_ids.add(instr.dest.id)

        # IDs should be different since caller has pre-existing vregs
        assert cloned_vreg_ids != original_vreg_ids
        # Cloned IDs should start from 2 (after pre_existing1 and pre_existing2)
        assert min(cloned_vreg_ids) >= 2

    def test_clone_remaps_block_ids(self):
        """Cloned blocks should have new block IDs."""
        callee = create_function_with_multiple_returns()
        caller = create_caller_with_call("multi_return")

        cloner = BlockCloner(caller, callee)
        cloned_blocks = cloner.clone_blocks()

        original_block_ids = set(callee.blocks.keys())
        cloned_block_ids = set(cloned_blocks.keys())

        # IDs should be different (except possibly by coincidence)
        # At minimum, check that we have the same number of blocks
        assert len(cloned_block_ids) == len(original_block_ids)

    def test_clone_invalidates_block_mode_metadata(self):
        """Cloned blocks must arrive with entry_mode/exit_mode = None.

        After Bug 1 was rediagnosed as a metadata-staleness issue, the
        inliner's contract is to TREAT block-level mode metadata as
        invalidated by any inline. Callers (codegen) re-run
        MIRModeTracker on every function the inliner mutated. Preserving
        the callee's stale mode info on cloned blocks was a workaround
        that happened to work when callee and caller had compatible
        mode flow but broke at general join points.
        """
        callee = create_simple_callee()
        # Even if the callee carries mode info, the cloned blocks must
        # not. (The MIR builder would normally set these.)
        for block in callee.blocks.values():
            block.entry_mode = ProcessorMode(ModeState.M16, XModeState.X16)
            block.exit_mode = ProcessorMode(ModeState.M16, XModeState.X16)

        caller = create_caller_with_call("add_one")
        cloner = BlockCloner(caller, callee)
        cloned_blocks = cloner.clone_blocks()

        assert len(cloned_blocks) == len(callee.blocks)
        for new_id, dst in cloned_blocks.items():
            assert dst.entry_mode is None, (
                f"cloned block {new_id} should NOT carry stale entry_mode "
                f"(got {dst.entry_mode})"
            )
            assert dst.exit_mode is None, (
                f"cloned block {new_id} should NOT carry stale exit_mode "
                f"(got {dst.exit_mode})"
            )

    def test_inliner_tracks_mutated_funcs(self):
        """FunctionInliner.run() must populate self.mutated_funcs with the
        names of every caller whose MIR was actually mutated by inlining.

        This is the load-bearing handle the codegen uses to know which
        functions need their mode metadata recomputed via
        mode_tracker.reanalyze_function(). If the inliner forgets to
        record a mutation, downstream passes see stale metadata.
        """
        callee = create_simple_callee()
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        assert inlined_count == 1
        assert "main" in inliner.mutated_funcs, (
            f"expected caller 'main' in mutated_funcs, got {inliner.mutated_funcs}"
        )
        # The callee was not itself a caller of anything inlinable.
        assert "add_one" not in inliner.mutated_funcs


class TestFunctionInliner:
    """Tests for FunctionInliner."""

    def test_simple_inline(self):
        """Test basic function inlining with #[inline]."""
        callee = create_simple_callee()
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        assert inlined_count == 1

        # After inlining, caller should have more blocks
        assert len(caller.blocks) > 1

    def test_recursion_not_inlined(self):
        """Test that recursive functions are not inlined."""
        recursive_func = create_recursive_function()
        caller = create_caller_with_call("factorial")
        program = MIRProgram(functions=[caller, recursive_func])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        # The recursive call should not be inlined
        # (caller's call to factorial might be inlined if factorial is small)
        # But the recursive call within factorial should not be
        has_recursive_call = False
        for block in recursive_func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Call) and instr.function == "factorial":
                    has_recursive_call = True
                    break

        assert has_recursive_call is True

    def test_bank_safe_far_function_inlined(self):
        """Bank-safe far functions ARE inlined under the relaxed rule
        (no memory access, no nested calls → body is bank-independent).
        """
        far_func = create_far_function()
        caller = create_caller_with_call("far_helper")
        program = MIRProgram(functions=[caller, far_func])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        assert inlined_count == 1

    def test_multiple_returns_inlined(self):
        """Test inlining a function with multiple return paths."""
        callee = create_function_with_multiple_returns()
        caller = create_caller_with_call("multi_return")
        program = MIRProgram(functions=[caller, callee])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        assert inlined_count == 1

        # All return paths should jump to the same merge block
        merge_block_ids = set()
        for block in caller.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Jump):
                    merge_block_ids.add(instr.target)

        # Should have jumps to merge block
        assert len(merge_block_ids) > 0

    def test_called_once_auto_inline(self):
        """Test that functions called exactly once are auto-inlined."""
        callee = create_simple_callee()
        callee.inline_attr = None  # No inline attribute
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        assert inlined_count == 1

    def test_large_function_not_inlined(self):
        """Test that large functions are not inlined even with #[inline]."""
        large_func = create_large_function(INLINE_COST_WITH_ATTR + 10)
        large_func.inline_attr = InlineAttribute(name='inline')

        # Create two callers so the function is called twice
        caller1 = create_caller_with_call("large_func")
        caller1.name = "caller1"
        caller1.is_entry = True

        caller2 = MIRFunction(
            name="caller2",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )
        vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
        caller2.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="large_func", args=[], returns=[vreg_result], is_far=False),
                Return(values=[vreg_result])
            ],
            predecessors=[],
            successors=[]
        )

        program = MIRProgram(functions=[caller1, caller2, large_func])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        assert inlined_count == 0

    def test_inline_always_small_function(self):
        """Small #[inline(always)] function should be inlined (same as
        plain #[inline] would be)."""
        callee = create_simple_callee()
        callee.inline_attr = InlineAttribute(name='inline', mode=InlineMode.ALWAYS)
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        assert checker.should_inline("add_one") is True

    def test_inline_always_bypasses_size_budget(self):
        """#[inline(always)] should inline regardless of size — that's
        the difference from plain #[inline], which is size-gated. A
        large body that plain #[inline] would refuse must still inline
        under ALWAYS, matching Rust/LLVM semantics where ALWAYS is a
        directive rather than a hint."""
        # Build a body whose cycle cost exceeds INLINE_COST_WITH_ATTR
        # by a comfortable margin.
        large_func = create_large_function(INLINE_COST_WITH_ATTR + 40)
        large_func.inline_attr = InlineAttribute(name='inline',
                                                 mode=InlineMode.ALWAYS)
        caller = create_caller_with_call("large_func")
        program = MIRProgram(functions=[caller, large_func])

        checker = InlinabilityChecker(program)
        # Sanity: cost actually exceeds the size-gated budget.
        assert checker._estimate_cycle_cost(large_func) > INLINE_COST_WITH_ATTR
        # ALWAYS should still inline it.
        assert checker.should_inline("large_func") is True

    def test_plain_inline_hint_respects_size_budget(self):
        """Counterpart to the ALWAYS bypass: bare #[inline] (HINT mode)
        must NOT inline a body whose cost exceeds INLINE_COST_WITH_ATTR.
        Pins the size-gating contract for the HINT mode."""
        large_func = create_large_function(INLINE_COST_WITH_ATTR + 40)
        # Bare #[inline] parses to HINT (size-gated).
        large_func.inline_attr = InlineAttribute(name='inline',
                                                 mode=InlineMode.HINT)
        caller = create_caller_with_call("large_func")

        # Add a second caller so the implicit "called-once" rule
        # doesn't accidentally save us when implicit inlining is on.
        caller2 = make_mir_function("caller2",
                                    return_type=BasicTypeInfo('u8'))
        vr = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="r")
        caller2.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="large_func", args=[], returns=[vr],
                     is_far=False),
                Return(values=[vr]),
            ],
            predecessors=[], successors=[],
        )
        program = MIRProgram(functions=[caller, caller2, large_func])

        checker = InlinabilityChecker(program, implicit_inline=False)
        assert checker.should_inline("large_func") is False

    def test_inline_never_not_inlined_even_when_called_once(self):
        """Test that #[inline(never)] prevents inlining even when called exactly once."""
        callee = create_simple_callee()
        callee.inline_attr = InlineAttribute(name='inline', mode=InlineMode.NEVER)
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        # Function is called exactly once, but #[inline(never)] should prevent inlining
        assert checker.should_inline("add_one") is False

    def test_inline_never_small_function_not_inlined(self):
        """Test that #[inline(never)] prevents inlining even for trivial functions."""
        getter = create_getter_function()
        getter.inline_attr = InlineAttribute(name='inline', mode=InlineMode.NEVER)
        caller = create_caller_with_call("get_value")
        program = MIRProgram(functions=[caller, getter])

        checker = InlinabilityChecker(program)
        # Even trivial getters should not be inlined with #[inline(never)]
        assert checker.should_inline("get_value") is False

    def test_inline_never_prevents_actual_inlining(self):
        """Test that #[inline(never)] actually prevents inlining in the FunctionInliner pass."""
        callee = create_simple_callee()
        callee.inline_attr = InlineAttribute(name='inline', mode=InlineMode.NEVER)
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        # No functions should be inlined
        assert inlined_count == 0
        # Both functions should still exist
        assert len(program.functions) == 2


# =============================================================================
# Boundary SetMode tests
# =============================================================================

def _make_callee_with_modes(name: str, entry_mode: ModeState, exit_mode: ModeState,
                            return_type: str = 'u8') -> MIRFunction:
    """Build a one-block callee with explicit entry/exit M modes.

    The body is a stand-in (Move + Return) — its content doesn't matter for
    boundary-SetMode shape tests; only the signature modes do.
    """
    func = make_mir_function(name, return_type=return_type, inline_attr=INLINE)
    func.entry_m_mode = entry_mode
    func.exit_m_mode = exit_mode
    vreg = VirtualRegister(id=0, type_info=BasicTypeInfo(return_type), hint="r")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg, source=Immediate(1), type_info=BasicTypeInfo(return_type)),
            Return(values=[vreg]),
        ],
        predecessors=[], successors=[],
    )
    # Pre-populate entry_mode so the inliner's "caller mode at call site"
    # computation finds a non-None starting point in the cloned body.
    func.blocks[0].entry_mode = ProcessorMode(entry_mode, XModeState.X16)
    func.blocks[0].exit_mode = ProcessorMode(exit_mode, XModeState.X16)
    return func


def _make_caller_with_call(callee_name: str, caller_entry_mode: ModeState,
                            return_type: str = 'u8') -> MIRFunction:
    """Build a single-block caller in the given M mode that calls `callee_name`.

    Mirrors create_caller_with_call but lets the test choose the caller's
    entry mode, and seeds block 0's entry_mode (the inliner relies on it).
    """
    func = make_mir_function("main", return_type=return_type, is_entry=True)
    func.entry_m_mode = caller_entry_mode
    func.exit_m_mode = caller_entry_mode
    vreg = VirtualRegister(id=0, type_info=BasicTypeInfo(return_type), hint="r")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Call(function=callee_name, args=[], returns=[vreg], is_far=False,
                 callee_entry_m_mode=None, callee_exit_m_mode=None),
            Return(values=[vreg]),
        ],
        predecessors=[], successors=[],
    )
    func.blocks[0].entry_mode = ProcessorMode(caller_entry_mode, XModeState.X16)
    func.blocks[0].exit_mode = ProcessorMode(caller_entry_mode, XModeState.X16)
    return func


def _setmode_count(blocks) -> int:
    return sum(
        1
        for b in blocks.values()
        for i in b.instructions
        if isinstance(i, SetMode)
    )


def _setmodes_in_block(block) -> list:
    return [i for i in block.instructions if isinstance(i, SetMode)]


class TestBoundarySetMode:
    """Inliner inserts MIR-level SetMode at the inline boundary.

    These tests assert MIR shape after `_inline_call` so they're cheap and
    isolated from the codegen / assembler / emulator. They cover the same
    behavior the e2e tests verify end-to-end, except for hardware-only
    semantics (B-register preservation across SEP/REP).
    """

    def _run_inline(self, caller, callee):
        program = MIRProgram(functions=[caller, callee])
        FunctionInliner(verbose=False).run(program)
        return caller

    def test_inserts_entry_setmode_for_m16_callee_in_m8_caller(self):
        """m8 caller → m16 callee: a REP #$20 must appear before the Jump
        into the inlined entry so the body runs in the mode it expects."""
        callee = _make_callee_with_modes("f", ModeState.M16, ModeState.M16, 'u16')
        caller = _make_caller_with_call("f", ModeState.M8, 'u16')
        self._run_inline(caller, callee)

        call_block = caller.blocks[0]
        # The original Call+Return are gone; the call block now ends with
        # [..., SetMode(REP #$20), Jump(inlined_entry)].
        assert isinstance(call_block.instructions[-1], Jump)
        entry_setmode = call_block.instructions[-2]
        assert isinstance(entry_setmode, SetMode)
        assert entry_setmode.mask == 0x20
        assert entry_setmode.is_set is False  # REP

    def test_inserts_exit_setmode_on_each_return_path(self):
        """m16 callee returning to m8 caller: every cloned return-block
        gets a SEP #$20 between its result Move and the Jump-to-merge so
        merge predecessors agree on mode."""
        callee = _make_callee_with_modes("f", ModeState.M16, ModeState.M16, 'u16')
        caller = _make_caller_with_call("f", ModeState.M8, 'u16')
        self._run_inline(caller, callee)

        # Every block that ends in Jump-to-merge (i.e., every former Return
        # block) should have a SetMode SEP just before that Jump.
        merge_block_id = max(caller.blocks.keys())
        return_blocks = [
            b for b in caller.blocks.values()
            if b.successors == [merge_block_id]
        ]
        assert return_blocks, "expected at least one return-rewritten block"
        for block in return_blocks:
            assert isinstance(block.instructions[-1], Jump)
            exit_setmode = block.instructions[-2]
            assert isinstance(exit_setmode, SetMode), (
                f"expected SetMode before Jump in return block, "
                f"got {type(exit_setmode).__name__}"
            )
            assert exit_setmode.mask == 0x20
            assert exit_setmode.is_set is True  # SEP back to m8

    def test_no_boundary_setmode_when_modes_match(self):
        """m8 caller + m8 callee: no entry or exit boundary SetMode.

        The pre-fix behavior emitted nothing here; the post-fix behavior
        should also emit nothing, since the explicit mode comparison
        short-circuits when caller and callee agree.
        """
        callee = _make_callee_with_modes("f", ModeState.M8, ModeState.M8, 'u8')
        caller = _make_caller_with_call("f", ModeState.M8, 'u8')
        self._run_inline(caller, callee)

        # The only SetMode the inliner could have inserted are at the
        # entry boundary (call_block tail) and exit boundary (return-block
        # tails). With matching modes there should be zero of either.
        assert _setmode_count(caller.blocks) == 0

    def test_inserts_entry_setmode_for_m8_callee_in_m16_caller(self):
        """Reverse direction: m16 caller → m8 callee. Entry boundary must
        be SEP #$20; exit boundary must be REP #$20 to restore the m16
        the caller's continuation expects."""
        callee = _make_callee_with_modes("f", ModeState.M8, ModeState.M8, 'u8')
        caller = _make_caller_with_call("f", ModeState.M16, 'u8')
        self._run_inline(caller, callee)

        call_block = caller.blocks[0]
        entry_setmode = call_block.instructions[-2]
        assert isinstance(entry_setmode, SetMode)
        assert entry_setmode.mask == 0x20
        assert entry_setmode.is_set is True  # SEP into m8

        merge_block_id = max(caller.blocks.keys())
        return_blocks = [
            b for b in caller.blocks.values()
            if b.successors == [merge_block_id]
        ]
        for block in return_blocks:
            exit_setmode = block.instructions[-2]
            assert isinstance(exit_setmode, SetMode)
            assert exit_setmode.mask == 0x20
            assert exit_setmode.is_set is False  # REP back to m16

    def test_no_exit_setmode_when_callee_exit_matches_caller(self):
        """m8 caller → m16-entry / m8-exit callee: the entry boundary needs
        a REP, but the exit boundary needs nothing because the callee
        returns in the same mode the caller expects."""
        callee = _make_callee_with_modes("f", ModeState.M16, ModeState.M8, 'u8')
        caller = _make_caller_with_call("f", ModeState.M8, 'u8')
        self._run_inline(caller, callee)

        # Entry boundary: REP present.
        call_block = caller.blocks[0]
        entry_setmode = call_block.instructions[-2]
        assert isinstance(entry_setmode, SetMode)
        assert entry_setmode.is_set is False  # REP into m16

        # Exit boundary: no SetMode in any return block (last instr is
        # the Jump itself, not preceded by a SetMode).
        merge_block_id = max(caller.blocks.keys())
        for block in caller.blocks.values():
            if block.successors != [merge_block_id]:
                continue
            assert isinstance(block.instructions[-1], Jump)
            # second-to-last must NOT be a SetMode that touches M
            penult = block.instructions[-2] if len(block.instructions) >= 2 else None
            if isinstance(penult, SetMode):
                assert not (penult.mask & 0x20), (
                    "no exit-boundary M-flag SetMode expected when "
                    "callee.exit_m_mode == caller's mode"
                )


# =============================================================================
# Hard-rejection tests: properties that make a callee non-inlinable regardless
# of size or call count. Each property exercises one of the can_inline()
# guards added to prevent silent metadata loss when the callee carries
# codegen-relevant state the inliner doesn't merge into the caller.
# =============================================================================

class TestCanInlineRejections:
    """can_inline() must refuse callees whose body or metadata the inliner
    can't safely splice into a caller."""

    def _checker_for(self, callee):
        caller = create_caller_with_call(callee.name)
        program = MIRProgram(functions=[caller, callee])
        return InlinabilityChecker(program), program

    def test_rejects_trait_method(self):
        """Trait methods receive self in Y with a pre-allocated self_y_vreg.
        Splicing them into a caller breaks the Y-self contract."""
        callee = create_simple_callee()
        callee.is_trait_method = True
        checker, _ = self._checker_for(callee)
        assert checker.can_inline(callee.name) is False

    def test_rejects_scratch_param_addrs(self):
        """Callees with promoted scratch params reserve DP slots tied to
        their signature; the caller's frame doesn't reflect those."""
        callee = create_simple_callee()
        callee.scratch_param_addrs = {0: 0x40}
        checker, _ = self._checker_for(callee)
        assert checker.can_inline(callee.name) is False

    def test_rejects_hw_param_regs(self):
        """Callees with hw-promoted params (FixedStack ABI) carry per-param
        register pre-allocations the caller doesn't see."""
        callee = create_simple_callee()
        callee.hw_param_regs = {0: 'X'}
        checker, _ = self._checker_for(callee)
        assert checker.can_inline(callee.name) is False

    def test_rejects_far_ptr_strategy(self):
        """far_ptr_strategy is per-function (D=S vs SET_DBR); inlining a
        callee under one strategy into a caller under the other breaks
        LoadIndirect/StoreIndirect lowering."""
        callee = create_simple_callee()
        callee.far_ptr_strategy = FarPtrStrategy.D_EQUALS_S
        checker, _ = self._checker_for(callee)
        assert checker.can_inline(callee.name) is False

    def test_rejects_preserves_attr(self):
        """#[preserves(...)] contract is enforced at the call boundary; the
        inliner removes the boundary without emitting equivalent
        SaveRegister/RestoreRegister, so this is unsound today."""
        callee = create_simple_callee()
        callee.preserves_attr = PreservesAttribute(name='preserves',
                                                   registers=['X'])
        checker, _ = self._checker_for(callee)
        assert checker.can_inline(callee.name) is False

    def test_bank_mismatch_blocks_inlining_at_call_site(self):
        """A #[bank(n)] callee called from a #[bank(m)] caller (m != n)
        must not be inlined — assembling the callee inside the caller's
        bank would change where its labels resolve. Verified via
        FunctionInliner.run() because bank compatibility is a property
        of the (caller, callee) pair, not the callee alone."""
        callee = create_simple_callee()
        callee.bank_attr = BankAttribute(name='bank', bank_number=2)
        caller = create_caller_with_call(callee.name)
        caller.bank_attr = BankAttribute(name='bank', bank_number=1)
        program = MIRProgram(functions=[caller, callee])
        inliner = FunctionInliner(verbose=False)
        assert inliner.run(program) == 0

    def test_same_bank_allows_inlining(self):
        """Sanity check the bank-compatibility logic: matching bank
        numbers should still inline."""
        callee = create_simple_callee()
        callee.bank_attr = BankAttribute(name='bank', bank_number=1)
        caller = create_caller_with_call(callee.name)
        caller.bank_attr = BankAttribute(name='bank', bank_number=1)
        program = MIRProgram(functions=[caller, callee])
        inliner = FunctionInliner(verbose=False)
        assert inliner.run(program) == 1


# =============================================================================
# Multi-return: callees with -> u8, u8 style signatures must plumb every
# Return.values[i] into the matching call.returns[i]. Pre-fix, only [0]
# was emitted, silently dropping secondary return values.
# =============================================================================

def _make_multi_return_callee(name: str = "two_returns"):
    """Callee whose Return has two values, one per signature slot.

    Models a `fn f() -> u8, u16` body that produces (a, b) and returns
    both. The MIR shape is: load two vregs from immediates, then
    `Return(values=[a, b])`.
    """
    func = make_mir_function(name, inline_attr=INLINE,
                             return_type=BasicTypeInfo('u8'))
    a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='a')
    b = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='b')
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=a, source=Immediate(7), type_info=BasicTypeInfo('u8')),
            Move(dest=b, source=Immediate(9), type_info=BasicTypeInfo('u8')),
            Return(values=[a, b]),
        ],
        predecessors=[], successors=[],
    )
    return func


def _make_multi_return_caller(callee_name: str):
    """Caller capturing both return values into separate caller-side vregs."""
    func = make_mir_function("main", is_entry=True,
                             return_type=BasicTypeInfo('u8'))
    ret_a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='ret_a')
    ret_b = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='ret_b')
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Call(function=callee_name, args=[],
                 returns=[ret_a, ret_b], is_far=False),
            Return(values=[ret_a]),
        ],
        predecessors=[], successors=[],
    )
    return func


class TestMultiReturnInlining:
    """The Return-to-Move rewrite must emit one Move per return slot, not
    only the primary slot."""

    def test_both_return_values_plumbed(self):
        callee = _make_multi_return_callee()
        caller = _make_multi_return_caller(callee.name)
        program = MIRProgram(functions=[caller, callee])

        inlined = FunctionInliner(verbose=False).run(program)
        assert inlined == 1

        # The inliner replaced the Return with Move(dst, src) pairs. Find
        # the Moves whose destinations are the caller's ret_a and ret_b.
        ret_a_id = caller.blocks[0].instructions[0]  # original placeholder
        # The original Call is gone — pull the caller-side return vreg
        # destinations directly from caller's vreg space.
        ret_a_dst = None
        ret_b_dst = None
        for block in caller.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Move) and isinstance(instr.dest, VirtualRegister):
                    if instr.dest.hint == 'ret_a':
                        ret_a_dst = instr
                    elif instr.dest.hint == 'ret_b':
                        ret_b_dst = instr

        assert ret_a_dst is not None, "primary return value not plumbed"
        assert ret_b_dst is not None, (
            "secondary return value not plumbed — multi-return broken"
        )

    def test_secondary_return_dropped_when_caller_ignores_it(self):
        """If the caller's Call has only one returns slot, only that one
        should be plumbed (the second source is discarded). Matches the
        Pascal semantics of `_ , x = f()` where the caller doesn't bind."""
        callee = _make_multi_return_callee()
        # Caller with only one returns slot
        caller = make_mir_function("main", is_entry=True,
                                   return_type=BasicTypeInfo('u8'))
        ret_a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'),
                                hint='ret_a')
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function=callee.name, args=[], returns=[ret_a],
                     is_far=False),
                Return(values=[ret_a]),
            ],
            predecessors=[], successors=[],
        )
        program = MIRProgram(functions=[caller, callee])
        inlined = FunctionInliner(verbose=False).run(program)
        assert inlined == 1

        # Exactly one Move into a "ret_*" destination should exist (the
        # secondary slot has no caller-side destination to plumb into).
        ret_moves = [
            instr for block in caller.blocks.values()
            for instr in block.instructions
            if isinstance(instr, Move)
            and isinstance(instr.dest, VirtualRegister)
            and (instr.dest.hint or '').startswith('ret_')
        ]
        assert len(ret_moves) == 1


# =============================================================================
# _clone_instruction: every MIR instruction type the inliner may encounter
# must be cloned with operand remapping. Pre-fix, all but ~13 fell through
# to deepcopy and silently preserved callee-side VirtualRegister objects.
# =============================================================================

class TestCloneInstructionCoverage:
    """Coverage tests for the explicit-construction _clone_instruction."""

    def _make_cloner(self):
        callee = create_simple_callee()
        caller = create_caller_with_call(callee.name)
        return BlockCloner(caller, callee), callee, caller

    def test_clone_bittest_remaps_vreg(self):
        cloner, _, _ = self._make_cloner()
        v = VirtualRegister(id=99, type_info=BasicTypeInfo('u8'), hint='x')
        instr = BitTest(value=v, test_bit=7, type_info=BasicTypeInfo('u8'))
        cloned = cloner._clone_instruction(instr)
        assert isinstance(cloned, BitTest)
        assert cloned is not instr
        assert isinstance(cloned.value, VirtualRegister)
        assert cloned.value is not v
        # The id should be reissued from the caller's allocator
        assert cloned.test_bit == 7

    def test_clone_rotate_remaps_dest_and_source(self):
        cloner, _, _ = self._make_cloner()
        src = VirtualRegister(id=10, type_info=BasicTypeInfo('u8'), hint='s')
        dst = VirtualRegister(id=11, type_info=BasicTypeInfo('u8'), hint='d')
        instr = Rotate(dest=dst, source=src, direction='left', count=1,
                       type_info=BasicTypeInfo('u8'))
        cloned = cloner._clone_instruction(instr)
        assert isinstance(cloned, Rotate)
        assert cloned.dest is not dst
        assert cloned.source is not src
        assert cloned.direction == 'left'
        assert cloned.count == 1

    def test_clone_saverestore_remaps_save_location(self):
        cloner, _, _ = self._make_cloner()
        slot = VirtualRegister(id=20, type_info=BasicTypeInfo('u8'),
                               hint='slot')
        save = SaveRegister(register=HardwareRegister('X'), save_location=slot)
        restore = RestoreRegister(register=HardwareRegister('X'),
                                  save_location=slot)
        cs = cloner._clone_instruction(save)
        cr = cloner._clone_instruction(restore)
        assert isinstance(cs, SaveRegister)
        assert isinstance(cr, RestoreRegister)
        assert cs.save_location is not slot
        # Same callee vreg referenced twice must remap to the same caller
        # vreg (cloner uses an id-keyed cache).
        assert cs.save_location is cr.save_location

    def test_clone_push_pull_preserve_register(self):
        cloner, _, _ = self._make_cloner()
        p = Push(register=HardwareRegister('A'))
        q = Pull(register=HardwareRegister('A'))
        cp = cloner._clone_instruction(p)
        cq = cloner._clone_instruction(q)
        assert isinstance(cp, Push)
        assert isinstance(cq, Pull)
        # HardwareRegister is a flyweight-style value, sharing is fine.
        assert cp.register.name == 'A'
        assert cq.register.name == 'A'

    def test_clone_call_remaps_args_and_returns(self):
        cloner, _, _ = self._make_cloner()
        arg_v = VirtualRegister(id=30, type_info=BasicTypeInfo('u8'), hint='a')
        ret_v = VirtualRegister(id=31, type_info=BasicTypeInfo('u8'), hint='r')
        instr = Call(
            function='other',
            args=[Argument(value=arg_v, mechanism=ArgumentMechanism.STACK,
                           param_type=BasicTypeInfo('u8'))],
            returns=[ret_v],
            is_far=False,
        )
        cloned = cloner._clone_instruction(instr)
        assert isinstance(cloned, Call)
        # Fresh argument object, not the original
        assert cloned.args[0] is not instr.args[0]
        assert cloned.args[0].value is not arg_v
        assert isinstance(cloned.args[0].value, VirtualRegister)
        assert cloned.returns[0] is not ret_v
        # Scalar fields preserved
        assert cloned.function == 'other'

    def test_clone_unknown_raises(self):
        """An unrecognized MIR instruction type must fail loudly rather
        than fall through to a deepcopy that aliases callee operands."""
        cloner, _, _ = self._make_cloner()

        class FakeInstr:
            source_loc = None

        with pytest.raises(NotImplementedError):
            cloner._clone_instruction(FakeInstr())

    def test_clone_inlineasm_raises(self):
        """InlineAsm callees are rejected by can_inline; reaching _clone
        with one means a bypass somewhere — fail loudly."""
        cloner, _, _ = self._make_cloner()
        with pytest.raises(NotImplementedError):
            cloner._clone_instruction(InlineAsm(instructions=["NOP"]))


# =============================================================================
# Worklist / topological ordering: nested inlining (A→B→C) should collapse
# in a single run when C is small enough to be inlined into B, and B into
# A. Pre-refactor this only worked via the fixed-point outer loop.
# =============================================================================

class TestNestedInlining:
    """Verify reverse-topological processing handles A→B→C in one pass."""

    def test_three_level_chain_inlines_fully(self):
        # Leaf: trivial function returning a constant.
        leaf = make_mir_function("leaf", inline_attr=INLINE)
        v0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='r')
        leaf.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Move(dest=v0, source=Immediate(1), type_info=BasicTypeInfo('u8')),
                Return(values=[v0]),
            ],
            predecessors=[], successors=[],
        )

        # Middle: calls leaf, returns its result.
        middle = make_mir_function("middle", inline_attr=INLINE)
        m0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='m')
        middle.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function='leaf', args=[], returns=[m0], is_far=False),
                Return(values=[m0]),
            ],
            predecessors=[], successors=[],
        )

        # Top: calls middle.
        top = make_mir_function("main", is_entry=True)
        t0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='t')
        top.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function='middle', args=[], returns=[t0], is_far=False),
                Return(values=[t0]),
            ],
            predecessors=[], successors=[],
        )

        program = MIRProgram(functions=[top, middle, leaf])
        inlined = FunctionInliner(verbose=False).run(program)

        # Two call sites: leaf into middle, middle into top.
        assert inlined == 2
        # Top should have no remaining Call instructions.
        remaining_calls = [
            i for b in top.blocks.values() for i in b.instructions
            if isinstance(i, Call)
        ]
        assert remaining_calls == []


# =============================================================================
# Register-parameter setup ordering: when a callee has multiple register
# parameters, any A-target Move must come LAST. Non-A loads (to B via XBA,
# DBR via PHA/PLB, D via TCD) route through A in codegen and would clobber
# the A param before the body reads it.
# =============================================================================

def _make_two_reg_param_callee(reg_a: str, reg_b: str):
    """Callee with two register-bound parameters."""
    func = make_mir_function("twoparams", inline_attr=INLINE)
    p0 = HIRParameter(name='p0', param_type=BasicTypeInfo('u8'),
                      binding=RegisterBinding(register_name=reg_a))
    p1 = HIRParameter(name='p1', param_type=BasicTypeInfo('u8'),
                      binding=RegisterBinding(register_name=reg_b))
    func.parameters = [p0, p1]
    v0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='saved0')
    v1 = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='saved1')
    # The MIR builder emits one save Move per register-bound param at the
    # top of the entry block. The inliner uses these to detect which HW
    # registers the body observes; without them the param loads are
    # treated as dead and pruned.
    #
    # Both params are mutated (p += 1) so the inliner's copy-propagation
    # (operand substitution) does NOT elide the register loads: a written
    # param vreg is not a pure copy of the caller's arg, so the binding load
    # is still emitted. This keeps the A-load-last ordering path — the whole
    # point of these tests — exercised.
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=v0, source=HardwareRegister(reg_a),
                 type_info=BasicTypeInfo('u8')),
            Move(dest=v1, source=HardwareRegister(reg_b),
                 type_info=BasicTypeInfo('u8')),
            BinaryOp(dest=v0, left=v0, right=Immediate(1), op='+',
                     type_info=BasicTypeInfo('u8')),
            BinaryOp(dest=v1, left=v1, right=Immediate(1), op='+',
                     type_info=BasicTypeInfo('u8')),
            Return(values=[v0]),
        ],
        predecessors=[], successors=[],
    )
    return func


class TestRegisterParamOrdering:
    """Item 1 from the review punch-list: register-param load ordering."""

    def test_a_load_emitted_last(self):
        """Callee `fn f(a @ A, b @ B)` called with two vreg sources. The
        inserted register setup must end with a Move-to-A (so B's load —
        which goes through A via XBA in codegen — doesn't clobber the
        A param value)."""
        callee = _make_two_reg_param_callee('A', 'B')
        caller = make_mir_function("main", is_entry=True,
                                   return_type=BasicTypeInfo('u8'))
        arg_a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'),
                                hint='arg_a')
        arg_b = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'),
                                hint='arg_b')
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function=callee.name, args=[
                    Argument(value=arg_a, mechanism=ArgumentMechanism.REGISTER,
                             location=HardwareRegister('A'),
                             param_type=BasicTypeInfo('u8')),
                    Argument(value=arg_b, mechanism=ArgumentMechanism.REGISTER,
                             location=HardwareRegister('B'),
                             param_type=BasicTypeInfo('u8')),
                ], returns=[], is_far=False),
                Return(values=[]),
            ],
            predecessors=[], successors=[],
        )
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        # Register loads are inserted at the head of the inlined entry
        # block (see _inline_call's `inlined_entry.instructions = ...`).
        # Scan every block in the caller for Move-to-HardwareRegister
        # instructions whose source is one of the call args; collect them
        # in instruction order to verify the A-load comes last.
        reg_loads = [
            i for b in caller.blocks.values() for i in b.instructions
            if isinstance(i, Move) and isinstance(i.dest, HardwareRegister)
        ]
        assert len(reg_loads) == 2, (
            f"expected 2 register loads, got {len(reg_loads)}: {reg_loads}"
        )
        # Last register load must target A
        assert reg_loads[-1].dest.name == 'A', (
            f"A-load should be emitted last; got order: "
            f"{[m.dest.name for m in reg_loads]}"
        )

    def test_a_first_in_callee_signature_still_emitted_last(self):
        """Order in the callee signature shouldn't matter. `fn f(a @ A, x @ X)`
        with the A param FIRST in the signature must still emit A LAST."""
        callee = _make_two_reg_param_callee('A', 'X')
        caller = make_mir_function("main", is_entry=True,
                                   return_type=BasicTypeInfo('u8'))
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function=callee.name, args=[
                    Argument(value=Immediate(5),
                             mechanism=ArgumentMechanism.REGISTER,
                             location=HardwareRegister('A'),
                             param_type=BasicTypeInfo('u8')),
                    Argument(value=Immediate(7),
                             mechanism=ArgumentMechanism.REGISTER,
                             location=HardwareRegister('X'),
                             param_type=BasicTypeInfo('u8')),
                ], returns=[], is_far=False),
                Return(values=[]),
            ],
            predecessors=[], successors=[],
        )
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        reg_loads = [
            i for b in caller.blocks.values() for i in b.instructions
            if isinstance(i, Move) and isinstance(i.dest, HardwareRegister)
        ]
        assert reg_loads[-1].dest.name == 'A'


# =============================================================================
# Parameter substitution (copy-propagation at inline time): the inliner
# splices the caller's arg operand directly into the body instead of emitting
# a bridge Move, when it is safe to do so. These tests exercise conditions
# 1-4 from _compute_param_substitution.
# =============================================================================

def _all_instrs(func):
    return [i for b in func.blocks.values() for i in b.instructions]


def _make_reg_param_callee(body_instrs, reg='A'):
    """Callee with one register-bound param whose entry save Move is v0."""
    func = make_mir_function("callee", inline_attr=INLINE)
    func.parameters = [HIRParameter(name='v', param_type=BasicTypeInfo('u8'),
                                    binding=RegisterBinding(register_name=reg))]
    v0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='saved')
    save = Move(dest=v0, source=HardwareRegister(reg), type_info=BasicTypeInfo('u8'))
    func.blocks[0] = BasicBlock(block_id=0, instructions=[save] + body_instrs,
                               predecessors=[], successors=[])
    return func, v0


def _caller_calling(callee_name, arg_value, mechanism, *, ret_id=90):
    caller = make_mir_function("main", is_entry=True)
    ret = VirtualRegister(id=ret_id, type_info=BasicTypeInfo('u8'), hint='ret')
    caller.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Call(function=callee_name,
                 args=[Argument(value=arg_value, mechanism=mechanism,
                                location=None, param_type=BasicTypeInfo('u8'))],
                 returns=[ret], is_far=False),
            Return(values=[ret]),
        ],
        predecessors=[], successors=[],
    )
    return caller


class TestParamSubstitution:
    def test_stack_param_vreg_substituted(self):
        """f(v){ res = v + v } called with a vreg arg: the arg vreg is spliced
        into the body and no bridge Move copying it into the param vreg remains."""
        # Build explicitly so left/right reference the param vreg object.
        pv = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='v')
        res = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='res')
        callee = make_mir_function("dbl", inline_attr=INLINE)
        callee.parameters = [HIRParameter(name='v', param_type=BasicTypeInfo('u8'),
                                          binding=None)]
        callee.param_to_vreg = {0: pv}
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                BinaryOp(dest=res, left=pv, right=pv, op='+',
                         type_info=BasicTypeInfo('u8')),
                Return(values=[res]),
            ],
            predecessors=[], successors=[],
        )
        arg = VirtualRegister(id=42, type_info=BasicTypeInfo('u8'), hint='arg')
        caller = _caller_calling("dbl", arg, ArgumentMechanism.STACK)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        binops = [i for i in _all_instrs(caller) if isinstance(i, BinaryOp)]
        assert any(isinstance(bo.left, VirtualRegister) and bo.left.id == arg.id
                   for bo in binops), "arg vreg should be spliced into the body"
        # No Move copies the arg into a fresh vreg (bridge eliminated).
        bridges = [i for i in _all_instrs(caller)
                   if isinstance(i, Move) and isinstance(i.dest, VirtualRegister)
                   and isinstance(i.source, VirtualRegister) and i.source.id == arg.id]
        assert bridges == [], f"unexpected bridge Move(s): {bridges}"

    def test_register_param_vreg_substituted(self):
        """f(x @ A){ res = x + 1 } called with a vreg: no A-load and no entry
        save Move survive — the arg vreg is used directly."""
        res = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='res')
        callee, v0 = _make_reg_param_callee([
            BinaryOp(dest=res, left=None, right=Immediate(1), op='+',
                     type_info=BasicTypeInfo('u8')),
            Return(values=[res]),
        ])
        # Fix up the body's left operand to reference the saved param vreg.
        callee.blocks[0].instructions[1].left = v0
        arg = VirtualRegister(id=42, type_info=BasicTypeInfo('u8'), hint='arg')
        caller = _caller_calling("callee", arg, ArgumentMechanism.REGISTER)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        instrs = _all_instrs(caller)
        # No register LOAD (Move into A) — the arg is used in place instead.
        assert not any(isinstance(i, Move) and isinstance(i.dest, HardwareRegister)
                       for i in instrs), "no register load should remain"
        # The arg vreg is spliced directly into the body ALU op (proving the
        # entry save Move -> param-vreg chain was replaced by copy-propagation).
        binops = [i for i in instrs if isinstance(i, BinaryOp)]
        assert any(isinstance(bo.left, VirtualRegister) and bo.left.id == arg.id
                   for bo in binops)

    def test_reassigned_param_not_substituted(self):
        """f(v){ v = v + 1; return v } — the param is written, so substitution
        is refused and the bridge Move is retained."""
        pv = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='v')
        callee = make_mir_function("inc", inline_attr=INLINE)
        callee.parameters = [HIRParameter(name='v', param_type=BasicTypeInfo('u8'),
                                          binding=None)]
        callee.param_to_vreg = {0: pv}
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                BinaryOp(dest=pv, left=pv, right=Immediate(1), op='+',
                         type_info=BasicTypeInfo('u8')),   # writes pv
                Return(values=[pv]),
            ],
            predecessors=[], successors=[],
        )
        arg = VirtualRegister(id=42, type_info=BasicTypeInfo('u8'), hint='arg')
        caller = _caller_calling("inc", arg, ArgumentMechanism.STACK)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        bridges = [i for i in _all_instrs(caller)
                   if isinstance(i, Move) and isinstance(i.dest, VirtualRegister)
                   and isinstance(i.source, VirtualRegister) and i.source.id == arg.id]
        assert len(bridges) == 1, "written param must keep its bridge Move"

    def test_direct_hw_read_keeps_register_load(self):
        """f(x @ A){ y = A; ... } — the body reads A directly, so the A-load
        must be kept (substitution refused)."""
        res = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='res')
        callee, v0 = _make_reg_param_callee([
            Move(dest=res, source=HardwareRegister('A'),
                 type_info=BasicTypeInfo('u8')),   # direct HW read of A
            Return(values=[res]),
        ])
        arg = VirtualRegister(id=42, type_info=BasicTypeInfo('u8'), hint='arg')
        caller = _caller_calling("callee", arg, ArgumentMechanism.REGISTER)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        a_loads = [i for i in _all_instrs(caller)
                   if isinstance(i, Move) and isinstance(i.dest, HardwareRegister)
                   and i.dest.name == 'A']
        assert len(a_loads) == 1, "A-load must be kept when body reads A directly"

    def test_immediate_into_pointer_not_substituted(self):
        """f(p){ *p = 1 } called with a constant pointer: an Immediate must NOT
        land in StoreIndirect.pointer, so the bridge Move is retained."""
        pv = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='p')
        callee = make_mir_function("wr", inline_attr=INLINE)
        callee.parameters = [HIRParameter(name='p', param_type=BasicTypeInfo('u8'),
                                          binding=None)]
        callee.param_to_vreg = {0: pv}
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                StoreIndirect(source=Immediate(1), pointer=pv, is_far=False,
                              type_info=BasicTypeInfo('u8')),
                Return(values=[]),
            ],
            predecessors=[], successors=[],
        )
        caller = _caller_calling("wr", Immediate(0x2000), ArgumentMechanism.STACK)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        stores = [i for i in _all_instrs(caller) if isinstance(i, StoreIndirect)]
        assert stores, "expected the inlined StoreIndirect"
        assert all(isinstance(s.pointer, VirtualRegister) for s in stores), \
            "pointer must stay a vreg, not become an Immediate"
        bridges = [i for i in _all_instrs(caller)
                   if isinstance(i, Move) and isinstance(i.dest, VirtualRegister)
                   and isinstance(i.source, Immediate) and i.source.value == 0x2000]
        assert len(bridges) == 1, "immediate pointer must keep its bridge Move"

    def test_immediate_into_alu_substituted(self):
        """f(v){ res = v + 1 } called with an Immediate: the constant lands in
        BinaryOp.left (immediate-legal), so it is substituted and no bridge remains."""
        pv = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='v')
        res = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='res')
        callee = make_mir_function("addone", inline_attr=INLINE)
        callee.parameters = [HIRParameter(name='v', param_type=BasicTypeInfo('u8'),
                                          binding=None)]
        callee.param_to_vreg = {0: pv}
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                BinaryOp(dest=res, left=pv, right=Immediate(1), op='+',
                         type_info=BasicTypeInfo('u8')),
                Return(values=[res]),
            ],
            predecessors=[], successors=[],
        )
        caller = _caller_calling("addone", Immediate(9), ArgumentMechanism.STACK)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        binops = [i for i in _all_instrs(caller) if isinstance(i, BinaryOp)]
        assert any(isinstance(bo.left, Immediate) and bo.left.value == 9
                   for bo in binops), "immediate arg should be spliced into ALU op"
        bridges = [i for i in _all_instrs(caller)
                   if isinstance(i, Move) and isinstance(i.dest, VirtualRegister)
                   and isinstance(i.source, Immediate) and i.source.value == 9]
        assert bridges == [], "no bridge Move for the substituted immediate"

    def test_width_mismatch_not_substituted(self):
        """u8 vreg arg into a u16 param needs zero-extension, so substitution is
        refused (the TypeConvert+Move widening path is kept)."""
        pv = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint='v')
        res = VirtualRegister(id=1, type_info=BasicTypeInfo('u16'), hint='res')
        callee = make_mir_function("wide", inline_attr=INLINE)
        callee.parameters = [HIRParameter(name='v', param_type=BasicTypeInfo('u16'),
                                          binding=None)]
        callee.param_to_vreg = {0: pv}
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                BinaryOp(dest=res, left=pv, right=Immediate(1), op='+',
                         type_info=BasicTypeInfo('u16')),
                Return(values=[res]),
            ],
            predecessors=[], successors=[],
        )
        arg = VirtualRegister(id=42, type_info=BasicTypeInfo('u8'), hint='narrow')
        caller = _caller_calling("wide", arg, ArgumentMechanism.STACK)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        binops = [i for i in _all_instrs(caller) if isinstance(i, BinaryOp)]
        # The narrow arg must NOT be spliced directly into the u16 op.
        assert not any(isinstance(bo.left, VirtualRegister) and bo.left.id == arg.id
                       for bo in binops), "narrow arg must not be substituted into wide param"

    def test_symbolic_immediate_not_substituted(self):
        """An Immediate carrying a `.symbol` (e.g. &ARR[2]) stores a byte OFFSET
        in .value that codegen combines with the symbol base — not a
        self-contained constant. It must keep its bridge Move, never be spliced
        directly into an operand slot."""
        pv = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='v')
        res = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='res')
        callee = make_mir_function("addoff", inline_attr=INLINE)
        callee.parameters = [HIRParameter(name='v', param_type=BasicTypeInfo('u8'),
                                          binding=None)]
        callee.param_to_vreg = {0: pv}
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                BinaryOp(dest=res, left=pv, right=Immediate(1), op='+',
                         type_info=BasicTypeInfo('u8')),
                Return(values=[res]),
            ],
            predecessors=[], successors=[],
        )
        sym_imm = Immediate(2)
        sym_imm.symbol = MockSymbol("ARR")   # symbolic immediate (address+offset)
        caller = _caller_calling("addoff", sym_imm, ArgumentMechanism.STACK)
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        binops = [i for i in _all_instrs(caller) if isinstance(i, BinaryOp)]
        assert not any(isinstance(bo.left, Immediate) and getattr(bo.left, 'symbol', None)
                       for bo in binops), "symbolic immediate must not be spliced into ALU op"
        bridges = [i for i in _all_instrs(caller)
                   if isinstance(i, Move) and isinstance(i.dest, VirtualRegister)
                   and isinstance(i.source, Immediate) and getattr(i.source, 'symbol', None)]
        assert len(bridges) == 1, "symbolic immediate must keep its bridge Move"


# =============================================================================
# exit_block_ids consistency: after inlining, the original call_block is
# no longer a function exit (it ends with Jump into the inlined entry),
# while the merge block — which holds the original post-call instructions
# including the function's terminating Return — IS the new exit.
# =============================================================================

class TestExitBlockIdsUpdate:
    """Item 2 from the review punch-list."""

    def test_original_exit_block_removed_merge_added(self):
        """Caller has one block, marked exit, ending in Call + Return.
        After inlining, that block is no longer an exit and the new
        merge block (containing Return) takes its place."""
        callee = create_simple_callee()
        caller = create_caller_with_call(callee.name)
        caller.exit_block_ids = [0]  # caller's block 0 is the function exit
        program = MIRProgram(functions=[caller, callee])

        assert FunctionInliner(verbose=False).run(program) == 1

        # Old block id (0) must no longer be in exit list.
        assert 0 not in caller.exit_block_ids, (
            "original call_block should be removed from exit_block_ids "
            "after inlining — it now ends with a Jump, not a Return"
        )
        # Exactly one exit, and it must end with a Return.
        assert len(caller.exit_block_ids) == 1
        new_exit = caller.blocks[caller.exit_block_ids[0]]
        assert isinstance(new_exit.instructions[-1], Return), (
            f"new exit block must end in Return, got "
            f"{type(new_exit.instructions[-1]).__name__}"
        )

    def test_non_exit_call_block_doesnt_get_added(self):
        """If the original call_block was NOT an exit (e.g., the call
        was mid-function with control continuing), inlining must not
        spuriously add the merge block as an exit — it should keep
        whatever successors the original call_block had."""
        # Build a caller whose call_block has a successor (not an exit).
        caller = make_mir_function("main", is_entry=True,
                                   return_type=BasicTypeInfo('u8'))
        vr = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'),
                             hint='r')
        # Block 0: Call then Jump to block 1; Block 1: Return.
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function='add_one', args=[], returns=[vr],
                     is_far=False),
                Jump(target=1),
            ],
            predecessors=[], successors=[1],
        )
        caller.blocks[1] = BasicBlock(
            block_id=1,
            instructions=[Return(values=[vr])],
            predecessors=[0], successors=[],
        )
        caller.exit_block_ids = [1]

        callee = create_simple_callee()
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        # Block 1 is still the function exit; the merge block is NOT an
        # exit (it ends with a Jump, not a Return).
        assert 1 in caller.exit_block_ids
        for eid in caller.exit_block_ids:
            assert isinstance(
                caller.blocks[eid].instructions[-1], Return
            ), "every exit block id must point to a Return-terminated block"


# =============================================================================
# A-substitution gating: when the exit boundary needs a SEP/REP, the
# HardwareRegister('A') substitution on Return must be disabled to avoid
# mode/width interactions in the substituted Move's codegen.
# =============================================================================

class TestASubstitutionGating:
    """Item 3 from the review punch-list."""

    def test_a_substitution_disabled_on_mode_boundary(self):
        """m16 callee → m8 caller forces a SEP at exit. The return-value
        Move in the cloned return block must NOT use HardwareRegister('A')
        as source (the mode flip can re-interpret A's width).
        """
        # Callee: u16 BinaryOp result → Return. Producer's dest IS the
        # returned vreg, so the substitution WOULD fire if not gated.
        callee = make_mir_function("u16_add", inline_attr=INLINE,
                                   return_type=BasicTypeInfo('u16'))
        callee.entry_m_mode = ModeState.M16
        callee.exit_m_mode = ModeState.M16
        v = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint='r')
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                BinaryOp(dest=v, left=Immediate(1), right=Immediate(2),
                         op='+', type_info=BasicTypeInfo('u16')),
                Return(values=[v]),
            ],
            predecessors=[], successors=[],
        )
        callee.blocks[0].entry_mode = ProcessorMode(ModeState.M16,
                                                   XModeState.X16)
        callee.blocks[0].exit_mode = ProcessorMode(ModeState.M16,
                                                  XModeState.X16)

        caller = make_mir_function("main", is_entry=True,
                                   return_type=BasicTypeInfo('u16'))
        caller.entry_m_mode = ModeState.M8
        caller.exit_m_mode = ModeState.M8
        ret = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'),
                              hint='ret')
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function='u16_add', args=[], returns=[ret],
                     is_far=False),
                Return(values=[ret]),
            ],
            predecessors=[], successors=[],
        )
        caller.blocks[0].entry_mode = ProcessorMode(ModeState.M8,
                                                   XModeState.X16)
        caller.blocks[0].exit_mode = ProcessorMode(ModeState.M8,
                                                   XModeState.X16)

        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        # Find the return-block Move (the one whose dest is `ret`). Its
        # source must be a VirtualRegister, NOT HardwareRegister('A').
        ret_moves = [
            instr for block in caller.blocks.values()
            for instr in block.instructions
            if isinstance(instr, Move)
            and isinstance(instr.dest, VirtualRegister)
            and (instr.dest.hint or '').startswith('ret')
        ]
        assert ret_moves, "expected at least one Move into the ret vreg"
        for m in ret_moves:
            assert not isinstance(m.source, HardwareRegister), (
                f"A-substitution should be DISABLED when exit-boundary "
                f"SetMode is pending, but got Move source = {m.source}"
            )

    def test_a_substitution_active_when_modes_match(self):
        """Sanity check: when caller and callee modes agree (no exit
        SetMode), the A-substitution optimization should still fire so
        we don't regress the common path."""
        callee = make_mir_function("u8_add", inline_attr=INLINE,
                                   return_type=BasicTypeInfo('u8'))
        callee.entry_m_mode = ModeState.M8
        callee.exit_m_mode = ModeState.M8
        v = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='r')
        callee.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                BinaryOp(dest=v, left=Immediate(1), right=Immediate(2),
                         op='+', type_info=BasicTypeInfo('u8')),
                Return(values=[v]),
            ],
            predecessors=[], successors=[],
        )
        callee.blocks[0].entry_mode = ProcessorMode(ModeState.M8,
                                                   XModeState.X16)
        callee.blocks[0].exit_mode = ProcessorMode(ModeState.M8,
                                                  XModeState.X16)

        caller = make_mir_function("main", is_entry=True,
                                   return_type=BasicTypeInfo('u8'))
        caller.entry_m_mode = ModeState.M8
        caller.exit_m_mode = ModeState.M8
        ret = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'),
                              hint='ret')
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function='u8_add', args=[], returns=[ret],
                     is_far=False),
                Return(values=[ret]),
            ],
            predecessors=[], successors=[],
        )
        caller.blocks[0].entry_mode = ProcessorMode(ModeState.M8,
                                                   XModeState.X16)
        caller.blocks[0].exit_mode = ProcessorMode(ModeState.M8,
                                                   XModeState.X16)

        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1

        # In the same-mode path the substitution should fire — at least
        # one Move into the ret vreg should have HardwareRegister('A')
        # as its source.
        ret_moves = [
            instr for block in caller.blocks.values()
            for instr in block.instructions
            if isinstance(instr, Move)
            and isinstance(instr.dest, VirtualRegister)
            and (instr.dest.hint or '').startswith('ret')
        ]
        has_a_sub = any(
            isinstance(m.source, HardwareRegister) and m.source.name == 'A'
            for m in ret_moves
        )
        assert has_a_sub, (
            "A-substitution should fire in the same-mode path so we "
            "don't regress the common case"
        )


# =============================================================================
# Far-function inlining: the blanket "no far fns" rule was over-conservative.
# A far fn whose body produces identical bytes in any bank — only far indirects,
# zero-page, or WRAM long-addressing — can be inlined into a different-bank
# caller without further codegen work. Bank-dependent bodies are still rejected.
# =============================================================================

def _make_far_callee(name: str, body_instrs):
    """Build a one-block far function with the given body."""
    func = make_mir_function(name, is_far=True, inline_attr=INLINE)
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=body_instrs,
        predecessors=[], successors=[],
    )
    return func


class TestFarFunctionInlining:
    """Tests for the relaxed far-fn rule."""

    def test_far_with_wram_long_store_inlinable(self):
        """The motivating case: a far fn that writes a #[ram] array at
        $7E2400 (the put_str shape — long-addressable, bank-safe)."""
        v = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='v')
        nametable = MemoryLocation(
            storage_type='ram',
            address=0x7E2400,
            symbol=MockSymbol('nametable1'),
        )
        callee = _make_far_callee("write_tile", [
            Move(dest=v, source=Immediate(0x21),
                 type_info=BasicTypeInfo('u8')),
            Store(source=v, dest=nametable,
                  type_info=BasicTypeInfo('u8')),
            Return(values=[]),
        ])
        checker = InlinabilityChecker(
            MIRProgram(functions=[callee]),
            implicit_inline=True,
        )
        assert checker.can_inline("write_tile") is True

    def test_far_with_far_indirect_load_inlinable(self):
        """LoadIndirect/StoreIndirect with is_far=True is bank-safe."""
        ptr = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint='p')
        out = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='r')
        callee = _make_far_callee("read_via_far_ptr", [
            LoadIndirect(dest=out, pointer=ptr, is_far=True,
                         index_register='Y', type_info=BasicTypeInfo('u8')),
            Return(values=[out]),
        ])
        checker = InlinabilityChecker(MIRProgram(functions=[callee]))
        assert checker.can_inline("read_via_far_ptr") is True

    def test_far_with_near_indirect_rejected(self):
        """Near indirect (`(zp),Y`) uses DBR; cross-bank inlining
        would access whatever DBR happens to be — unsafe."""
        ptr = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint='p')
        out = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint='r')
        callee = _make_far_callee("read_via_near_ptr", [
            LoadIndirect(dest=out, pointer=ptr, is_far=False,
                         index_register='Y', type_info=BasicTypeInfo('u8')),
            Return(values=[out]),
        ])
        checker = InlinabilityChecker(MIRProgram(functions=[callee]))
        assert checker.can_inline("read_via_near_ptr") is False

    def test_far_with_rom_load_rejected(self):
        """ROM data Load may be lowered as 3-byte absolute (DBR-relative);
        cross-bank inlining could read the wrong bank."""
        out = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='r')
        rom_loc = MemoryLocation(
            storage_type='rom',
            address=0x8000,  # in some bank, < $10000 so codegen may use abs
            symbol=MockSymbol('lookup_table'),
        )
        callee = _make_far_callee("read_rom", [
            Load(dest=out, source=rom_loc, type_info=BasicTypeInfo('u8')),
            Return(values=[out]),
        ])
        checker = InlinabilityChecker(MIRProgram(functions=[callee]))
        assert checker.can_inline("read_rom") is False

    def test_far_with_nested_near_call_rejected(self):
        """A near JSR in the body targets current PBR; after inlining
        into a different-bank caller, PBR is the caller's, so the JSR
        lands in the wrong bank."""
        callee = _make_far_callee("with_near_call", [
            Call(function='other_near_fn', args=[], returns=[],
                 is_far=False),
            Return(values=[]),
        ])
        checker = InlinabilityChecker(MIRProgram(functions=[callee]))
        assert checker.can_inline("with_near_call") is False

    def test_far_with_nested_far_call_allowed(self):
        """A nested JSL is bank-explicit (24-bit target), so it stays
        correct when inlined into any caller."""
        callee = _make_far_callee("with_far_call", [
            Call(function='other_far_fn', args=[], returns=[],
                 is_far=True),
            Return(values=[]),
        ])
        checker = InlinabilityChecker(MIRProgram(functions=[callee]))
        assert checker.can_inline("with_far_call") is True

    def test_far_with_databank_inline_rejected(self):
        """`#[mode(databank=inline)]` relies on prologue PHB/PLB that
        the inliner doesn't emit; the body may use absolute addressing
        inside that bracket. Refuse until we model DBR management at
        the inline boundary."""
        callee = _make_far_callee("with_dbr", [Return(values=[])])
        callee.mode_attr = ModeAttribute(name='mode',
                                         databank=DataBankMode.INLINE)
        checker = InlinabilityChecker(MIRProgram(functions=[callee]))
        assert checker.can_inline("with_dbr") is False

    def test_far_with_far_ptr_stack_param_rejected(self):
        """Far fns with far-pointer stack params rely on their prologue
        to set DBR = ptr bank (via PHB/PLA/PLB or PHD/TSC/TCD), and the
        body's `(d,S),Y` / `[dp],Y` indirect derefs depend on that
        setup. Inlining removes the prologue but keeps the body —
        miscompile risk. Reject until we emit equivalent DBR
        management at the inline boundary. (This is what blocks put_str
        from being inlined today.)"""
        callee = _make_far_callee("has_far_ptr_arg", [Return(values=[])])
        callee.has_far_ptr_stack_params = True
        checker = InlinabilityChecker(MIRProgram(functions=[callee]))
        assert checker.can_inline("has_far_ptr_arg") is False

    def test_far_inlining_crosses_bank_attr(self):
        """End-to-end: caller in bank 7 inlines a far callee defined in
        bank 0. Verifies _bank_compatible doesn't block cross-bank
        inlining for far callees the way it does for near callees."""
        v = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint='r')
        callee = _make_far_callee("bank0_helper", [
            Move(dest=v, source=Immediate(7),
                 type_info=BasicTypeInfo('u8')),
            Return(values=[v]),
        ])
        callee.bank_attr = BankAttribute(name='bank', bank_number=0)

        caller_ret = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'),
                                     hint='out')
        caller = make_mir_function("main", is_entry=True,
                                   return_type=BasicTypeInfo('u8'))
        caller.bank_attr = BankAttribute(name='bank', bank_number=7)
        caller.blocks[0] = BasicBlock(
            block_id=0,
            instructions=[
                Call(function='bank0_helper', args=[],
                     returns=[caller_ret], is_far=True),
                Return(values=[caller_ret]),
            ],
            predecessors=[], successors=[],
        )
        program = MIRProgram(functions=[caller, callee])
        assert FunctionInliner(verbose=False).run(program) == 1
        # No Call should remain in the caller.
        remaining = [
            i for b in caller.blocks.values() for i in b.instructions
            if isinstance(i, Call)
        ]
        assert remaining == []
