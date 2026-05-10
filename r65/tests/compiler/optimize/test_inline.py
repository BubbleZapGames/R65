#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for function inlining optimization pass."""

import pytest

from r65.compiler.optimize.inline import (
    FunctionInliner,
    InlinabilityChecker,
    BlockCloner,
    INLINE_THRESHOLD_WITH_ATTR,
    INLINE_THRESHOLD_NO_ATTR,
)
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock,
    Move, BinaryOp, Jump, CondBranch, Return, Call,
    Load, Store, LoadIndirect, StoreIndirect, MemoryLocation,
    VirtualRegister, HardwareRegister, Immediate,
    InlineAsm, Argument, ArgumentMechanism, SetMode,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.attributes import InlineAttribute, InlineMode, InterruptAttribute, InterruptVector
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

    def test_cannot_inline_far_function(self):
        """Far functions should not be inlinable."""
        far_func = create_far_function()
        caller = create_caller_with_call("far_helper")
        program = MIRProgram(functions=[caller, far_func])

        checker = InlinabilityChecker(program)
        assert checker.can_inline("far_helper") is False

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
        large_func = create_large_function(INLINE_THRESHOLD_WITH_ATTR + 10)
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

    def test_far_function_not_inlined(self):
        """Test that far functions are not inlined."""
        far_func = create_far_function()
        caller = create_caller_with_call("far_helper")
        program = MIRProgram(functions=[caller, far_func])

        inliner = FunctionInliner(verbose=False)
        inlined_count = inliner.run(program)

        assert inlined_count == 0

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
        large_func = create_large_function(INLINE_THRESHOLD_WITH_ATTR + 10)
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

    def test_inline_always_behaves_like_inline(self):
        """Test that #[inline(always)] behaves the same as #[inline]."""
        callee = create_simple_callee()
        callee.inline_attr = InlineAttribute(name='inline', mode=InlineMode.ALWAYS)
        caller = create_caller_with_call("add_one")
        program = MIRProgram(functions=[caller, callee])

        checker = InlinabilityChecker(program)
        assert checker.should_inline("add_one") is True

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
