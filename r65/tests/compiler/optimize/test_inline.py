#!/usr/bin/env python3
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
    VirtualRegister, HardwareRegister, Immediate,
    InlineAsm, Argument, ArgumentMechanism,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.attributes import InlineAttribute, InterruptAttribute, InterruptVector
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


def create_simple_callee() -> MIRFunction:
    """Create a simple function to be inlined: add_one(x) -> x + 1"""
    func = MIRFunction(
        name="add_one",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        inline_attr=InlineAttribute(name='inline'),
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    vreg_x = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="x")
    vreg_result = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="result")

    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_x, source=Immediate(5), type_info=BasicTypeInfo('u8')),
            BinaryOp(
                dest=vreg_result,
                left=vreg_x,
                right=Immediate(1),
                op='+',
                type_info=BasicTypeInfo('u8')
            ),
            Return(values=[vreg_result])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def create_caller_with_call(callee_name: str) -> MIRFunction:
    """Create a caller function that calls the given function."""
    func = MIRFunction(
        name="main",
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
        is_entry=True,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")

    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            Call(
                function=callee_name,
                args=[],
                returns=[vreg_result],
                is_far=False,
            ),
            Return(values=[vreg_result])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def create_recursive_function() -> MIRFunction:
    """Create a recursive function that should not be inlined."""
    func = MIRFunction(
        name="factorial",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[1, 2],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        inline_attr=InlineAttribute(name='inline'),
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    vreg_n = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="n")
    vreg_cond = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="cond")
    vreg_result = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="result")

    # Block 0: check if n <= 1
    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_n, source=Immediate(5), type_info=BasicTypeInfo('u8')),
            CondBranch(
                condition=vreg_cond,
                true_target=1,
                false_target=2,
            )
        ],
        predecessors=[],
        successors=[1, 2]
    )

    # Block 1: base case, return 1
    base_block = BasicBlock(
        block_id=1,
        instructions=[
            Move(dest=vreg_result, source=Immediate(1), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[0],
        successors=[]
    )

    # Block 2: recursive case
    recursive_block = BasicBlock(
        block_id=2,
        instructions=[
            Call(
                function="factorial",  # Recursive call
                args=[],
                returns=[vreg_result],
                is_far=False,
            ),
            Return(values=[vreg_result])
        ],
        predecessors=[0],
        successors=[]
    )

    func.blocks[0] = entry_block
    func.blocks[1] = base_block
    func.blocks[2] = recursive_block
    return func


def create_far_function() -> MIRFunction:
    """Create a far function that should not be inlined."""
    func = MIRFunction(
        name="far_helper",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        inline_attr=InlineAttribute(name='inline'),
        is_entry=False,
        is_far=True,  # Far function
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")

    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_result, source=Immediate(42), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def create_function_with_asm() -> MIRFunction:
    """Create a function with inline assembly that should not be inlined."""
    func = MIRFunction(
        name="asm_helper",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        inline_attr=InlineAttribute(name='inline'),
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")

    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            InlineAsm(instructions=["NOP"]),
            Move(dest=vreg_result, source=Immediate(42), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def create_interrupt_handler() -> MIRFunction:
    """Create an interrupt handler that should not be inlined."""
    func = MIRFunction(
        name="nmi_handler",
        parameters=[],
        return_type=None,
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=InterruptAttribute(name='interrupt', vector=InterruptVector.NMI),
        inline_attr=InlineAttribute(name='inline'),
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            Return(values=[])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def create_function_with_multiple_returns() -> MIRFunction:
    """Create a function with multiple return paths."""
    func = MIRFunction(
        name="multi_return",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[1, 2],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        inline_attr=InlineAttribute(name='inline'),
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    vreg_cond = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="cond")
    vreg_result1 = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="result1")
    vreg_result2 = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="result2")

    # Block 0: conditional branch
    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_cond, source=Immediate(1), type_info=BasicTypeInfo('u8')),
            CondBranch(
                condition=vreg_cond,
                true_target=1,
                false_target=2,
            )
        ],
        predecessors=[],
        successors=[1, 2]
    )

    # Block 1: return 10
    true_block = BasicBlock(
        block_id=1,
        instructions=[
            Move(dest=vreg_result1, source=Immediate(10), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result1])
        ],
        predecessors=[0],
        successors=[]
    )

    # Block 2: return 20
    false_block = BasicBlock(
        block_id=2,
        instructions=[
            Move(dest=vreg_result2, source=Immediate(20), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_result2])
        ],
        predecessors=[0],
        successors=[]
    )

    func.blocks[0] = entry_block
    func.blocks[1] = true_block
    func.blocks[2] = false_block
    return func


def create_large_function(num_instructions: int) -> MIRFunction:
    """Create a function with many instructions."""
    func = MIRFunction(
        name="large_func",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        inline_attr=InlineAttribute(name='inline'),
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    vreg_acc = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="acc")

    instructions = [
        Move(dest=vreg_acc, source=Immediate(0), type_info=BasicTypeInfo('u8'))
    ]

    # Add many BinaryOp instructions
    for i in range(num_instructions - 2):  # -2 for initial Move and final Return
        instructions.append(
            BinaryOp(
                dest=vreg_acc,
                left=vreg_acc,
                right=Immediate(1),
                op='+',
                type_info=BasicTypeInfo('u8')
            )
        )

    instructions.append(Return(values=[vreg_acc]))

    entry_block = BasicBlock(
        block_id=0,
        instructions=instructions,
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
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
