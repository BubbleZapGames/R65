#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for far-to-near call optimization pass."""

import pytest

from r65.compiler.optimize.far_to_near import FarToNearOptimizer
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock,
    Move, Return, Call,
    VirtualRegister, Immediate,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.attributes import (
    BankAttribute, InterruptAttribute, InterruptVector
)
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.tests.language.common import make_mir_function


def create_near_function(name: str, is_entry: bool = False):
    """Create a simple near function."""
    func = make_mir_function(name, is_entry=is_entry)

    vreg = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg, source=Immediate(42), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_far_function(name: str, bank: int = 0):
    """Create a simple far function in the specified bank."""
    func = make_mir_function(name, is_far=True,
                             bank_attr=BankAttribute(name='bank', bank_number=bank))

    vreg = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg, source=Immediate(42), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_caller_in_bank(name: str, callee_name: str, bank: int = 0,
                          is_entry: bool = False, call_is_far: bool = False):
    """Create a caller function that calls the given function."""
    func = make_mir_function(name, is_entry=is_entry,
                             bank_attr=BankAttribute(name='bank', bank_number=bank))

    vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Call(function=callee_name, args=[], returns=[vreg_result],
                 is_far=call_is_far),
            Return(values=[vreg_result])
        ],
        predecessors=[], successors=[]
    )
    return func


class TestFarToNearOptimizer:
    """Test cases for far-to-near call optimization."""

    def test_convert_far_to_near_same_bank(self):
        """Far function with all callers in same bank should be converted."""
        # Create: caller (bank 0) -> far_callee (bank 0)
        far_callee = create_far_function("far_callee", bank=0)
        caller = create_caller_in_bank("caller", "far_callee", bank=0,
                                        is_entry=True, call_is_far=True)

        program = MIRProgram(functions=[caller, far_callee])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 1
        assert not far_callee.is_far, "Function should be converted to near"

        # Verify call site was updated
        call_instr = caller.blocks[0].instructions[0]
        assert isinstance(call_instr, Call)
        assert not call_instr.is_far, "Call should be converted to near"

    def test_no_convert_different_banks(self):
        """Far function with callers in different bank should NOT be converted."""
        # Create: caller (bank 1) -> far_callee (bank 0)
        far_callee = create_far_function("far_callee", bank=0)
        caller = create_caller_in_bank("caller", "far_callee", bank=1,
                                        is_entry=True, call_is_far=True)

        program = MIRProgram(functions=[caller, far_callee])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 0
        assert far_callee.is_far, "Function should remain far"

    def test_no_convert_mixed_banks(self):
        """Far function with callers in mixed banks should NOT be converted."""
        # Create: caller1 (bank 0), caller2 (bank 1) -> far_callee (bank 0)
        far_callee = create_far_function("far_callee", bank=0)
        caller1 = create_caller_in_bank("caller1", "far_callee", bank=0, call_is_far=True)
        caller2 = create_caller_in_bank("caller2", "far_callee", bank=1,
                                         is_entry=True, call_is_far=True)

        program = MIRProgram(functions=[caller1, caller2, far_callee])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 0
        assert far_callee.is_far, "Function should remain far"

    def test_no_convert_interrupt_handler(self):
        """Interrupt handlers should NOT be converted (called by hardware)."""
        func = MIRFunction(
            name="nmi_handler",
            parameters=[],
            return_type=None,
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=BankAttribute(name='bank', bank_number=0),
            interrupt_attr=InterruptAttribute(name='interrupt', vector=InterruptVector.NMI),
            inline_attr=None,
            is_entry=False,
            is_far=True,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )

        vreg = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'))
        entry_block = BasicBlock(
            block_id=0,
            instructions=[
                Move(dest=vreg, source=Immediate(0), type_info=BasicTypeInfo('u8')),
                Return(values=[])
            ],
            predecessors=[],
            successors=[]
        )
        func.blocks[0] = entry_block

        # Add a caller in the same bank
        caller = create_caller_in_bank("caller", "nmi_handler", bank=0,
                                        is_entry=True, call_is_far=True)

        program = MIRProgram(functions=[caller, func])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 0
        assert func.is_far, "Interrupt handler should remain far"

    def test_no_convert_address_taken(self):
        """Functions with address taken should NOT be converted."""
        far_callee = create_far_function("far_callee", bank=0)

        # Create caller that takes address of far_callee
        caller = MIRFunction(
            name="caller",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=BankAttribute(name='bank', bank_number=0),
            interrupt_attr=None,
            inline_attr=None,
            is_entry=True,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )

        vreg_ptr = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint="fptr")
        vreg_result = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="result")

        entry_block = BasicBlock(
            block_id=0,
            instructions=[
                # Take address of function (function pointer)
                Move(dest=vreg_ptr, source=Immediate("far_callee"), type_info=BasicTypeInfo('u16')),
                # Also call directly
                Call(function="far_callee", args=[], returns=[vreg_result], is_far=True),
                Return(values=[vreg_result])
            ],
            predecessors=[],
            successors=[]
        )
        caller.blocks[0] = entry_block

        program = MIRProgram(functions=[caller, far_callee])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 0
        assert far_callee.is_far, "Function with address taken should remain far"

    def test_multiple_callers_same_bank(self):
        """Multiple callers in the same bank should allow conversion."""
        far_callee = create_far_function("far_callee", bank=0)
        caller1 = create_caller_in_bank("caller1", "far_callee", bank=0, call_is_far=True)
        caller2 = create_caller_in_bank("caller2", "far_callee", bank=0, call_is_far=True)
        main = create_caller_in_bank("main", "caller1", bank=0, is_entry=True)

        program = MIRProgram(functions=[main, caller1, caller2, far_callee])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 1
        assert not far_callee.is_far, "Function should be converted to near"

        # Verify both call sites were updated
        call1 = caller1.blocks[0].instructions[0]
        call2 = caller2.blocks[0].instructions[0]
        assert isinstance(call1, Call) and not call1.is_far
        assert isinstance(call2, Call) and not call2.is_far

    def test_near_function_unchanged(self):
        """Near functions should not be affected by the optimization."""
        near_func = create_near_function("near_func")
        caller = create_caller_in_bank("caller", "near_func", bank=0,
                                        is_entry=True, call_is_far=False)

        program = MIRProgram(functions=[caller, near_func])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 0
        assert not near_func.is_far

    def test_no_callers_not_converted(self):
        """Functions with no callers should NOT be converted (may be externally called)."""
        far_func = create_far_function("far_func", bank=0)
        main = create_near_function("main", is_entry=True)

        program = MIRProgram(functions=[main, far_func])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 0
        assert far_func.is_far, "Function with no callers should remain far"

    def test_default_bank_handling(self):
        """Functions without explicit bank attribute should use bank 0."""
        # Create far function without bank attribute (defaults to bank 0)
        far_callee = MIRFunction(
            name="far_callee",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,  # No explicit bank
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            is_far=True,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )

        vreg = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'))
        entry_block = BasicBlock(
            block_id=0,
            instructions=[
                Move(dest=vreg, source=Immediate(42), type_info=BasicTypeInfo('u8')),
                Return(values=[vreg])
            ],
            predecessors=[],
            successors=[]
        )
        far_callee.blocks[0] = entry_block

        # Caller also without bank attribute (defaults to bank 0)
        caller = MIRFunction(
            name="caller",
            parameters=[],
            return_type=BasicTypeInfo('u8'),
            blocks={},
            entry_block_id=0,
            exit_block_ids=[0],
            mode_attr=None,
            preserves_attr=None,
            bank_attr=None,  # No explicit bank
            interrupt_attr=None,
            inline_attr=None,
            is_entry=True,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=None,
        )

        vreg_result = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'))
        caller_block = BasicBlock(
            block_id=0,
            instructions=[
                Call(function="far_callee", args=[], returns=[vreg_result], is_far=True),
                Return(values=[vreg_result])
            ],
            predecessors=[],
            successors=[]
        )
        caller.blocks[0] = caller_block

        program = MIRProgram(functions=[caller, far_callee])

        optimizer = FarToNearOptimizer(verbose=False)
        converted = optimizer.optimize(program)

        assert converted == 1
        assert not far_callee.is_far, "Function should be converted to near"


class TestFarToNearVerbose:
    """Test verbose output for debugging."""

    def test_verbose_output(self, capsys):
        """Test that verbose mode prints conversion info."""
        far_callee = create_far_function("far_callee", bank=0)
        caller = create_caller_in_bank("caller", "far_callee", bank=0,
                                        is_entry=True, call_is_far=True)

        program = MIRProgram(functions=[caller, far_callee])

        optimizer = FarToNearOptimizer(verbose=True)
        converted = optimizer.optimize(program)

        captured = capsys.readouterr()
        assert converted == 1
        assert "Convert far_callee" in captured.out or "Converted function far_callee" in captured.out
