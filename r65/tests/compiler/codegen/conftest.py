# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Shared fixtures for codegen tests."""

import pytest
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.register_alloc import RegisterAllocator, ScratchRegisterPool
from r65.compiler.codegen.instruction_select import InstructionSelector


@pytest.fixture
def codegen():
    """Create a standard codegen test harness with empty scratch pool.

    Returns a namespace with emitter, mem_alloc, pool, reg_alloc, selector.
    Pool starts empty — call pool.add_scratch() to customize.
    """
    class CodegenHarness:
        def __init__(self):
            self.emitter = AssemblyEmitter()
            self.mem_alloc = MemoryAllocator()
            self.pool = ScratchRegisterPool()
            self.reg_alloc = RegisterAllocator(scratch_pool=self.pool)
            self.selector = InstructionSelector(
                self.emitter, self.reg_alloc, self.mem_alloc
            )
    return CodegenHarness()
