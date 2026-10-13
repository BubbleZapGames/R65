# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Code generation: MIR → WLA-DX assembly for 65816.

Transforms MIR (Mid-level Intermediate Representation) into readable,
debuggable WLA-DX assembly code.
"""

from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.codegen import ProgramCodeGenerator
from r65.compiler.codegen.memory_alloc import MemoryAllocator, AllocationInfo
from r65.compiler.codegen.symbol_gen import SymbolDefinitionGenerator
from r65.compiler.codegen.register_alloc import (
    RegisterAllocator,
    ScratchRegisterPool,
    PhysicalLocation,
    LocationKind,
)
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.function_gen import FunctionCodeGenerator, ProgramFunctionGenerator
from r65.compiler.codegen.hw_register_tracker import HardwareRegisterTracker

__all__ = [
    'AssemblyEmitter',
    'ProgramCodeGenerator',
    'MemoryAllocator',
    'AllocationInfo',
    'SymbolDefinitionGenerator',
    'RegisterAllocator',
    'ScratchRegisterPool',
    'PhysicalLocation',
    'LocationKind',
    'InstructionSelector',
    'FunctionCodeGenerator',
    'ProgramFunctionGenerator',
    'HardwareRegisterTracker',
]
