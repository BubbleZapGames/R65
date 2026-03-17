# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
MIR (Mid-level Intermediate Representation) for R65 compiler.

Provides a CFG-based representation with virtual registers for code generation.
"""

from r65.compiler.mir.nodes import (
    # Instructions
    MIRInstruction,
    Load, Store, Move,
    BinaryOp, UnaryOp, Compare, BitTest, Rotate,
    Jump, CondBranch, JumpTable, LookupTable, Return, ReturnFromInterrupt,
    StatusFlagTest, StatusFlagSet, StatusFlagRead,
    Call, Argument, ArgumentMechanism,
    SetMode, Push, Pull, SaveRegister, RestoreRegister,
    InlineAsm,
    # Operands
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    # CFG
    BasicBlock, MIRFunction, MIRProgram
)

from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.mir.register_tracker import RegisterAliasTracker, RegisterAlias
from r65.compiler.mir.builder import MIRBuilder
from r65.compiler.mir.mode_tracker import MIRModeTracker

__all__ = [
    # Instructions
    'MIRInstruction',
    'Load', 'Store', 'Move',
    'BinaryOp', 'UnaryOp', 'Compare', 'BitTest', 'Rotate',
    'Jump', 'CondBranch', 'JumpTable', 'LookupTable', 'Return', 'ReturnFromInterrupt',
    'StatusFlagTest', 'StatusFlagSet', 'StatusFlagRead',
    'Call', 'Argument', 'ArgumentMechanism',
    'SetMode', 'Push', 'Pull', 'SaveRegister', 'RestoreRegister',
    'InlineAsm',
    # Operands
    'VirtualRegister', 'HardwareRegister', 'Immediate', 'MemoryLocation',
    # CFG
    'BasicBlock', 'MIRFunction', 'MIRProgram',
    # Utilities
    'VirtualRegisterAllocator',
    'RegisterAliasTracker', 'RegisterAlias',
    # Builder
    'MIRBuilder',
    # Mode tracking
    'MIRModeTracker',
]
