# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
R65 65816 CPU Emulator

A headless 65816 CPU emulator for testing R65-compiled code with execution trace logging.
"""

from .cpu import CPU65816
from .memory import Memory, SNESMemory
from .trace import TraceLogger
from .disasm import disassemble
from .compare import RomComparator, NormalizedTrace, Divergence, CompareTraceLogger

__all__ = [
    'CPU65816', 'Memory', 'SNESMemory', 'TraceLogger', 'disassemble',
    'RomComparator', 'NormalizedTrace', 'Divergence', 'CompareTraceLogger'
]
