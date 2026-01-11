"""
R65 65816 CPU Emulator

A headless 65816 CPU emulator for testing R65-compiled code with execution trace logging.
"""

from .cpu import CPU65816
from .memory import Memory
from .trace import TraceLogger
from .disasm import disassemble

__all__ = ['CPU65816', 'Memory', 'TraceLogger', 'disassemble']
