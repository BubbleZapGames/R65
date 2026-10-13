# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Liveness of register-bound parameters during instruction selection.

A parameter bound to a register (`idx @ X`, `v @ A`) occupies it until its
last use; after that the register is free for scratch. Instruction selection
asks `is_free('A')` before borrowing A for a temporary.
"""

from dataclasses import dataclass, field
from typing import Dict

from r65.compiler.mir.liveness import liveness_uses
from r65.compiler.mir.nodes import HardwareRegister


@dataclass
class HardwareRegisterTracker:
    """Which registers still hold a parameter some later instruction reads."""

    # Register name -> index of the last instruction reading its parameter
    # (-1 when nothing does).
    _last_use: Dict[str, int] = field(default_factory=dict)

    # Index of the instruction being generated, in block-id order.
    current_index: int = 0

    def initialize_from_parameters(self, mir_func):
        """Find the last read of each register a parameter is bound to."""
        self.current_index = 0
        aliases = mir_func.alias_tracker.aliases if getattr(mir_func, 'alias_tracker', None) else {}
        self._last_use = {alias.hardware_reg.name: -1 for alias in aliases.values()}
        if not self._last_use:
            return
        index = 0
        for block_id in sorted(mir_func.blocks):
            for instr in mir_func.blocks[block_id].instructions:
                for use in liveness_uses(instr):
                    if isinstance(use, HardwareRegister) and use.name in self._last_use:
                        self._last_use[use.name] = index
                index += 1

    def advance_to(self, instruction_index: int):
        """Move to the instruction at *instruction_index*."""
        self.current_index = instruction_index

    def is_free(self, reg_name: str) -> bool:
        """True unless *reg_name* holds a parameter still to be read."""
        last = self._last_use.get(reg_name.upper(), -1)
        return self.current_index > last
