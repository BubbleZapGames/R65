# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Liveness of register-bound parameters during instruction selection.

A parameter bound to a register (`idx @ X`, `v @ A`) occupies it while some
instruction still to run may read it; after that the register is free for
scratch. Instruction selection asks `is_free('A')` before borrowing A for a
temporary.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List

from r65.compiler.mir.liveness import liveness_uses
from r65.compiler.mir.nodes import HardwareRegister


@dataclass
class HardwareRegisterTracker:
    """Which registers still hold a parameter some later instruction reads."""

    # Per instruction, in emission order: the tracked registers that it, or
    # any instruction reachable after it in the CFG (back edges included),
    # reads.
    _read_from: List[FrozenSet[str]] = field(default_factory=list)

    # Index of the instruction being generated, in emission order
    # (function_gen.compute_block_order, then each block's instructions).
    current_index: int = 0

    def initialize_from_parameters(self, mir_func):
        """Find, for each instruction, which bound registers are read from it on."""
        from r65.compiler.codegen.function_gen import compute_block_order

        self.current_index = 0
        self._read_from = []
        aliases = mir_func.alias_tracker.aliases if getattr(mir_func, 'alias_tracker', None) else {}
        tracked = {alias.hardware_reg.name for alias in aliases.values()}
        # Parameters the FixedStack pass promoted to a register arrive there
        # the same way, read by the Move it injects at entry
        tracked |= set(getattr(mir_func, 'hw_param_regs', {}).values())
        if not tracked:
            return

        def reads(instr) -> FrozenSet[str]:
            return frozenset(use.name for use in liveness_uses(instr)
                             if isinstance(use, HardwareRegister) and use.name in tracked)

        blocks = mir_func.blocks
        block_reads: Dict[int, FrozenSet[str]] = {
            bid: frozenset().union(*(reads(i) for i in block.instructions))
            for bid, block in blocks.items()}

        # Registers read anywhere reachable after a block's end: a fixpoint
        # over successors, so a read in a loop header counts for the body.
        reach_out: Dict[int, FrozenSet[str]] = {bid: frozenset() for bid in blocks}
        changed = True
        while changed:
            changed = False
            for bid, block in blocks.items():
                out = frozenset().union(*(block_reads[s] | reach_out[s]
                                          for s in block.successors if s in blocks))
                if out != reach_out[bid]:
                    reach_out[bid] = out
                    changed = True

        for bid in compute_block_order(mir_func):
            acc = reach_out[bid]
            per_instr = []
            for instr in reversed(blocks[bid].instructions):
                acc = acc | reads(instr)
                per_instr.append(acc)
            self._read_from.extend(reversed(per_instr))

    def advance_to(self, instruction_index: int):
        """Move to the instruction at *instruction_index*."""
        self.current_index = instruction_index

    def is_free(self, reg_name: str) -> bool:
        """True unless *reg_name* holds a parameter still to be read."""
        if self.current_index >= len(self._read_from):
            return True
        return reg_name.upper() not in self._read_from[self.current_index]
