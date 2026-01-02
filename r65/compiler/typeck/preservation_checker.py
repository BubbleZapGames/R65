"""
Register preservation validation.

Validates that functions correctly preserve registers declared
in #[preserves(...)] attribute.
"""

from typing import Set
from r65.compiler.typeck.errors import *
from r65.compiler.hir import *


class PreservationChecker:
    """Validates register preservation contracts."""

    def __init__(self, func_decl, cfg):
        self.func_decl = func_decl
        self.cfg = cfg
        self.preserves_attr = func_decl.preserves_attr

    def check(self):
        """
        Validate register preservation.

        Simplified: just checks if preserved registers are modified.
        """
        if not self.preserves_attr:
            return  # No preservation contract

        preserved_regs = set(self.preserves_attr.registers)

        # Track which registers are modified
        modified_regs = self._find_modified_registers()

        # Check each preserved register
        for reg in preserved_regs:
            if reg in modified_regs:
                # Register is modified - error
                raise TypeCheckError(
                    f"Register {reg} is in #[preserves(...)] but is modified in function\n"
                    f"  Functions must manually save/restore preserved registers\n"
                    f"  Example:\n"
                    f"    let saved = {reg};\n"
                    f"    {reg} = ...;  // use register\n"
                    f"    {reg} = saved;  // restore before return",
                    source_loc=self.func_decl.source_loc
                )

    def _find_modified_registers(self) -> Set[str]:
        """Find all registers modified in the function."""
        modified = set()

        for block in self.cfg.blocks.values():
            for stmt in block.statements:
                self._check_statement(stmt, modified)

        return modified

    def _check_statement(self, stmt, modified: Set[str]):
        """Check if statement modifies any registers."""
        if isinstance(stmt, HIRAssignment):
            # Check if target is a register
            if isinstance(stmt.target, HIRRegister):
                modified.add(stmt.target.name)

        elif isinstance(stmt, HIRLetStmt):
            # Register alias bindings alias the register
            if isinstance(stmt.binding, RegisterLetBinding):
                # If variable is mutable, register can be modified through it
                if stmt.is_mutable:
                    modified.add(stmt.binding.register_name)
