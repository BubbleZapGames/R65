"""
Register preservation validation.

With auto-generated save/restore, this checker now only validates
that invalid registers (B, PBR) are not in the preserves list.
The actual register modifications are allowed since the compiler
generates PHA/PLA, PHX/PLX, PHY/PLY, PHD/PLD, PHB/PLB automatically.
"""

from r65.compiler.typeck.errors import *
from r65.compiler.hir import *


class PreservationChecker:
    """Validates register preservation contracts."""

    # Registers that cannot be preserved
    INVALID_PRESERVES = {'B', 'PBR'}

    # Valid registers for preservation
    VALID_PRESERVES = {'A', 'X', 'Y', 'D', 'DBR', 'STATUS'}

    def __init__(self, func_decl):
        self.func_decl = func_decl
        self.preserves_attr = func_decl.preserves_attr

    def check(self):
        """
        Validate register preservation attribute.

        Only checks for invalid registers (B, PBR).
        Modifications to preserved registers are allowed since
        the compiler auto-generates save/restore code.
        """
        if not self.preserves_attr:
            return  # No preservation contract

        preserved_regs = set(self.preserves_attr.registers)

        # Check for invalid registers
        for reg in preserved_regs:
            if reg in self.INVALID_PRESERVES:
                if reg == 'B':
                    raise TypeCheckError(
                        f"B register cannot be in #[preserves(...)]\n"
                        f"  B is the high byte of the A register and cannot be preserved independently\n"
                        f"  Remove B from the preserves list",
                        source_loc=self.func_decl.source_loc
                    )
                elif reg == 'PBR':
                    raise TypeCheckError(
                        f"PBR (Program Bank Register) cannot be in #[preserves(...)]\n"
                        f"  PBR is read-only and cannot be modified\n"
                        f"  Remove PBR from the preserves list",
                        source_loc=self.func_decl.source_loc
                    )

            if reg not in self.VALID_PRESERVES and reg not in self.INVALID_PRESERVES:
                raise TypeCheckError(
                    f"Unknown register '{reg}' in #[preserves(...)]\n"
                    f"  Valid registers: A, X, Y, D, DBR, STATUS\n"
                    f"  Invalid registers: B (tied to A), PBR (read-only)",
                    source_loc=self.func_decl.source_loc
                )

        # Note: We no longer check for modifications since the compiler
        # automatically generates PHA/PLA, PHX/PLX, etc. for preserved registers
