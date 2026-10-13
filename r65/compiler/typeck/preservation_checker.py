# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Register preservation validation.

With auto-generated save/restore, this checker validates that invalid
registers (B, PBR) are not in the preserves list, and that A is not
preserved by a function that returns a value. The actual register
modifications are allowed since the compiler generates PHA/PLA, PHX/PLX,
PHY/PLY, PHD/PLD, PHB/PLB automatically.
"""

from r65.compiler.typeck.errors import *
from r65.compiler.hir import *
from r65.compiler.hir.types import BasicTypeInfo, NeverTypeInfo


class PreservationChecker:
    """Validates register preservation contracts."""

    def __init__(self, func_decl):
        self.func_decl = func_decl
        self.preserves_attr = func_decl.preserves_attr

    def check(self):
        """
        Validate register preservation attribute.

        The register names were checked when HIR read the attribute.
        Modifications to preserved registers are allowed since
        the compiler auto-generates save/restore code.
        """
        if not self.preserves_attr:
            return  # No preservation contract

        preserved_regs = set(self.preserves_attr.registers)

        # Preserving A and returning a value are the same register asking to
        # hold two things. The epilogue's PLA runs after the body, so it
        # restores the entry value straight over the result.
        if 'A' in preserved_regs and self._returns_a_value():
            raise TypeCheckError(
                f"A cannot be in #[preserves(...)] on a function that returns a value\n"
                f"  The first return value is passed in A, and the restore at exit\n"
                f"  would overwrite it with the value A held on entry\n"
                f"  Remove A from the preserves list, or drop the return value",
                source_loc=self.func_decl.source_loc
            )

    def _returns_a_value(self) -> bool:
        """Whether this function hands a value back in a register.

        The first return value always lands in A, whatever its type, so any
        declared return type conflicts. A `-> !` function never reaches its
        epilogue, so there is nothing to overwrite.
        """
        ret = self.func_decl.return_type
        if ret is None or isinstance(ret, NeverTypeInfo):
            return False
        if isinstance(ret, BasicTypeInfo) and ret.name == 'void':
            return False
        return True
