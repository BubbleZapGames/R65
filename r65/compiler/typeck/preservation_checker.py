# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Register preservation validation.

With auto-generated save/restore, this checker validates that invalid
registers (B, PBR) are not in the preserves list, and that A is not
preserved by a function that returns a value. The actual register
modifications are allowed since the compiler generates PHA/PLA, PHX/PLX,
PHY/PLY, PHD/PLD, PHB/PLB automatically.

Also #[clobbers(...)]: its conflicts with the other attributes, and that a
function clobbering D or DBR is only ever reached by a direct call.
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


def check_clobbers(func_decl):
    """Validate a function's #[clobbers(...)] against its other attributes.

    The registers were checked when HIR read the attribute. A clobbered D or
    DBR is saved by each direct call, so a function that clobbers one must
    not be reached any other way: by `*dyn` dispatch here, by a function
    pointer in `check_clobbers_not_address_taken`.
    """
    attr = func_decl.clobbers_attr
    if attr is None:
        return
    name = func_decl.name
    if func_decl.is_entry or func_decl.interrupt_attr is not None:
        raise TypeCheckError(
            f"#[clobbers(...)] on '{name}', which is reached by a vector, not called",
            source_loc=func_decl.source_loc)

    clobbered = set(attr.registers)
    preserved = set(func_decl.preserves_attr.registers) if func_decl.preserves_attr else set()
    both = clobbered & preserved
    if both:
        raise TypeCheckError(
            f"'{name}' both preserves and clobbers {', '.join(sorted(both))}",
            source_loc=func_decl.source_loc)

    from r65.compiler.hir.attributes import DataBankMode
    if ('DBR' in clobbered and func_decl.mode_attr is not None
            and func_decl.mode_attr.databank == DataBankMode.INLINE):
        raise TypeCheckError(
            f"'{name}' clobbers DBR, but #[mode(databank=inline)] restores it",
            source_loc=func_decl.source_loc)

    saved = clobbered & {'D', 'DBR'}
    if saved and func_decl.is_trait_method:
        raise TypeCheckError(
            f"trait method '{name}' cannot clobber {', '.join(sorted(saved))}\n"
            f"  only a direct call saves it, and *dyn dispatch is not one",
            source_loc=func_decl.source_loc)


def check_clobbers_not_address_taken(program):
    """A function that clobbers D or DBR cannot be used as a value: a call
    through a function pointer would not save what it leaves changed."""
    saving = {f.name: f for f in program.all_functions()
              if f.clobbers_attr is not None
              and {'D', 'DBR'} & set(f.clobbers_attr.registers)}
    if not saving:
        return
    from r65.compiler.lint.call_graph import build_call_graph
    for name in sorted(build_call_graph(program).address_taken & set(saving)):
        func = saving[name]
        saved = {'D', 'DBR'} & set(func.clobbers_attr.registers)
        raise TypeCheckError(
            f"'{name}' clobbers {', '.join(sorted(saved))}, so it cannot be used as a value\n"
            f"  only a direct call saves it, and a call through a pointer is not one",
            source_loc=func.source_loc)
