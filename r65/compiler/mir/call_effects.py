# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
What a call does to the registers: which it reads, returns in and leaves alone.

One record per call, read by everything that has to know: call selection
(spills, reloads, where A may be parked during argument setup), clobber
regions, and slot allocation's coalescing. The views differ only in how much
they may assume:

- `preserved` adds the index registers the callee's code never writes
  (clobber_summary). The summaries exist only during codegen, which generates
  callees first.
- `declared` is `#[preserves(...)]` alone, for analyses that run before the
  callee's code exists - and, for an `extern fn`, the index registers its
  `#[clobbers(...)]` does not list (none listed: both kept). An extern's code
  is never seen, so its declaration is the whole contract.
- `caller_saves` is the D and DBR a callee's `#[clobbers(...)]` lists: they
  are callee-saved by convention, so listing one makes the call save it.
- The MIR passes (alias check, index streams, loop promotion, X reuse) assume
  a call clobbers everything (`CLOBBERS_ALL`).

The registers a result is *collected* into (a return vreg the allocator put in
X, landed there by TAX) are not here: they depend on slot allocation, not on
the call, and call selection adds them itself.
"""

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional, Tuple

from r65.compiler.mir.nodes import ArgumentMechanism, Call, CallBase

INDEX_REGS: FrozenSet[str] = frozenset({'X', 'Y'})
CALLER_SAVED_BY_CLOBBERS: FrozenSet[str] = frozenset({'D', 'DBR'})


@dataclass(frozen=True)
class CallEffects:
    results: Tuple[str, ...]      # registers the callee returns in, in order
    arg_regs: FrozenSet[str]      # registers arguments are passed in ('Y' for self)
    has_stack_args: bool
    declared: FrozenSet[str]      # #[preserves(...)]
    inferred: FrozenSet[str]      # index registers the callee's code never writes
    caller_saves: FrozenSet[str]  # D/DBR the callee's #[clobbers(...)] lists
    entry_mode: Optional[object]  # ModeState the callee expects, if known
    exit_mode: Optional[object]

    @property
    def preserved(self) -> FrozenSet[str]:
        return self.declared | self.inferred

    def may_write(self, reg: str, *, declared_only: bool = False) -> bool:
        """Whether *reg* may hold something else after the call - a result
        included. B cannot be preserved: it is A's high byte."""
        kept = self.declared if declared_only else self.preserved
        return reg == 'B' or reg not in kept


def _result_registers(instr: CallBase) -> Tuple[str, ...]:
    """The callee's return registers, one per value it returns.

    `instr.returns` may be empty when the values are captured by separate
    Moves after the call, so the count comes from callee_return_type.
    """
    from r65.compiler.codegen.constants import get_return_registers
    from r65.compiler.hir.types import BasicTypeInfo, MultiReturnTypeInfo
    ret = instr.callee_return_type
    if ret is None:
        return ('A', 'X', 'Y')[:len(instr.returns)]
    if ret == BasicTypeInfo('void'):
        return ()
    regs = get_return_registers(ret, getattr(instr, 'callee_entry_m_mode', None))
    count = len(ret.element_types) if isinstance(ret, MultiReturnTypeInfo) else 1
    return tuple(regs[:count])


def call_effects(instr: CallBase,
                 summaries: Optional[Mapping[str, FrozenSet[str]]] = None
                 ) -> CallEffects:
    """The register effects of *instr*, a Call or TraitDispatch.

    *summaries* maps a generated function to the index registers its code
    may change. A named call keeps the rest - except a register one of its
    own arguments goes to, and nothing at all for a call with stack
    arguments: parking A across their setup and pulling them off after
    both go through X or Y.
    """
    args = instr.args
    arg_regs = {a.location.name for a in args
                if a.mechanism == ArgumentMechanism.REGISTER
                and getattr(a.location, 'name', None) is not None}
    if any(a.mechanism == ArgumentMechanism.SELF_Y for a in args):
        arg_regs.add('Y')
    has_stack_args = any(a.mechanism == ArgumentMechanism.STACK for a in args)

    attr = getattr(instr, 'preserves_attr', None)
    declared = frozenset(attr.registers) if attr else frozenset()
    clobbers_attr = getattr(instr, 'clobbers_attr', None)
    clobbers = frozenset(clobbers_attr.registers) if clobbers_attr else frozenset()
    if getattr(instr, 'callee_is_extern', False):
        declared |= INDEX_REGS - clobbers - arg_regs

    inferred = frozenset()
    if (summaries and type(instr) is Call and isinstance(instr.function, str)
            and not instr.builtin_name and not has_stack_args):
        writes = summaries.get(instr.function)
        if writes is not None:
            inferred = INDEX_REGS - writes - arg_regs

    return CallEffects(
        results=_result_registers(instr),
        arg_regs=frozenset(arg_regs),
        has_stack_args=has_stack_args,
        declared=declared,
        inferred=inferred,
        caller_saves=clobbers & CALLER_SAVED_BY_CLOBBERS,
        entry_mode=getattr(instr, 'callee_entry_m_mode', None),
        exit_mode=getattr(instr, 'callee_exit_m_mode', None),
    )
