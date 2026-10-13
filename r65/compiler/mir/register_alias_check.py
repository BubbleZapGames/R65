# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Reject a register alias whose register is overwritten before its last use.

`let x @ A = ...` (or `@ B`, `@ X`, `@ Y`) makes `x` a name for the register,
not a copy of a value: nothing preserves it. So code between the binding and a
later read of `x` that writes the register - an expression evaluated through A,
anything done with a 16-bit accumulator (whose high byte is B), an array index
loaded into X, a pointer read indexing with Y, a call - leaves `x` reading
whatever was put there instead, so that write is an error at the
overwriting line.

The aliases are tracked through the MIR right after lowering, before any
optimisation can move or remove a clobber: every operand `get_alias` returned
carries its alias (`HardwareRegister.alias`), a Move into the register with a
tagged dest is the alias's (re)definition, and `RegisterAliasTracker.def_points`
holds the definitions that emit no Move. A forward dataflow over the CFG keeps,
per alias, whether its register still holds it - clean, or clobbered at some
instruction - and a tagged read of a clobbered alias is the error. A clobber on
any path into a join counts.

The "may write R" predicates model what codegen emits for each node, and lean
conservative where codegen chooses later: anything evaluated into a virtual
register goes through A; an array index reaches X, and a pointer read's index
or field offset Y, by a Move the MIR already shows, except that a pointer read
without an explicit Y index still puts its offset in Y.
"""

from typing import Dict, Optional

from r65.compiler.errors import MIRLoweringError
from r65.compiler.mir.nodes import (
    MIRFunction, HardwareRegister, VirtualRegister, Immediate,
    CLOBBERS_ALL, fixed_clobbers, Pull, Push,
    StatusFlagRead, Store, StoreIndirect, LoadIndirect, Compare, BitTest,
    CondBranch, Move, BinaryOp, UnaryOp, IncDecMemory, SetMode,
    OperandRole, iter_operands,
)
from r65.compiler.codegen.type_utils import get_type_size, get_vreg_size

CHECKED = ('A', 'B', 'X', 'Y')



def _hw(v, name=None) -> bool:
    return isinstance(v, HardwareRegister) and (name is None or v.name == name)


def _size(type_info) -> int:
    return get_type_size(type_info) if type_info is not None else 1


def _wide(instr) -> bool:
    """True if codegen may run *instr* with a 16-bit accumulator."""
    if _size(getattr(instr, 'type_info', None)) >= 2:
        return True
    return any(isinstance(v, VirtualRegister) and get_vreg_size(v) >= 2
               for _, v in iter_operands(instr, role=OperandRole.WRITE))


def _writes_b(instr) -> bool:
    """B is A's high byte: whatever writes A at 16 bits writes it too. An
    8-bit operation in A leaves it alone, so this asks more than "writes A"."""
    if isinstance(instr, Pull):
        return _hw(instr.register, 'A')     # the pull's width is the push's
    explicit_a = any(_hw(v, 'A') for _, v in iter_operands(instr, role=OperandRole.WRITE)) \
        and not (isinstance(instr, Move) and _hw(instr.source, 'A'))
    return (explicit_a or _implicit_clobbers(instr, 'A')) and _wide(instr)


def _implicit_clobbers(instr, reg: str) -> bool:
    """True if codegen for *instr* may overwrite *reg* in a way its explicit
    hardware-register dest (handled by the caller) does not show."""
    if isinstance(instr, CLOBBERS_ALL) or reg in fixed_clobbers(instr):
        return True
    if reg == 'B':
        return 'A' in fixed_clobbers(instr) or _writes_b(instr)
    if isinstance(instr, Pull):
        return _hw(instr.register, reg)

    if reg == 'Y':
        # A pointer read or write puts its index or field offset in Y,
        # unless the index is already there (set by a Move the MIR shows).
        if isinstance(instr, (LoadIndirect, StoreIndirect)):
            return not (instr.index_register == 'Y' and not getattr(instr, 'offset', 0))
        return False
    if reg == 'X':
        return False

    # A.
    if isinstance(instr, (Push, SetMode, IncDecMemory)):
        return False
    if isinstance(instr, StatusFlagRead):
        return True
    if isinstance(instr, Move):
        if isinstance(instr.dest, VirtualRegister):
            # Copying A out (STA) leaves it; anything else is evaluated in A.
            return not _hw(instr.source, 'A')
        return False      # into X/Y: LDX/TAX and friends; into A: the caller
    if isinstance(instr, (BinaryOp, UnaryOp)) and _hw(instr.dest) and instr.dest.name in ('X', 'Y'):
        # X = X ± 1 is INX/DEX; any other arithmetic on X/Y goes through A.
        return not (isinstance(instr, BinaryOp) and instr.op in ('+', '-')
                    and _hw(instr.left, instr.dest.name)
                    and isinstance(instr.right, Immediate) and instr.right.value == 1)
    if isinstance(instr, (Store, StoreIndirect)):
        src = instr.source
        if _hw(src, 'A'):
            return False
        # A 2-byte STX/STY; a 1-byte store from X/Y goes through A.
        if _hw(src) and src.name in ('X', 'Y') and _size(instr.type_info) == 2:
            return False
        return True
    if isinstance(instr, Compare):
        return not (_hw(instr.left) and instr.left.name in ('A', 'X', 'Y'))
    if isinstance(instr, (BitTest, CondBranch)):
        v = instr.value if isinstance(instr, BitTest) else instr.condition
        return v is not None and not _hw(v, 'A')
    # Anything else that produces a virtual register computes it in A.
    for _, v in iter_operands(instr, role=OperandRole.WRITE):
        if isinstance(v, VirtualRegister):
            return True
    return False


class _State:
    """Per alias: None while its register holds it, else the source_loc of
    the instruction that overwrote it (or True when that has none)."""

    def __init__(self, held: Optional[Dict] = None):
        self.held: Dict[int, object] = dict(held or {})

    def copy(self):
        return _State(self.held)

    def merge(self, other: '_State') -> bool:
        changed = False
        for k, v in other.held.items():
            mine = self.held.get(k, 'absent')
            if mine == 'absent':
                self.held[k] = v
                changed = True
            elif mine is None and v is not None:
                self.held[k] = v
                changed = True
        return changed


def _run_block(func, block, state: _State, aliases, defs_at, check):
    """Advance *state* through *block*; with *check*, raise on a read of an
    overwritten alias."""
    insts = block.instructions
    for i in range(len(insts) + 1):
        for alias, read, loc in defs_at.get((block.block_id, i), ()):
            if read is not None:
                _read(state, read, loc, check)
            _define(state, alias, aliases, loc)
        if i == len(insts):
            break
        instr = insts[i]
        loc = instr.source_loc
        # Reads first: an instruction reads its operands before it writes.
        for _, v in iter_operands(instr, role=OperandRole.READ):
            if _hw(v) and v.alias is not None:
                _read(state, v.alias, loc, check)
        # Implicit overwrites.
        for reg in CHECKED:
            if _implicit_clobbers(instr, reg):
                _clobber(state, aliases, reg, loc)
        # Explicit hardware-register dests.
        for _, v in iter_operands(instr, role=OperandRole.WRITE):
            if not _hw(v) or v.name not in CHECKED:
                continue
            if isinstance(instr, Move) and _hw(instr.source, v.name):
                continue      # R = R: the register keeps what it held
            if v.alias is not None:
                _define(state, v.alias, aliases, loc)
            else:
                _clobber(state, aliases, v.name, loc)


def _read(state, alias, loc, check):
    where = state.held.get(id(alias), None)
    if check and where is not None:
        reg = alias.hardware_reg.name
        name = alias.symbol.name
        clobber_loc = where if where is not True else loc
        raise MIRLoweringError(
            f"'{name}' is bound to {reg}, but {reg} is overwritten here "
            f"before {name}'s last use",
            source_loc=clobber_loc,
            hint=f"copy '{name}' into a plain local before this line, "
                 f"or use it first",
        )


def _define(state, alias, aliases, loc):
    reg = alias.hardware_reg.name
    for k, other in aliases.items():
        if k != id(alias) and other.hardware_reg.name == reg and k in state.held:
            if state.held[k] is None:
                state.held[k] = loc if loc is not None else True
    state.held[id(alias)] = None


def _clobber(state, aliases, reg, loc):
    for k, alias in aliases.items():
        if alias.hardware_reg.name == reg and state.held.get(k, 'absent') is None:
            state.held[k] = loc if loc is not None else True


def check_register_aliases(func: MIRFunction):
    """Raise MIRLoweringError if *func* reads a register alias after its
    register was overwritten."""
    tracker = getattr(func, 'alias_tracker', None)
    if tracker is None:
        return
    aliases = {id(a): a for a in tracker.aliases.values()
               if a.hardware_reg.name in CHECKED}
    if not aliases:
        return

    defs_at: Dict = {}
    for alias, block_id, index, read, loc in tracker.def_points:
        defs_at.setdefault((block_id, index), []).append((alias, read, loc))

    blocks = func.blocks
    entry = func.entry_block_id
    in_state: Dict[int, _State] = {entry: _State()}
    work = [entry]
    while work:
        bid = work.pop()
        block = blocks.get(bid)
        if block is None:
            continue
        out = in_state[bid].copy()
        _run_block(func, block, out, aliases, defs_at, check=False)
        for s in block.successors:
            if s not in in_state:
                in_state[s] = out.copy()
                work.append(s)
            elif in_state[s].merge(out):
                work.append(s)

    for bid in sorted(in_state):
        block = blocks.get(bid)
        if block is not None:
            _run_block(func, block, in_state[bid].copy(), aliases, defs_at, check=True)
