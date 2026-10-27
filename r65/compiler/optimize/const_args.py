# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Constant stack arguments folded into the callee.

A method called on a single instance - `PLAYER.update()` everywhere, which
is `Player__update(&PLAYER)` - received `&PLAYER` on the stack at every
call, and read each field through it: a push and a cleanup per call, and
`LDY #off / LDA (sr,S),Y` per access. When every direct call passes a
stack parameter the same constant - a static's address or a number - the
callee materializes it itself: its entry block gets `Move %param = #PLAYER`,
the parameter leaves the signature and the calls drop the argument. The
parameter's vreg then has one definition, a constant, and an unindexed
access through it - `self.field` - becomes a load or store of the static
itself, `PLAYER+off`, which leaves the pointer unread.

A function qualifies only when every caller is visible: one with a direct
caller and no address taken, no trait dispatch reaching it, no `#[used]`
(a debugger may call it with anything), no recursion and no `asm!` (which
may read a parameter at its stack offset). A parameter
qualifies when the callee never writes it: the new definition sits in the
entry block, which a loop may come back to. The Pascal ABI is left alone,
since its callee pops the arguments.

Runs before dead code elimination, which removes the caller's `Move` of
the address once the call no longer reads it, and before `far_to_near`,
whose return-address adjustment applies to the offsets recomputed here.
"""

from typing import Dict, List, Optional, Set, Tuple

from r65.compiler.analysis.call_graph import build_call_graph
from r65.compiler.codegen.abi import ABIInfo
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.hir.types import PointerTypeInfo
from r65.compiler.mir.nodes import (
    ArgumentMechanism, Call, HardwareRegister, Immediate, InlineAsm, Load, LoadIndirect,
    MemoryLocation, MIRFunction, MIRProgram, Move, OperandRole, Store, StoreIndirect,
    VirtualRegister, iter_operands,
)

# A constant's identity: its value and the static it is an offset from.
_Key = Tuple[int, Optional[str]]


def _key(imm: Immediate) -> Optional[_Key]:
    """The constant `imm` stands for, or None for one this pass leaves alone
    (a function's or a local's address)."""
    if imm.symbol is None:
        return (imm.value, None) if isinstance(imm.value, int) else None
    from r65.compiler.hir.symbol_table import SymbolKind
    if getattr(imm.symbol, 'kind', None) is not SymbolKind.STATIC_VAR:
        return None
    return (imm.value, imm.symbol.name)


def _write_counts(func: MIRFunction) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for block in func.blocks.values():
        for instr in block.instructions:
            for _, v in iter_operands(instr, role=OperandRole.WRITE):
                if isinstance(v, VirtualRegister):
                    counts[v.id] = counts.get(v.id, 0) + 1
    return counts


def _constant_vregs(func: MIRFunction) -> Dict[int, Immediate]:
    """vreg id -> the constant it always holds: its one definition a Move of
    a constant, or a copy of a vreg that does. A parameter's vreg never
    qualifies; its entry value is a definition no instruction shows."""
    writes = _write_counts(func)
    params = {v.id for v in func.param_to_vreg.values() if v is not None}
    if func.self_y_vreg is not None:
        params.add(func.self_y_vreg.id)

    consts: Dict[int, Immediate] = {}
    copies = []
    for block in func.blocks.values():
        for instr in block.instructions:
            if not (isinstance(instr, Move) and isinstance(instr.dest, VirtualRegister)
                    and writes.get(instr.dest.id) == 1 and instr.dest.id not in params):
                continue
            if isinstance(instr.source, Immediate) and _key(instr.source) is not None:
                consts[instr.dest.id] = instr.source
            elif isinstance(instr.source, VirtualRegister):
                copies.append((instr.dest.id, instr.source.id))
    changed = True
    while changed:
        changed = False
        for dest, src in copies:
            if dest not in consts and src in consts:
                consts[dest] = consts[src]
                changed = True
    return consts


class ConstantArgFolder:
    """Fold stack arguments every call passes the same constant into the callee."""

    def __init__(self, abi_kind=None, verbose: bool = False):
        self.abi_kind = abi_kind
        self.verbose = verbose

    def optimize(self, mir_program: MIRProgram) -> int:
        """Fold every qualifying parameter. Returns how many were folded.

        Repeats until nothing changes: a method that hands its now-constant
        `self` to another method makes that call's argument constant too.
        """
        from r65.compiler.codegen.abi_model import ABIKind
        if self.abi_kind == ABIKind.PASCAL:
            return 0
        total = 0
        while True:
            folded = self._round(mir_program)
            if not folded:
                return total
            total += folded

    def _round(self, mir_program: MIRProgram) -> int:
        graph = build_call_graph(mir_program)
        recursive = graph.recursive_functions()
        funcs = {f.name: f for f in mir_program.functions}
        consts = {f.name: _constant_vregs(f) for f in mir_program.functions}

        calls: Dict[str, List[Tuple[MIRFunction, Call]]] = {}
        for func in mir_program.functions:
            for block in func.blocks.values():
                for instr in block.instructions:
                    if (isinstance(instr, Call) and isinstance(instr.function, str)
                            and instr.builtin_name is None and instr.function in funcs):
                        calls.setdefault(instr.function, []).append((func, instr))

        folded = 0
        for name, sites in calls.items():
            callee = funcs[name]
            if (name in graph.address_taken or name in graph.dispatch_targets
                    or name in recursive or callee.is_trait_method or callee.is_used
                    or callee.is_entry or callee.interrupt_attr is not None
                    or any(isinstance(i, InlineAsm)
                           for b in callee.blocks.values() for i in b.instructions)):
                continue
            indices = self._foldable(callee, sites, consts)
            if indices:
                self._fold(callee, sites, indices)
                folded += len(indices)
                if self.verbose:
                    print(f"  {name}: folded {len(indices)} constant argument(s)")
        return folded

    @staticmethod
    def _foldable(callee: MIRFunction, sites, consts) -> Dict[int, Immediate]:
        """Parameter index -> the constant every call passes it."""
        writes = _write_counts(callee)
        out: Dict[int, Immediate] = {}
        for idx, vreg in callee.param_to_vreg.items():
            if vreg is None or writes.get(vreg.id) or idx not in callee.stack_param_offsets:
                continue
            size = get_type_size(callee.parameters[idx].param_type)
            seen: Optional[Immediate] = None
            for caller, call in sites:
                if idx >= len(call.args) or call.args[idx].mechanism is not ArgumentMechanism.STACK:
                    seen = None
                    break
                value = call.args[idx].value
                if isinstance(value, VirtualRegister):
                    if get_type_size(value.type_info) != size:
                        seen = None
                        break
                    imm = consts[caller.name].get(value.id)
                elif isinstance(value, Immediate):
                    imm = value
                else:
                    imm = None
                if imm is None or _key(imm) is None or (seen is not None and _key(imm) != _key(seen)):
                    seen = None
                    break
                seen = imm
            if seen is not None:
                out[idx] = seen
        return out

    @staticmethod
    def _fold(callee: MIRFunction, sites, indices: Dict[int, Immediate]) -> None:
        # The callee defines each parameter itself, after the entry block's
        # saves of register parameters.
        entry = callee.blocks[callee.entry_block_id]
        at = 0
        while (at < len(entry.instructions)
               and isinstance(entry.instructions[at], Move)
               and isinstance(entry.instructions[at].source, HardwareRegister)):
            at += 1
        for idx, imm in sorted(indices.items()):
            param = callee.parameters[idx]
            entry.instructions.insert(at, Move(
                dest=callee.param_to_vreg[idx],
                source=Immediate(imm.value, symbol=imm.symbol),
                type_info=param.param_type,
            ))
            at += 1

        # Drop the parameters, renumbering what is keyed by index.
        def renumber(idx: int) -> int:
            return idx - sum(1 for gone in indices if gone < idx)

        callee.parameters = [p for i, p in enumerate(callee.parameters) if i not in indices]
        callee.param_to_vreg = {renumber(i): v for i, v in callee.param_to_vreg.items()
                                if i not in indices}
        kept = sorted(renumber(i) for i in callee.stack_param_offsets if i not in indices)
        offset = ABIInfo(is_far=callee.is_far).return_addr_size + 1
        callee.stack_param_offsets = {}
        for idx in kept:
            callee.stack_param_offsets[idx] = offset
            offset += get_type_size(callee.parameters[idx].param_type)
        callee.far_ptr_param_indices = {renumber(i) for i in callee.far_ptr_param_indices
                                        if i not in indices}
        callee.fn_ptr_param_indices = {renumber(i) for i in callee.fn_ptr_param_indices
                                       if i not in indices}
        callee.has_far_ptr_stack_params = any(
            isinstance(callee.parameters[i].param_type, PointerTypeInfo)
            and callee.parameters[i].param_type.is_far
            for i in kept)

        for _, call in sites:
            call.args = [a for i, a in enumerate(call.args) if i not in indices]


def _static_location(imm: Immediate, is_far: bool, offset: int) -> Optional[MemoryLocation]:
    """The static `imm` points into, at `offset` bytes further, as a direct
    location - or None when an access through the pointer would not reach
    the same bytes. A far pointer holds the static's own address. A near one
    reads the data bank, which reaches a low-RAM, direct-page or I/O static
    at its address as direct addressing does; a `#[ram]` or ROM static it
    reaches only while the data bank is that static's."""
    from r65.compiler.hir.symbol_table import SymbolKind
    symbol = imm.symbol
    if symbol is None or getattr(symbol, 'kind', None) is not SymbolKind.STATIC_VAR:
        return None
    decl = symbol.definition
    if getattr(decl, 'is_extern', False):
        return None
    attr = getattr(decl, 'storage_attr', None)
    if attr is not None:
        storage, address = attr.storage_kind.value, attr.address
    elif not getattr(decl, 'is_mutable', True):
        storage, address = 'rom', None
    else:
        return None
    if not is_far and storage not in ('zeropage', 'lowram', 'hw'):
        return None
    offset += imm.value
    if address is not None:
        return MemoryLocation(storage_type=storage, address=address + offset, symbol=symbol)
    return MemoryLocation(storage_type=storage, address=None, symbol=symbol, offset=offset)


def direct_constant_pointer_accesses(func: MIRFunction) -> int:
    """Rewrite each unindexed load or store through a pointer that always
    holds a static's address into a load or store of that static. Returns
    how many were rewritten."""
    consts = _constant_vregs(func)
    if not consts:
        return 0
    rewritten = 0
    for block in func.blocks.values():
        for i, instr in enumerate(block.instructions):
            if (not isinstance(instr, (LoadIndirect, StoreIndirect))
                    or instr.index_register is not None
                    or instr.pointer.id not in consts):
                continue
            loc = _static_location(consts[instr.pointer.id], instr.is_far, instr.offset)
            if loc is None:
                continue
            if isinstance(instr, LoadIndirect):
                block.instructions[i] = Load(dest=instr.dest, source=loc,
                                             type_info=instr.type_info)
            else:
                block.instructions[i] = Store(source=instr.source, dest=loc,
                                              type_info=instr.type_info)
            rewritten += 1
    return rewritten


def fold_constant_args(mir_program: MIRProgram, abi_kind=None, verbose: bool = False) -> int:
    """Run the pass; see the module docstring. Returns how many arguments
    were folded."""
    folded = ConstantArgFolder(abi_kind=abi_kind, verbose=verbose).optimize(mir_program)
    for func in mir_program.functions:
        direct_constant_pointer_accesses(func)
    return folded
