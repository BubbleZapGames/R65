# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Which index registers a function's emitted code can change.

X and Y are caller-saved, so a value held in one across a call is pushed
before it and pulled after. A callee whose code - and every callee of its
own - never writes the register leaves it as it was, and the save is moot.
Functions are generated callees first (`callee_first`), so each call site
sees the finished code of what it calls.
"""

from typing import Dict, FrozenSet, Iterable, List, Optional

from r65.compiler.codegen.asm_nodes import Address, Immediate, Instruction, RawAsm
from r65.compiler.codegen.constants import X_FLAG
from r65.compiler.codegen.opcodes import REGISTER_WRITES, Opcode, addressing_mode, mnemonic
from r65.compiler.mir.call_effects import INDEX_REGS
from r65.compiler.mir.nodes import Call, InlineAsm, MIRFunction

# Control leaves for code whose effect is unknown.
_UNKNOWN = frozenset({'BRK', 'COP', 'WDM', 'XCE', 'STP'})
_CALLS = frozenset({'JSR', 'JSL'})
_JUMPS = frozenset({'JMP', 'JML'})
_MNEMONICS = frozenset(mnemonic(op) for op in Opcode)


def _raw_parts(text: str):
    parts = text.split(';', 1)[0].split(None, 1)
    if not parts:
        return None, None
    return parts[0].upper(), (parts[1].strip() if len(parts) > 1 else None)


def index_writes(nodes: Iterable, summaries: Dict[str, FrozenSet[str]],
                 functions: FrozenSet[str]) -> FrozenSet[str]:
    """The index registers running *nodes* may change.

    *summaries* maps a function label to what its code changes, and
    *functions* names every function label. A call or tail jump to a
    summarized function takes its summary; to anything else (a runtime
    helper, a dispatch wrapper, a function generated later - the cycle
    case) or an indirect target, both registers. A jump to a
    label that is not a function stays in the function and writes nothing.
    """
    written = set()
    for node in nodes:
        if isinstance(node, Instruction):
            mn = mnemonic(node.opcode)
            operand = node.operand
            if mn in _CALLS or mn in _JUMPS:
                target = operand.value if isinstance(operand, Address) else None
                indirect = addressing_mode(node.opcode) not in (None, 'ABSOLUTE', 'LONG')
                if indirect or not isinstance(target, str):
                    return INDEX_REGS
                if mn in _JUMPS and target not in functions:
                    continue
                if target not in summaries:
                    return INDEX_REGS
                written |= summaries[target]
                continue
            if node.opcode == Opcode.SEP_IMMEDIATE:
                mask = operand.value if isinstance(operand, Immediate) else None
                if not isinstance(mask, int) or mask & X_FLAG:
                    written |= INDEX_REGS     # x8 drops the high bytes
                continue
        elif isinstance(node, RawAsm):
            # One instruction codegen spelled as text (`asm!` is ruled out
            # from the MIR: `has_inline_asm`), or a directive.
            mn, _ = _raw_parts(node.text)
            if mn is None or mn.startswith('.') or mn.endswith(':'):
                continue
            if mn not in _MNEMONICS or mn in _CALLS or mn in _JUMPS:
                return INDEX_REGS
            if mn == 'SEP':
                written |= INDEX_REGS     # mask unread: assume it sets x8
                continue
        else:
            continue
        if mn in _UNKNOWN:
            return INDEX_REGS
        written |= REGISTER_WRITES.get(mn, frozenset()) & INDEX_REGS
        if written == INDEX_REGS:
            break
    return frozenset(written)


def has_inline_asm(func: MIRFunction) -> bool:
    """Whether *func* runs `asm!` text, which is never read for its writes."""
    return any(isinstance(instr, InlineAsm)
               for block in func.blocks.values() for instr in block.instructions)


def direct_callees(func: MIRFunction) -> List[str]:
    """The functions *func* calls by name, in first-call order."""
    seen: Dict[str, None] = {}
    for block in func.blocks.values():
        for instr in block.instructions:
            if (type(instr) is Call and isinstance(instr.function, str)
                    and not instr.builtin_name):
                seen.setdefault(instr.function, None)
    return list(seen)


def callee_first(functions: List[MIRFunction]) -> List[MIRFunction]:
    """*functions* with each one after the functions it calls.

    Depth-first post-order from each function in its given order, so
    unrelated functions keep their relative order. A cycle is broken where
    the search first re-enters it; its members then see one another as
    not yet generated, which is the conservative answer.
    """
    by_name = {f.name: f for f in functions}
    done: Dict[str, None] = {}
    order: List[MIRFunction] = []
    for root in functions:
        if root.name in done:
            continue
        done[root.name] = None
        stack = [(root, iter(direct_callees(root)))]
        while stack:
            func, callees = stack[-1]
            nxt: Optional[MIRFunction] = None
            for name in callees:
                callee = by_name.get(name)
                if callee is not None and name not in done:
                    nxt = callee
                    break
            if nxt is None:
                stack.pop()
                order.append(func)
            else:
                done[nxt.name] = None
                stack.append((nxt, iter(direct_callees(nxt))))
    return order


def check_after_peephole(nodes: Iterable, ends: Dict[str, object],
                         summaries: Dict[str, FrozenSet[str]],
                         functions: FrozenSet[str]):
    """Re-derive each summary from the optimized code.

    Callers dropped their saves on the summaries taken before the peephole
    ran, so it must not have added an index write to any function. *ends*
    maps each function to its last node before the peephole; a body runs
    from its label to that node (or the next function label, if it is gone).
    """
    from r65.compiler.codegen.asm_nodes import Label
    from r65.compiler.errors import CodegenError
    bodies: Dict[str, list] = {}
    name = None
    for node in nodes:
        if isinstance(node, Label) and node.name in functions:
            name = node.name if node.name in summaries else None
        if name is not None:
            bodies.setdefault(name, []).append(node)
            if node is ends.get(name):
                name = None
    for name, body in bodies.items():
        grown = index_writes(body, summaries, functions) - summaries[name]
        if grown:
            raise CodegenError(
                f"internal: peephole added a write to {sorted(grown)} in "
                f"'{name}', whose callers keep it unsaved")

