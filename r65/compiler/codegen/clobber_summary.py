# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Which registers a run of emitted code can change.

The one reading of emitted code for its register writes. A function's
summary is what its body writes, its callees' summaries included:

- X and Y are caller-saved, so a value held in one across a call is pushed
  before it and pulled after. A callee whose summary leaves the register out
  leaves it as it was, and the save is moot (mir/call_effects `inferred`).
- An interrupt handler saves what its body writes, and nothing else.

Functions are generated callees first (`callee_first`), so each call site
sees the finished code of what it calls.
"""

from typing import Callable, Dict, FrozenSet, Iterable, List, Optional

from r65.compiler.codegen.asm_nodes import Address, Instruction, RawAsm
from r65.compiler.codegen.opcodes import (
    ACCUMULATOR_OPCODES, Opcode, addressing_mode, mnemonic, registers_written,
)
from r65.compiler.mir.nodes import Call, MIRFunction

# The registers a summary speaks for. B goes with A; P, S and PBR do not hold
# values across code.
WRITABLE: FrozenSet[str] = frozenset({'A', 'X', 'Y', 'D', 'DBR'})

# Control leaves for code whose effect is unknown.
_UNKNOWN = frozenset({'BRK', 'COP', 'WDM', 'STP'})
_CALLS = frozenset({'JSR', 'JSL'})
_JUMPS = frozenset({'JMP', 'JML'})
_OPCODES = {mnemonic(op): op for op in Opcode if addressing_mode(op) is None}
_MNEMONICS = frozenset(mnemonic(op) for op in Opcode)


def _raw_parts(text: str):
    """*text*'s mnemonic, without a `.b`/`.w`/`.l` size suffix, and operand."""
    parts = text.split(';', 1)[0].split(None, 1)
    if not parts:
        return None, None
    mn = parts[0].upper()
    if not mn.startswith('.') and '.' in mn:
        mn = mn.split('.', 1)[0]
    return mn, (parts[1].strip() if len(parts) > 1 else None)


def _raw_writes(text: str) -> Optional[FrozenSet[str]]:
    """What one line of raw text writes, or None for a call, a jump or text
    that is not an instruction - code whose effect it cannot say."""
    mn, operand = _raw_parts(text)
    if mn is None or mn.startswith('.') or mn.endswith(':'):
        return frozenset()
    if mn not in _MNEMONICS or mn in _CALLS or mn in _JUMPS or mn in _UNKNOWN:
        return None
    # The implied/accumulator form when there is no operand (or it is `A`);
    # otherwise any form of the mnemonic - their writes agree, and a SEP's
    # mask is not read, so it may set x8.
    accumulator = operand is None or operand.upper() == 'A'
    op = _OPCODES.get(mn) if accumulator else None
    if op is None:
        op = next(o for o in Opcode if mnemonic(o) == mn
                  and (o not in ACCUMULATOR_OPCODES or accumulator))
    return registers_written(op)


def code_writes(nodes: Iterable, summaries: Optional[Dict[str, FrozenSet[str]]] = None,
                functions: FrozenSet[str] = frozenset(),
                skip_user_asm: bool = False) -> FrozenSet[str]:
    """The registers among A, X, Y, D and DBR running *nodes* may change.

    *summaries* maps a function label to what its code changes, and
    *functions* names every function label. A call or tail jump to a
    summarized function takes its summary; to anything else (a runtime
    helper, a dispatch wrapper, a function generated later - the cycle
    case) or an indirect target, everything. A jump to a label that is not
    a function stays in the function and writes nothing. `asm!` text is
    never read: it writes everything, unless *skip_user_asm* passes over it
    for a function whose `#[clobbers(...)]` speaks for it.
    """
    summaries = summaries or {}
    written = set()
    for node in nodes:
        if isinstance(node, Instruction):
            mn = mnemonic(node.opcode)
            if mn in _CALLS or mn in _JUMPS:
                operand = node.operand
                target = operand.value if isinstance(operand, Address) else None
                indirect = addressing_mode(node.opcode) not in (None, 'ABSOLUTE', 'LONG')
                if indirect or not isinstance(target, str):
                    return WRITABLE
                if mn in _JUMPS and target not in functions:
                    continue
                if target not in summaries:
                    return WRITABLE
                written |= summaries[target]
                continue
            if mn in _UNKNOWN:
                return WRITABLE
            written |= registers_written(node.opcode, node.operand)
        elif isinstance(node, RawAsm):
            # `asm!`, or one instruction codegen spelled as text, or a directive
            if node.user_asm:
                if skip_user_asm:
                    continue
                return WRITABLE
            regs = _raw_writes(node.text)
            if regs is None:
                return WRITABLE
            written |= regs
        if written >= WRITABLE:
            break
    return frozenset(written) & WRITABLE


def declared_writes(attr) -> FrozenSet[str]:
    """What a `#[clobbers(...)]` (*attr*, None for none) says its function
    writes: A always, and the registers it lists."""
    listed = frozenset(attr.registers) if attr is not None else frozenset()
    return frozenset({'A'}) | (listed & WRITABLE)


def function_summary(func: MIRFunction, body: list,
                     summaries: Dict[str, FrozenSet[str]],
                     functions: FrozenSet[str]) -> FrozenSet[str]:
    """What *func*'s code, *body*, may change.

    What the code writes, read from it, plus whatever its `#[clobbers(...)]`
    lists. `asm!` text is never read: without the attribute it writes
    everything, with it the attribute speaks for the text and the rest is
    still read.
    """
    if func.clobbers_attr is None:
        return code_writes(body, summaries, functions)
    return (code_writes(body, summaries, functions, skip_user_asm=True)
            | declared_writes(func.clobbers_attr))


def extern_summaries(functions: Iterable[MIRFunction]) -> Dict[str, FrozenSet[str]]:
    """Each `extern fn` *functions* call, mapped to what its `#[clobbers(...)]`
    says it writes - its whole contract, as no code of it is seen."""
    found: Dict[str, FrozenSet[str]] = {}
    for func in functions:
        for block in func.blocks.values():
            for instr in block.instructions:
                if (type(instr) is Call and instr.callee_is_extern
                        and isinstance(instr.function, str)):
                    found[instr.function] = declared_writes(instr.clobbers_attr)
    return found


def callee_first(functions: List[MIRFunction],
                 callees_of: Callable[[str], Iterable[str]]) -> List[MIRFunction]:
    """*functions* with each one after the functions it calls (*callees_of*
    a name: the call graph's edges, a dispatch's implementors included).

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
        stack = [(root, iter(sorted(callees_of(root.name))))]
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
                stack.append((nxt, iter(sorted(callees_of(nxt.name)))))
    return order


def check_after_peephole(nodes: Iterable, ends: Dict[str, object],
                         summaries: Dict[str, FrozenSet[str]],
                         functions: FrozenSet[str],
                         declared: FrozenSet[str] = frozenset()):
    """Re-derive each summary from the optimized code.

    Callers dropped their saves on the summaries taken before the peephole
    ran, so it must not have added an index write to any function. *ends*
    maps each function to its last node before the peephole; a body runs
    from its label to that node (or the next function label, if it is gone).
    *declared* names the functions with a `#[clobbers(...)]`, which speaks
    for their `asm!` text here too.
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
        grown = (code_writes(body, summaries, functions,
                             skip_user_asm=name in declared)
                 - summaries[name])
        if grown:
            raise CodegenError(
                f"internal: peephole added a write to {sorted(grown)} in "
                f"'{name}', whose callers keep it unsaved")

