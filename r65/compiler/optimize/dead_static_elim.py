# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Dead Static Elimination.

Removes statics no live code reads, writes or takes the address of, with their
`__init_start` code and ROM data, so a static pulled in from a package costs
nothing unless it is used. Runs after dead function elimination: references
are counted from the functions that survived it.

Always kept:
- `#[hw]` and `extern` statics: they own no bytes, and asm may name them.
- A static at an explicit address: a deliberate placement, often for code the
  compiler cannot see (a debugger, a DMA list, another program).
- A scratch `register` static: the compiler's own temporaries.
- A `#[used]` static.

An `#[soa]` static and its planes, and the planes laid over them, live or die
as one: the planes are one static's storage.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from r65.compiler.hir.attributes import StorageKind
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, MemoryLocation, Store, BlockCopy, MemoryFill,
    Return, SymbolByte,
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INIT = "__init_start"


class DeadStaticEliminator:
    """Eliminates statics nothing live refers to."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def eliminate(self, mir_program: MIRProgram) -> int:
        """Prune `mir_program.statics`, their init code and their ROM data.

        Returns the number of statics removed.
        """
        if not any(func.is_entry for func in mir_program.functions):
            return 0                    # no roots: a unit test of a fragment

        refs = _References(mir_program.statics)
        if refs.blind:
            return 0
        live: Set[str] = {name for name, decl in refs.decls.items() if _pinned(decl)}

        # Code: every function but __init_start, whose stores belong to the
        # statics they initialize.
        init = None
        for func in mir_program.functions:
            if func.name == _INIT:
                init = func
            else:
                live |= refs.scan(func)
        segments = _init_segments(init, refs) if init is not None else None
        if init is not None and segments is None:
            live |= refs.scan(init)     # not the shape the lowerer emits
        edges: Dict[str, Set[str]] = {}
        for owner, instrs in (segments or {}).items():
            if owner is None:
                live |= refs.scan(instrs)
            else:
                edges.setdefault(owner, set()).update(refs.scan(instrs) - {owner})

        # ROM data: a table belongs to its static; one with no owner is a root.
        owned_rom: Dict[str, List] = {}
        for rom_data in mir_program.rom_data_sections:
            owner = refs.label_owner.get(rom_data.label)
            targets = {refs.label_owner[byte.label] for byte in rom_data.data
                       if isinstance(byte, SymbolByte) and byte.label in refs.label_owner}
            if owner is None:
                live |= targets
            else:
                owned_rom.setdefault(owner, []).append(rom_data)
                edges.setdefault(owner, set()).update(targets - {owner})

        for path in mir_program.asm_includes:
            text = _read(path)
            if text is None:
                return 0                # an include we cannot read may name any static
            live |= refs.names_in(text)

        # Close over initializers, ROM tables and plane groups.
        work = list(live)
        while work:
            name = work.pop()
            for other in edges.get(name, set()) | refs.group(name):
                if other not in live:
                    live.add(other)
                    work.append(other)

        dead = set(refs.decls) - live
        if not dead:
            return 0
        mir_program.statics = [d for d in mir_program.statics
                               if d.symbol.name not in dead]
        dead_rom = {id(r) for name in dead for r in owned_rom.get(name, ())}
        mir_program.rom_data_sections = [r for r in mir_program.rom_data_sections
                                         if id(r) not in dead_rom]
        if segments is not None:
            block = init.blocks[init.entry_block_id]
            dropped = {id(i) for name in dead for i in segments.get(name, ())}
            block.instructions = [i for i in block.instructions if id(i) not in dropped]

        if self.verbose:
            print(f"Dead static elimination: removed {len(dead)} static(s)")
            for name in sorted(dead):
                print(f"  - {name}")
        return len(dead)


def _pinned(decl) -> bool:
    """Whether *decl* is kept whether or not anything refers to it."""
    attr = decl.storage_attr
    return (getattr(decl, 'is_extern', False) or getattr(decl, 'is_used', False)
            or (attr is not None and (attr.storage_kind == StorageKind.HW
                                      or attr.address is not None
                                      or attr.is_register)))


class _References:
    """The names a piece of MIR refers to, among the program's statics."""

    def __init__(self, statics):
        self.decls = {d.symbol.name: d for d in statics if getattr(d, 'symbol', None)}
        self.blind = len(self.decls) != len(statics)
        self.by_symbol = {id(d.symbol): name for name, d in self.decls.items()}
        # Any label a static's bytes go by, to the static.
        self.label_owner: Dict[str, str] = {}
        for name, decl in self.decls.items():
            self.label_owner[name] = name
            self.label_owner[f"__{name}_data"] = name
            self.label_owner[f"{name}_data"] = name     # an include_bytes! blob
            rom_label = getattr(decl.symbol, 'rom_label', None)
            if rom_label:
                self.label_owner[rom_label] = name
        self._groups = self._plane_groups()

    def _plane_groups(self) -> Dict[str, Set[str]]:
        """Each planed static's group: parent, planes, and the planes over them."""
        parent = {name: name for name in self.decls}

        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        def union(a, b):
            if a in parent and b in parent:
                parent[find(a)] = find(b)

        for name, decl in self.decls.items():
            for plane in getattr(decl, 'soa_planes', None) or ():
                union(name, plane.child.symbol.name)
            overlay = getattr(decl, 'soa_overlay', None)
            if overlay is not None:
                union(name, overlay[0].symbol.name)
        groups: Dict[str, Set[str]] = {}
        for name in self.decls:
            groups.setdefault(find(name), set()).add(name)
        return {name: groups[find(name)] for name in self.decls}

    def group(self, name: str) -> Set[str]:
        return self._groups.get(name, set())

    def names_in(self, text: str) -> Set[str]:
        return {self.label_owner[t] for t in _IDENT.findall(text) if t in self.label_owner}

    def scan(self, root) -> Set[str]:
        """Statics named anywhere in *root*: a symbol, a label, asm text."""
        found: Set[str] = set()
        seen: Set[int] = set()
        stack = [root]
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            name = self.by_symbol.get(id(obj))
            if name is not None:
                found.add(name)
                continue
            if isinstance(obj, str):
                found |= self.names_in(obj)
            elif isinstance(obj, (list, tuple, set, frozenset)):
                stack.extend(obj)
            elif isinstance(obj, dict):
                stack.extend(obj.keys())
                stack.extend(obj.values())
            elif hasattr(obj, '__dict__') and _is_mir(obj):
                stack.extend(vars(obj).values())
        return found


def _is_mir(obj) -> bool:
    """MIR objects are walked; HIR symbols, types and nodes are not (they
    reach the whole program), and are matched by identity instead."""
    module = type(obj).__module__
    return module.startswith('r65.compiler.mir') or module.startswith('r65.compiler.codegen')


def _init_segments(init: MIRFunction, refs: _References) -> Optional[Dict[Optional[str], list]]:
    """`__init_start`'s instructions by the static each initializes.

    The lowerer emits one straight-line block: per static, the instructions
    computing its value, then the Store, BlockCopy or MemoryFill into it.
    Anything else returns None, and the whole function counts as live code.
    """
    if len(init.blocks) != 1:
        return None
    segments: Dict[Optional[str], list] = {}
    pending: list = []
    for instr in init.blocks[init.entry_block_id].instructions:
        if isinstance(instr, Return):
            continue
        pending.append(instr)
        dest = getattr(instr, 'dest', None)
        if isinstance(instr, (Store, BlockCopy, MemoryFill)) and isinstance(dest, MemoryLocation):
            owner = refs.by_symbol.get(id(dest.symbol))
            segments.setdefault(owner, []).extend(pending)
            pending = []
    if pending:
        segments.setdefault(None, []).extend(pending)
    return segments


def _read(include) -> Optional[str]:
    try:
        return Path(include.path).read_text(errors='replace')
    except (OSError, AttributeError):
        return None
