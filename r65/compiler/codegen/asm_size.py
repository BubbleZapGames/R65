# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Byte sizes of assembly written as text.

An `Instruction` the compiler emits is sized exactly from its Opcode. Text is
not: `asm!()` blocks, the inline tables `AssemblyEmitter.emit_directive` writes as
`RawAsm`, the entry stub, and data `Directive`s. Branch fixup and address
calculation both lay out code that contains it, and size it here so they
agree.
"""

import re
from typing import Mapping, Optional, Sequence

from r65.compiler.codegen.opcodes import Opcode, addressing_mode, mnemonic

# Bytes per argument of a data directive.
_DATA_WIDTHS = {
    '.DB': 1, '.BYTE': 1,
    '.DW': 2, '.WORD': 2,
    '.DL': 3, '.LONG': 3, '.FARADDR': 3,
    '.DD': 4, '.DWORD': 4,
}

_MNEMONICS = frozenset(mnemonic(op) for op in Opcode) | {'JML'}
_RELATIVE = frozenset({'BCC', 'BCS', 'BEQ', 'BMI', 'BNE', 'BPL', 'BVC', 'BVS', 'BRA'})
_RELATIVE_LONG = frozenset({'BRL', 'PER'})
# Immediates sized by the accumulator width, and by the index width.
_A_IMMEDIATE = frozenset({'ADC', 'AND', 'BIT', 'CMP', 'EOR', 'LDA', 'ORA', 'SBC'})
_INDEX_IMMEDIATE = frozenset({'CPX', 'CPY', 'LDX', 'LDY'})

# The addressing modes each mnemonic has. JSR and PEI are Opcodes without
# a mode suffix.
_MODES: dict = {}
for _op in Opcode:
    _MODES.setdefault(mnemonic(_op), set()).add(
        addressing_mode(_op) or {'JSR': 'ABSOLUTE', 'PEI': 'DP_INDIRECT'}.get(_op.name))

# Operand shape -> its modes, narrowest address first.
_SHAPES = (
    (re.compile(r'^\((.*),S\),Y$', re.IGNORECASE), (('STACK_INDIRECT_Y', 1),)),
    (re.compile(r'^\((.*)\),Y$', re.IGNORECASE), (('DP_INDIRECT_Y', 1),)),
    (re.compile(r'^\((.*),X\)$', re.IGNORECASE), (('DP_INDIRECT_X', 1), ('INDIRECT_X', 2))),
    (re.compile(r'^\((.*)\)$', re.IGNORECASE), (('DP_INDIRECT', 1), ('INDIRECT', 2))),
    (re.compile(r'^\[(.*)\],Y$', re.IGNORECASE), (('DP_INDIRECT_LONG_Y', 1),)),
    (re.compile(r'^\[(.*)\]$', re.IGNORECASE), (('DP_INDIRECT_LONG', 1), ('INDIRECT_LONG', 2))),
    (re.compile(r'^(.*),S$', re.IGNORECASE), (('STACK', 1),)),
    (re.compile(r'^(.*),X$', re.IGNORECASE), (('DP_X', 1), ('ABSOLUTE_X', 2), ('LONG_X', 3))),
    (re.compile(r'^(.*),Y$', re.IGNORECASE), (('DP_Y', 1), ('ABSOLUTE_Y', 2))),
    (re.compile(r'^(.*)$', re.IGNORECASE), (('DP', 1), ('ABSOLUTE', 2), ('LONG', 3))),
)
_TERM = re.compile(r'\s*([+-]?)\s*([^+\-\s]+)')
_NAME = re.compile(r'^[A-Za-z_.@][A-Za-z0-9_.@]*$', re.IGNORECASE)
_LABEL = object()   # a name WLA-DX resolves only at link time


def data_directive_size(name: str, args: Sequence[str]) -> int:
    """Bytes a data directive emits; 0 for any other directive."""
    name = name.upper()
    width = _DATA_WIDTHS.get(name)
    if width is not None:
        size = 0
        for arg in (str(a).strip() for a in args):
            if len(arg) >= 2 and arg[0] == arg[-1] == '"':
                size += len(arg) - 2      # a string's characters, one byte each
            elif arg:
                size += width
        return size
    if name in ('.DSB', '.DSW') and args:
        arg = str(args[0]).strip()
        try:
            count = int(arg[1:], 16) if arg.startswith('$') else int(arg, 0)
        except ValueError:
            return 0
        return count * (2 if name == '.DSW' else 1)
    return 0


def raw_asm_size(text: str, m16: bool = False, x16: bool = True,
                 symbols: Optional[Mapping[str, int]] = None) -> int:
    """Bytes the lines of *text* assemble to, in the given register widths."""
    return sum(asm_line_size(line, m16, x16, symbols) for line in text.splitlines())


def asm_line_size(line: str, m16: bool = False, x16: bool = True,
                  symbols: Optional[Mapping[str, int]] = None) -> int:
    """Bytes one line of assembly text assembles to.

    Comments, labels, non-data directives and WLA-DX's header keywords
    (`LOROM`, `NAME`, ...) are 0. An unsuffixed operand is sized as WLA-DX
    sizes it: a value known when assembling (a number, or a `.DEFINE` /
    `.EQU` in *symbols*) takes the narrowest mode that holds it, so `$0010`
    is direct page; a label, resolved only at link time, takes the narrowest
    mode the instruction has, so `(ptr),Y` is direct page and `LDA buf,Y`
    absolute. Without *symbols* a name may be a `.DEFINE` of any width, and
    is sized as absolute unless the mode has one width.
    """
    code = line.split(';', 1)[0].strip()
    if not code or code.endswith(':'):
        return 0
    parts = code.split(None, 1)
    word = parts[0].upper()
    operand = parts[1].strip() if len(parts) > 1 else ''

    if word.startswith('.'):
        return data_directive_size(word, operand.split(',')) if operand else 0

    mnem, _, suffix = word.partition('.')
    if mnem not in _MNEMONICS:
        return 0
    if suffix:
        return {'B': 2, 'W': 3, 'L': 4}.get(suffix, 3)
    if not operand or operand.upper() == 'A':
        return 2 if mnem == 'WDM' else 1
    if mnem in _RELATIVE:
        return 2
    if mnem in _RELATIVE_LONG or mnem in ('MVN', 'MVP', 'PEA'):
        return 3
    if mnem in ('JSL', 'JML') and not operand.startswith('['):
        return 4
    if operand.startswith('#'):
        if mnem in _A_IMMEDIATE:
            return 3 if m16 else 2
        if mnem in _INDEX_IMMEDIATE:
            return 3 if x16 else 2
        return 2
    size = _mode_address_bytes(mnem, operand.replace(' ', ''), symbols)
    return 1 + (size if size is not None else _address_bytes(operand))


def _mode_address_bytes(mnem: str, operand: str,
                        symbols: Optional[Mapping[str, int]]) -> Optional[int]:
    """Address bytes of the narrowest mode *mnem* has for *operand*'s
    shape that fits its value, or None when that is not known."""
    modes = _MODES.get(mnem)
    if not modes:
        return None
    for pattern, shape in _SHAPES:
        match = pattern.match(operand)
        if match:
            break
    available = [width for mode, width in shape if mode in modes]
    if not available:
        return None
    if len(available) == 1:
        return available[0]
    value = _operand_value(match.group(1), symbols)
    if value is None:
        return None
    need = 1 if value is _LABEL else 1 if value <= 0xFF else 2 if value <= 0xFFFF else 3
    return next((w for w in available if w >= need), None)


def _operand_value(expr: str, symbols: Optional[Mapping[str, int]]):
    """The value of a `+`/`-` chain of numbers and names, _LABEL when a
    name is not in *symbols*, or None when it cannot be told."""
    pos, total, label = 0, 0, False
    for match in _TERM.finditer(expr):
        if match.start() != pos:
            return None
        pos = match.end()
        sign, term = match.groups()
        value = _number(term)
        if value is None:
            if not _NAME.match(term) or symbols is None:
                return None
            value = symbols.get(term)
            if value is None:
                label = True
                continue
        total += -value if sign == '-' else value
    if pos != len(expr) or not expr:
        return None
    return _LABEL if label else total & 0xFFFFFF


def _number(term: str) -> Optional[int]:
    try:
        if term.startswith('$'):
            return int(term[1:], 16)
        if term.startswith('%'):
            return int(term[1:], 2)
        if term.isdigit():
            return int(term)
    except ValueError:
        return None
    return None


def _address_bytes(operand: str) -> int:
    """Bytes of the address in *operand*, whatever indexing wraps it."""
    inner = operand.lstrip('([').split(',')[0].rstrip(')]').strip()
    if inner.startswith('$'):
        digits = len(inner[1:])
        return 1 if digits <= 2 else 2 if digits <= 4 else 3
    if inner.isdigit():
        value = int(inner)
        return 1 if value <= 0xFF else 2 if value <= 0xFFFF else 3
    return 2
