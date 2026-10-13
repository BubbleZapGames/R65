# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Byte sizes of assembly written as text.

An `Instruction` the compiler emits is sized exactly from its Opcode. Text is
not: `asm!()` blocks, the inline tables `Emitter.emit_directive` writes as
`RawAsm`, the entry stub, and data `Directive`s. Branch fixup, address
calculation and bank sizing all lay out code that contains it, and size it
here so they agree.
"""

from typing import Sequence

from r65.compiler.codegen.opcodes import Opcode, mnemonic

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


def raw_asm_size(text: str, m16: bool = False, x16: bool = True) -> int:
    """Bytes the lines of *text* assemble to, in the given register widths."""
    return sum(asm_line_size(line, m16, x16) for line in text.splitlines())


def asm_line_size(line: str, m16: bool = False, x16: bool = True) -> int:
    """Bytes one line of assembly text assembles to.

    Comments, labels, non-data directives and WLA-DX's header keywords
    (`LOROM`, `NAME`, ...) are 0. An operand's size follows how WLA-DX reads
    it: by the digits written (`$10` direct page, `$0010` absolute), or
    absolute for a label.
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
    return 1 + _address_bytes(operand)


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
