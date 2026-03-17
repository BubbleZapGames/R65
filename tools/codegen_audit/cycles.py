# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Static 65816 cycle and byte counter.

Parses WLA-DX assembly lines and estimates cycle counts and instruction sizes
based on the 65816 datasheet. For loops, counts the body once (not unrolled).
"""

import re

# Addressing mode patterns (WLA-DX syntax) and their (bytes, cycles) for typical instructions
# Format: (regex_pattern, operand_bytes, base_cycles)
# Note: cycles vary by instruction; this table gives the most common base cost.
# Actual costs adjusted per-mnemonic where needed.
_ADDRESSING_MODES = [
    # Stack relative indirect indexed: (d,S),Y
    (re.compile(r'^\(\$[0-9A-Fa-f]+,\s*[Ss]\),\s*[Yy]$'), 1, 7),
    # Stack relative: d,S
    (re.compile(r'^\$[0-9A-Fa-f]+,\s*[Ss]$'), 1, 4),
    # Direct page indirect long indexed: [dp],Y
    (re.compile(r'^\[\$[0-9A-Fa-f]{1,2}\],\s*[Yy]$'), 1, 6),
    # Direct page indirect long: [dp]
    (re.compile(r'^\[\$[0-9A-Fa-f]{1,2}\]$'), 1, 6),
    # Direct page indirect indexed: (dp),Y
    (re.compile(r'^\(\$[0-9A-Fa-f]{1,2}\),\s*[Yy]$'), 1, 5),
    # Direct page indexed indirect: (dp,X)
    (re.compile(r'^\(\$[0-9A-Fa-f]{1,2},\s*[Xx]\)$'), 1, 6),
    # Direct page indirect: (dp)
    (re.compile(r'^\(\$[0-9A-Fa-f]{1,2}\)$'), 1, 5),
    # Absolute indirect long: [addr]
    (re.compile(r'^\[\$[0-9A-Fa-f]{3,4}\]$'), 2, 6),
    # Absolute indirect indexed: (addr,X)
    (re.compile(r'^\(\$[0-9A-Fa-f]{3,4},\s*[Xx]\)$'), 2, 6),
    # Absolute indirect: (addr)
    (re.compile(r'^\(\$[0-9A-Fa-f]{3,4}\)$'), 2, 5),
    # Long indexed: $addr,X (24-bit)
    (re.compile(r'^\$[0-9A-Fa-f]{5,6},\s*[Xx]$'), 3, 5),
    # Long: $addr (24-bit)
    (re.compile(r'^\$[0-9A-Fa-f]{5,6}$'), 3, 5),
    # Absolute indexed Y: $addr,Y (16-bit)
    (re.compile(r'^\$[0-9A-Fa-f]{3,4},\s*[Yy]$'), 2, 4),
    # Absolute indexed X: $addr,X (16-bit)
    (re.compile(r'^\$[0-9A-Fa-f]{3,4},\s*[Xx]$'), 2, 4),
    # Direct page indexed Y: $dp,Y
    (re.compile(r'^\$[0-9A-Fa-f]{1,2},\s*[Yy]$'), 1, 4),
    # Direct page indexed X: $dp,X
    (re.compile(r'^\$[0-9A-Fa-f]{1,2},\s*[Xx]$'), 1, 4),
    # Immediate (8 or 16-bit): #$xx or #$xxxx or #value
    (re.compile(r'^#'), -1, 2),  # -1 = depends on M/X flag
    # Absolute: $addr (16-bit)
    (re.compile(r'^\$[0-9A-Fa-f]{3,4}$'), 2, 4),
    # Direct page: $dp (8-bit address)
    (re.compile(r'^\$[0-9A-Fa-f]{1,2}$'), 1, 3),
    # Label references (symbols) — treat as absolute
    (re.compile(r'^[A-Za-z_]'), 2, 4),
]

# Instructions with no operand (implied/accumulator)
_IMPLIED_CYCLES: dict[str, tuple[int, int]] = {
    # (bytes, cycles)
    'NOP': (1, 2), 'CLC': (1, 2), 'SEC': (1, 2), 'CLI': (1, 2),
    'SEI': (1, 2), 'CLD': (1, 2), 'SED': (1, 2), 'CLV': (1, 2),
    'TAX': (1, 2), 'TAY': (1, 2), 'TXA': (1, 2), 'TYA': (1, 2),
    'TSX': (1, 2), 'TXS': (1, 2), 'TXY': (1, 2), 'TYX': (1, 2),
    'TCD': (1, 2), 'TCS': (1, 2), 'TDC': (1, 2), 'TSC': (1, 2),
    'PHA': (1, 3), 'PHP': (1, 3), 'PHX': (1, 3), 'PHY': (1, 3),
    'PHB': (1, 3), 'PHD': (1, 4), 'PHK': (1, 3),
    'PLA': (1, 4), 'PLP': (1, 4), 'PLX': (1, 4), 'PLY': (1, 4),
    'PLB': (1, 4), 'PLD': (1, 5),
    'RTS': (1, 6), 'RTL': (1, 6), 'RTI': (1, 7),
    'INX': (1, 2), 'INY': (1, 2), 'DEX': (1, 2), 'DEY': (1, 2),
    'INC': (1, 2), 'DEC': (1, 2),  # Accumulator
    'ASL': (1, 2), 'LSR': (1, 2), 'ROL': (1, 2), 'ROR': (1, 2),
    'XBA': (1, 3), 'XCE': (1, 2),
    'WAI': (1, 3), 'STP': (1, 3), 'WDM': (2, 2), 'BRK': (2, 7),
    'COP': (2, 7),
}

# Branch instructions: always 2 bytes (opcode + relative offset)
_BRANCH_MNEMONICS = frozenset({
    'BCC', 'BCS', 'BEQ', 'BNE', 'BMI', 'BPL', 'BRA', 'BVC', 'BVS',
})
# BRL is 3 bytes (16-bit relative offset)

# Instructions where immediate size depends on M flag (8/16-bit A)
_M_FLAG_IMMEDIATE = frozenset({
    'LDA', 'STA', 'ADC', 'SBC', 'AND', 'ORA', 'EOR', 'CMP', 'BIT',
})

# Instructions where immediate size depends on X flag (8/16-bit X/Y)
_X_FLAG_IMMEDIATE = frozenset({
    'LDX', 'LDY', 'CPX', 'CPY',
})

# Special cycle costs per mnemonic (overrides default for the addressing mode)
_SPECIAL_CYCLES: dict[str, dict[str, int]] = {
    'JSR': {'absolute': 6, 'indirect_x': 8},
    'JSL': {'long': 8},
    'JMP': {'absolute': 3, 'indirect': 5, 'indirect_x': 6, 'long': 4, 'indirect_long': 6},
    'STA': {'dp': 3, 'absolute': 4, 'long': 5},
    'STX': {'dp': 3, 'absolute': 4},
    'STY': {'dp': 3, 'absolute': 4},
    'STZ': {'dp': 3, 'absolute': 4},
}


def _parse_instruction(line: str) -> tuple[str, str] | None:
    """Parse a WLA-DX instruction line into (mnemonic, operand).

    Returns None for non-instruction lines.
    """
    # Strip comments
    if ';' in line:
        line = line[:line.index(';')]
    stripped = line.strip()

    if not stripped or stripped.startswith('.') or stripped.endswith(':'):
        return None

    parts = stripped.split(None, 1)
    mnemonic = parts[0].upper()
    operand = parts[1].strip() if len(parts) > 1 else ''
    return mnemonic, operand


def _estimate_instruction(mnemonic: str, operand: str) -> tuple[int, int]:
    """Estimate (bytes, cycles) for a single instruction.

    Returns (bytes, cycles). Uses conservative estimates (no page-crossing
    penalties, assumes 8-bit M/X for immediate sizing).
    """
    # REP/SEP: always 2 bytes, 3 cycles
    if mnemonic in ('REP', 'SEP'):
        return 2, 3

    # PEA: 3 bytes, 5 cycles
    if mnemonic == 'PEA':
        return 3, 5

    # PEI: 2 bytes, 6 cycles
    if mnemonic == 'PEI':
        return 2, 6

    # PER: 3 bytes, 6 cycles
    if mnemonic == 'PER':
        return 3, 6

    # MVN/MVP: 3 bytes, 7 cycles per byte moved
    if mnemonic in ('MVN', 'MVP'):
        return 3, 7

    # Implied/accumulator
    if not operand:
        if mnemonic in _IMPLIED_CYCLES:
            return _IMPLIED_CYCLES[mnemonic]
        return 1, 2  # Unknown implied

    # Branches
    if mnemonic in _BRANCH_MNEMONICS:
        return 2, 3  # 2 bytes, ~3 cycles (taken)
    if mnemonic == 'BRL':
        return 3, 4

    # Match addressing mode
    for pattern, op_bytes, base_cycles in _ADDRESSING_MODES:
        if pattern.search(operand):
            if op_bytes == -1:
                # Immediate — size depends on M/X flag
                # Default to 8-bit (2 bytes total), but 16-bit ops add 1
                if mnemonic in _M_FLAG_IMMEDIATE:
                    op_bytes = 1  # assume m8
                elif mnemonic in _X_FLAG_IMMEDIATE:
                    op_bytes = 2  # X/Y always 16-bit in R65
                else:
                    op_bytes = 1
            total_bytes = 1 + op_bytes  # opcode + operand
            return total_bytes, base_cycles

    # Fallback: assume absolute addressing
    return 3, 4


def count_cycles(asm_lines: list[str]) -> int:
    """Count total estimated cycles for a list of assembly lines."""
    total = 0
    for line in asm_lines:
        parsed = _parse_instruction(line)
        if parsed:
            _, cycles = _estimate_instruction(*parsed)
            total += cycles
    return total


def count_bytes(asm_lines: list[str]) -> int:
    """Count total instruction bytes for a list of assembly lines."""
    total = 0
    for line in asm_lines:
        parsed = _parse_instruction(line)
        if parsed:
            nbytes, _ = _estimate_instruction(*parsed)
            total += nbytes
    return total


def count_instructions(asm_lines: list[str]) -> int:
    """Count number of instructions in a list of assembly lines."""
    count = 0
    for line in asm_lines:
        if _parse_instruction(line) is not None:
            count += 1
    return count


def get_metrics(asm_lines: list[str]) -> dict[str, int]:
    """Get all metrics at once: instructions, bytes, cycles."""
    instructions = 0
    total_bytes = 0
    total_cycles = 0

    for line in asm_lines:
        parsed = _parse_instruction(line)
        if parsed:
            instructions += 1
            nbytes, cycles = _estimate_instruction(*parsed)
            total_bytes += nbytes
            total_cycles += cycles

    return {
        'instructions': instructions,
        'bytes': total_bytes,
        'cycles': total_cycles,
    }
