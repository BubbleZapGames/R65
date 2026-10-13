# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
65816 Opcode definitions.

Each enum variant encodes both the instruction mnemonic and addressing mode.
The enum value is the actual opcode byte, enabling direct machine code generation.

Naming convention: MNEMONIC_MODE
- No suffix for implied addressing (NOP, TAX, RTS)
- IMMEDIATE: #immediate
- DP: direct page (zero page)
- DP_X, DP_Y: direct page indexed
- ABSOLUTE: 16-bit absolute
- ABSOLUTE_X, ABSOLUTE_Y: absolute indexed
- LONG: 24-bit absolute (65816)
- LONG_X: 24-bit absolute indexed
- INDIRECT: (addr)
- INDIRECT_LONG: [addr]
- DP_INDIRECT: (dp)
- DP_INDIRECT_X: (dp,X)
- DP_INDIRECT_Y: (dp),Y
- DP_INDIRECT_LONG: [dp]
- DP_INDIRECT_LONG_Y: [dp],Y
- STACK: stack relative d,S
- STACK_INDIRECT_Y: (d,S),Y
- RELATIVE: for branches (8-bit offset)
- RELATIVE_LONG: for BRL (16-bit offset)
"""

from enum import IntEnum


class Opcode(IntEnum):
    """
    65816 opcodes with addressing mode encoded in the name.

    Value is the actual opcode byte for direct machine code generation.
    """

    # ========================================================================
    # ADC - Add with Carry
    # ========================================================================
    ADC_IMMEDIATE = 0x69
    ADC_DP = 0x65
    ADC_DP_X = 0x75
    ADC_ABSOLUTE = 0x6D
    ADC_ABSOLUTE_X = 0x7D
    ADC_ABSOLUTE_Y = 0x79
    ADC_DP_INDIRECT = 0x72
    ADC_DP_INDIRECT_X = 0x61
    ADC_DP_INDIRECT_Y = 0x71
    ADC_DP_INDIRECT_LONG = 0x67
    ADC_DP_INDIRECT_LONG_Y = 0x77
    ADC_LONG = 0x6F
    ADC_LONG_X = 0x7F
    ADC_STACK = 0x63
    ADC_STACK_INDIRECT_Y = 0x73

    # ========================================================================
    # AND - Logical AND
    # ========================================================================
    AND_IMMEDIATE = 0x29
    AND_DP = 0x25
    AND_DP_X = 0x35
    AND_ABSOLUTE = 0x2D
    AND_ABSOLUTE_X = 0x3D
    AND_ABSOLUTE_Y = 0x39
    AND_DP_INDIRECT = 0x32
    AND_DP_INDIRECT_X = 0x21
    AND_DP_INDIRECT_Y = 0x31
    AND_DP_INDIRECT_LONG = 0x27
    AND_DP_INDIRECT_LONG_Y = 0x37
    AND_LONG = 0x2F
    AND_LONG_X = 0x3F
    AND_STACK = 0x23
    AND_STACK_INDIRECT_Y = 0x33

    # ========================================================================
    # ASL - Arithmetic Shift Left
    # ========================================================================
    ASL = 0x0A              # Accumulator
    ASL_DP = 0x06
    ASL_DP_X = 0x16
    ASL_ABSOLUTE = 0x0E
    ASL_ABSOLUTE_X = 0x1E

    # ========================================================================
    # Branch Instructions
    # ========================================================================
    BCC = 0x90              # Branch if Carry Clear
    BCS = 0xB0              # Branch if Carry Set
    BEQ = 0xF0              # Branch if Equal (Zero set)
    BMI = 0x30              # Branch if Minus (Negative set)
    BNE = 0xD0              # Branch if Not Equal (Zero clear)
    BPL = 0x10              # Branch if Plus (Negative clear)
    BRA = 0x80              # Branch Always (65C02/65816)
    BRL = 0x82              # Branch Long (65816, 16-bit offset)
    BVC = 0x50              # Branch if Overflow Clear
    BVS = 0x70              # Branch if Overflow Set

    # ========================================================================
    # BIT - Test Bits
    # ========================================================================
    BIT_IMMEDIATE = 0x89
    BIT_DP = 0x24
    BIT_DP_X = 0x34
    BIT_ABSOLUTE = 0x2C
    BIT_ABSOLUTE_X = 0x3C

    # ========================================================================
    # BRK - Break
    # ========================================================================
    BRK = 0x00

    # ========================================================================
    # CLC, CLD, CLI, CLV - Clear Flags
    # ========================================================================
    CLC = 0x18              # Clear Carry
    CLD = 0xD8              # Clear Decimal
    CLI = 0x58              # Clear Interrupt Disable
    CLV = 0xB8              # Clear Overflow

    # ========================================================================
    # CMP - Compare Accumulator
    # ========================================================================
    CMP_IMMEDIATE = 0xC9
    CMP_DP = 0xC5
    CMP_DP_X = 0xD5
    CMP_ABSOLUTE = 0xCD
    CMP_ABSOLUTE_X = 0xDD
    CMP_ABSOLUTE_Y = 0xD9
    CMP_DP_INDIRECT = 0xD2
    CMP_DP_INDIRECT_X = 0xC1
    CMP_DP_INDIRECT_Y = 0xD1
    CMP_DP_INDIRECT_LONG = 0xC7
    CMP_DP_INDIRECT_LONG_Y = 0xD7
    CMP_LONG = 0xCF
    CMP_LONG_X = 0xDF
    CMP_STACK = 0xC3
    CMP_STACK_INDIRECT_Y = 0xD3

    # ========================================================================
    # COP - Coprocessor
    # ========================================================================
    COP = 0x02

    # ========================================================================
    # CPX - Compare X Register
    # ========================================================================
    CPX_IMMEDIATE = 0xE0
    CPX_DP = 0xE4
    CPX_ABSOLUTE = 0xEC

    # ========================================================================
    # CPY - Compare Y Register
    # ========================================================================
    CPY_IMMEDIATE = 0xC0
    CPY_DP = 0xC4
    CPY_ABSOLUTE = 0xCC

    # ========================================================================
    # DEC - Decrement
    # ========================================================================
    DEC = 0x3A              # Accumulator (65C02/65816)
    DEC_DP = 0xC6
    DEC_DP_X = 0xD6
    DEC_ABSOLUTE = 0xCE
    DEC_ABSOLUTE_X = 0xDE

    # ========================================================================
    # DEX, DEY - Decrement Index Registers
    # ========================================================================
    DEX = 0xCA
    DEY = 0x88

    # ========================================================================
    # EOR - Exclusive OR
    # ========================================================================
    EOR_IMMEDIATE = 0x49
    EOR_DP = 0x45
    EOR_DP_X = 0x55
    EOR_ABSOLUTE = 0x4D
    EOR_ABSOLUTE_X = 0x5D
    EOR_ABSOLUTE_Y = 0x59
    EOR_DP_INDIRECT = 0x52
    EOR_DP_INDIRECT_X = 0x41
    EOR_DP_INDIRECT_Y = 0x51
    EOR_DP_INDIRECT_LONG = 0x47
    EOR_DP_INDIRECT_LONG_Y = 0x57
    EOR_LONG = 0x4F
    EOR_LONG_X = 0x5F
    EOR_STACK = 0x43
    EOR_STACK_INDIRECT_Y = 0x53

    # ========================================================================
    # INC - Increment
    # ========================================================================
    INC = 0x1A              # Accumulator (65C02/65816)
    INC_DP = 0xE6
    INC_DP_X = 0xF6
    INC_ABSOLUTE = 0xEE
    INC_ABSOLUTE_X = 0xFE

    # ========================================================================
    # INX, INY - Increment Index Registers
    # ========================================================================
    INX = 0xE8
    INY = 0xC8

    # ========================================================================
    # JMP - Jump
    # ========================================================================
    JMP_ABSOLUTE = 0x4C
    JMP_INDIRECT = 0x6C
    JMP_INDIRECT_X = 0x7C     # (addr,X) - 65C02/65816
    JMP_INDIRECT_LONG = 0xDC  # [addr] - 65816
    JMP_LONG = 0x5C           # 24-bit absolute - 65816

    # ========================================================================
    # JSR/JSL - Jump to Subroutine
    # ========================================================================
    JSR = 0x20              # JSR absolute
    JSR_INDIRECT_X = 0xFC   # JSR (addr,X) - 65816
    JSL = 0x22              # JSR long (24-bit) - 65816

    # ========================================================================
    # LDA - Load Accumulator
    # ========================================================================
    LDA_IMMEDIATE = 0xA9
    LDA_DP = 0xA5
    LDA_DP_X = 0xB5
    LDA_ABSOLUTE = 0xAD
    LDA_ABSOLUTE_X = 0xBD
    LDA_ABSOLUTE_Y = 0xB9
    LDA_DP_INDIRECT = 0xB2
    LDA_DP_INDIRECT_X = 0xA1
    LDA_DP_INDIRECT_Y = 0xB1
    LDA_DP_INDIRECT_LONG = 0xA7
    LDA_DP_INDIRECT_LONG_Y = 0xB7
    LDA_LONG = 0xAF
    LDA_LONG_X = 0xBF
    LDA_STACK = 0xA3
    LDA_STACK_INDIRECT_Y = 0xB3

    # ========================================================================
    # LDX - Load X Register
    # ========================================================================
    LDX_IMMEDIATE = 0xA2
    LDX_DP = 0xA6
    LDX_DP_Y = 0xB6
    LDX_ABSOLUTE = 0xAE
    LDX_ABSOLUTE_Y = 0xBE

    # ========================================================================
    # LDY - Load Y Register
    # ========================================================================
    LDY_IMMEDIATE = 0xA0
    LDY_DP = 0xA4
    LDY_DP_X = 0xB4
    LDY_ABSOLUTE = 0xAC
    LDY_ABSOLUTE_X = 0xBC

    # ========================================================================
    # LSR - Logical Shift Right
    # ========================================================================
    LSR = 0x4A              # Accumulator
    LSR_DP = 0x46
    LSR_DP_X = 0x56
    LSR_ABSOLUTE = 0x4E
    LSR_ABSOLUTE_X = 0x5E

    # ========================================================================
    # MVN, MVP - Block Move (65816)
    # ========================================================================
    MVN = 0x54              # Move Negative (increment)
    MVP = 0x44              # Move Positive (decrement)

    # ========================================================================
    # NOP - No Operation
    # ========================================================================
    NOP = 0xEA

    # ========================================================================
    # ORA - Logical OR
    # ========================================================================
    ORA_IMMEDIATE = 0x09
    ORA_DP = 0x05
    ORA_DP_X = 0x15
    ORA_ABSOLUTE = 0x0D
    ORA_ABSOLUTE_X = 0x1D
    ORA_ABSOLUTE_Y = 0x19
    ORA_DP_INDIRECT = 0x12
    ORA_DP_INDIRECT_X = 0x01
    ORA_DP_INDIRECT_Y = 0x11
    ORA_DP_INDIRECT_LONG = 0x07
    ORA_DP_INDIRECT_LONG_Y = 0x17
    ORA_LONG = 0x0F
    ORA_LONG_X = 0x1F
    ORA_STACK = 0x03
    ORA_STACK_INDIRECT_Y = 0x13

    # ========================================================================
    # PEA, PEI, PER - Push Effective Address (65816)
    # ========================================================================
    PEA = 0xF4              # Push Effective Absolute Address
    PEI = 0xD4              # Push Effective Indirect Address
    PER = 0x62              # Push Effective PC Relative Address

    # ========================================================================
    # PHA, PHP, PHX, PHY - Push to Stack
    # ========================================================================
    PHA = 0x48              # Push Accumulator
    PHB = 0x8B              # Push Data Bank Register (65816)
    PHD = 0x0B              # Push Direct Page Register (65816)
    PHK = 0x4B              # Push Program Bank Register (65816)
    PHP = 0x08              # Push Processor Status
    PHX = 0xDA              # Push X (65C02/65816)
    PHY = 0x5A              # Push Y (65C02/65816)

    # ========================================================================
    # PLA, PLP, PLX, PLY - Pull from Stack
    # ========================================================================
    PLA = 0x68              # Pull Accumulator
    PLB = 0xAB              # Pull Data Bank Register (65816)
    PLD = 0x2B              # Pull Direct Page Register (65816)
    PLP = 0x28              # Pull Processor Status
    PLX = 0xFA              # Pull X (65C02/65816)
    PLY = 0x7A              # Pull Y (65C02/65816)

    # ========================================================================
    # REP - Reset Processor Status Bits (65816)
    # ========================================================================
    REP_IMMEDIATE = 0xC2

    # ========================================================================
    # ROL - Rotate Left
    # ========================================================================
    ROL = 0x2A              # Accumulator
    ROL_DP = 0x26
    ROL_DP_X = 0x36
    ROL_ABSOLUTE = 0x2E
    ROL_ABSOLUTE_X = 0x3E

    # ========================================================================
    # ROR - Rotate Right
    # ========================================================================
    ROR = 0x6A              # Accumulator
    ROR_DP = 0x66
    ROR_DP_X = 0x76
    ROR_ABSOLUTE = 0x6E
    ROR_ABSOLUTE_X = 0x7E

    # ========================================================================
    # RTI, RTL, RTS - Return Instructions
    # ========================================================================
    RTI = 0x40              # Return from Interrupt
    RTL = 0x6B              # Return Long (65816)
    RTS = 0x60              # Return from Subroutine

    # ========================================================================
    # SBC - Subtract with Borrow
    # ========================================================================
    SBC_IMMEDIATE = 0xE9
    SBC_DP = 0xE5
    SBC_DP_X = 0xF5
    SBC_ABSOLUTE = 0xED
    SBC_ABSOLUTE_X = 0xFD
    SBC_ABSOLUTE_Y = 0xF9
    SBC_DP_INDIRECT = 0xF2
    SBC_DP_INDIRECT_X = 0xE1
    SBC_DP_INDIRECT_Y = 0xF1
    SBC_DP_INDIRECT_LONG = 0xE7
    SBC_DP_INDIRECT_LONG_Y = 0xF7
    SBC_LONG = 0xEF
    SBC_LONG_X = 0xFF
    SBC_STACK = 0xE3
    SBC_STACK_INDIRECT_Y = 0xF3

    # ========================================================================
    # SEC, SED, SEI - Set Flags
    # ========================================================================
    SEC = 0x38              # Set Carry
    SED = 0xF8              # Set Decimal
    SEI = 0x78              # Set Interrupt Disable

    # ========================================================================
    # SEP - Set Processor Status Bits (65816)
    # ========================================================================
    SEP_IMMEDIATE = 0xE2

    # ========================================================================
    # STA - Store Accumulator
    # ========================================================================
    STA_DP = 0x85
    STA_DP_X = 0x95
    STA_ABSOLUTE = 0x8D
    STA_ABSOLUTE_X = 0x9D
    STA_ABSOLUTE_Y = 0x99
    STA_DP_INDIRECT = 0x92
    STA_DP_INDIRECT_X = 0x81
    STA_DP_INDIRECT_Y = 0x91
    STA_DP_INDIRECT_LONG = 0x87
    STA_DP_INDIRECT_LONG_Y = 0x97
    STA_LONG = 0x8F
    STA_LONG_X = 0x9F
    STA_STACK = 0x83
    STA_STACK_INDIRECT_Y = 0x93

    # ========================================================================
    # STP - Stop Processor (65816)
    # ========================================================================
    STP = 0xDB

    # ========================================================================
    # STX - Store X Register
    # ========================================================================
    STX_DP = 0x86
    STX_DP_Y = 0x96
    STX_ABSOLUTE = 0x8E

    # ========================================================================
    # STY - Store Y Register
    # ========================================================================
    STY_DP = 0x84
    STY_DP_X = 0x94
    STY_ABSOLUTE = 0x8C

    # ========================================================================
    # STZ - Store Zero (65C02/65816)
    # ========================================================================
    STZ_DP = 0x64
    STZ_DP_X = 0x74
    STZ_ABSOLUTE = 0x9C
    STZ_ABSOLUTE_X = 0x9E

    # ========================================================================
    # Transfer Instructions
    # ========================================================================
    TAX = 0xAA              # Transfer A to X
    TAY = 0xA8              # Transfer A to Y
    TCD = 0x5B              # Transfer C (16-bit A) to Direct Page (65816)
    TCS = 0x1B              # Transfer C (16-bit A) to Stack Pointer (65816)
    TDC = 0x7B              # Transfer Direct Page to C (16-bit A) (65816)
    TSC = 0x3B              # Transfer Stack Pointer to C (16-bit A) (65816)
    TSX = 0xBA              # Transfer Stack Pointer to X
    TXA = 0x8A              # Transfer X to A
    TXS = 0x9A              # Transfer X to Stack Pointer
    TXY = 0x9B              # Transfer X to Y (65816)
    TYA = 0x98              # Transfer Y to A
    TYX = 0xBB              # Transfer Y to X (65816)

    # ========================================================================
    # TRB, TSB - Test and Reset/Set Bits (65C02/65816)
    # ========================================================================
    TRB_DP = 0x14
    TRB_ABSOLUTE = 0x1C
    TSB_DP = 0x04
    TSB_ABSOLUTE = 0x0C

    # ========================================================================
    # WAI - Wait for Interrupt (65816)
    # ========================================================================
    WAI = 0xCB

    # ========================================================================
    # WDM - Reserved (65816)
    # ========================================================================
    WDM = 0x42

    # ========================================================================
    # XBA - Exchange B and A (65816)
    # ========================================================================
    XBA = 0xEB

    # ========================================================================
    # XCE - Exchange Carry and Emulation (65816)
    # ========================================================================
    XCE = 0xFB


# ============================================================================
# Helper Functions
# ============================================================================

def mnemonic(op: Opcode) -> str:
    """
    Extract the mnemonic from an opcode.

    Example: Opcode.LDA_IMMEDIATE -> "LDA"
    """
    name = op.name
    # Handle opcodes without addressing mode suffix
    if '_' not in name:
        return name
    return name.split('_')[0]


def addressing_mode(op: Opcode) -> str | None:
    """
    Extract the addressing mode from an opcode.

    Example: Opcode.LDA_IMMEDIATE -> "IMMEDIATE"
    Example: Opcode.NOP -> None
    """
    name = op.name
    if '_' not in name:
        return None
    return '_'.join(name.split('_')[1:])


def is_branch(op: Opcode) -> bool:
    """Check if opcode is a conditional or unconditional branch."""
    return op in BRANCH_OPCODES


def is_jump(op: Opcode) -> bool:
    """Check if opcode is a jump instruction."""
    return op in JUMP_OPCODES


def is_call(op: Opcode) -> bool:
    """Check if opcode is a subroutine call."""
    return op in CALL_OPCODES


def is_return(op: Opcode) -> bool:
    """Check if opcode is a return instruction."""
    return op in RETURN_OPCODES


def is_load(op: Opcode) -> bool:
    """Check if opcode loads a register."""
    return op in LOAD_OPCODES


def is_store(op: Opcode) -> bool:
    """Check if opcode stores a register."""
    return op in STORE_OPCODES


# ============================================================================
# Opcode Categories
# ============================================================================

BRANCH_OPCODES = frozenset({
    Opcode.BCC, Opcode.BCS, Opcode.BEQ, Opcode.BMI,
    Opcode.BNE, Opcode.BPL, Opcode.BRA, Opcode.BRL,
    Opcode.BVC, Opcode.BVS,
})

JUMP_OPCODES = frozenset({
    Opcode.JMP_ABSOLUTE, Opcode.JMP_INDIRECT, Opcode.JMP_INDIRECT_X,
    Opcode.JMP_INDIRECT_LONG, Opcode.JMP_LONG,
})

CALL_OPCODES = frozenset({
    Opcode.JSR, Opcode.JSR_INDIRECT_X, Opcode.JSL,
})

RETURN_OPCODES = frozenset({
    Opcode.RTS, Opcode.RTL, Opcode.RTI,
})

# Load instructions by register
LOAD_A_OPCODES = frozenset({
    Opcode.LDA_IMMEDIATE, Opcode.LDA_DP, Opcode.LDA_DP_X,
    Opcode.LDA_ABSOLUTE, Opcode.LDA_ABSOLUTE_X, Opcode.LDA_ABSOLUTE_Y,
    Opcode.LDA_DP_INDIRECT, Opcode.LDA_DP_INDIRECT_X, Opcode.LDA_DP_INDIRECT_Y,
    Opcode.LDA_DP_INDIRECT_LONG, Opcode.LDA_DP_INDIRECT_LONG_Y,
    Opcode.LDA_LONG, Opcode.LDA_LONG_X,
    Opcode.LDA_STACK, Opcode.LDA_STACK_INDIRECT_Y,
})

LOAD_X_OPCODES = frozenset({
    Opcode.LDX_IMMEDIATE, Opcode.LDX_DP, Opcode.LDX_DP_Y,
    Opcode.LDX_ABSOLUTE, Opcode.LDX_ABSOLUTE_Y,
})

LOAD_Y_OPCODES = frozenset({
    Opcode.LDY_IMMEDIATE, Opcode.LDY_DP, Opcode.LDY_DP_X,
    Opcode.LDY_ABSOLUTE, Opcode.LDY_ABSOLUTE_X,
})

LOAD_OPCODES = LOAD_A_OPCODES | LOAD_X_OPCODES | LOAD_Y_OPCODES

# Store instructions by register
STORE_A_OPCODES = frozenset({
    Opcode.STA_DP, Opcode.STA_DP_X,
    Opcode.STA_ABSOLUTE, Opcode.STA_ABSOLUTE_X, Opcode.STA_ABSOLUTE_Y,
    Opcode.STA_DP_INDIRECT, Opcode.STA_DP_INDIRECT_X, Opcode.STA_DP_INDIRECT_Y,
    Opcode.STA_DP_INDIRECT_LONG, Opcode.STA_DP_INDIRECT_LONG_Y,
    Opcode.STA_LONG, Opcode.STA_LONG_X,
    Opcode.STA_STACK, Opcode.STA_STACK_INDIRECT_Y,
})

STORE_X_OPCODES = frozenset({
    Opcode.STX_DP, Opcode.STX_DP_Y, Opcode.STX_ABSOLUTE,
})

STORE_Y_OPCODES = frozenset({
    Opcode.STY_DP, Opcode.STY_DP_X, Opcode.STY_ABSOLUTE,
})

STORE_Z_OPCODES = frozenset({
    Opcode.STZ_DP, Opcode.STZ_DP_X, Opcode.STZ_ABSOLUTE, Opcode.STZ_ABSOLUTE_X,
})

STORE_OPCODES = STORE_A_OPCODES | STORE_X_OPCODES | STORE_Y_OPCODES | STORE_Z_OPCODES

# Mnemonics that write memory. INC/DEC and the shifts write A instead in
# their accumulator form, which MEMORY_WRITE_OPCODES leaves out.
MEMORY_WRITE_MNEMONICS = frozenset({'STA', 'STX', 'STY', 'STZ', 'INC', 'DEC',
                                    'ASL', 'LSR', 'ROL', 'ROR', 'TSB', 'TRB'})
MEMORY_WRITE_OPCODES = frozenset(
    op for op in Opcode
    if mnemonic(op) in MEMORY_WRITE_MNEMONICS
    and addressing_mode(op) not in (None, 'ACCUMULATOR'))

# ============================================================================
# Instruction effects
# ============================================================================
# What an instruction writes and where its flags come from, by mnemonic.
# Every pass that follows register contents or flags through emitted code
# (emitter tracking, peephole, interrupt save scan) reads these.

# INC/DEC and the shifts in accumulator form (`ASL A`): they write A, where
# every other form writes memory.
ACCUMULATOR_OPCODES = frozenset({Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR,
                                 Opcode.INC, Opcode.DEC})

# Registers each mnemonic writes, beyond the accumulator forms above.
# MVN/MVP walk A, X and Y and leave DBR at the destination bank.
REGISTER_WRITES: dict[str, frozenset] = {
    **{mn: frozenset('A') for mn in ('LDA', 'ADC', 'SBC', 'AND', 'ORA', 'EOR',
                                     'TXA', 'TYA', 'PLA', 'XBA', 'TSC', 'TDC')},
    **{mn: frozenset('X') for mn in ('LDX', 'TAX', 'TSX', 'TYX', 'INX', 'DEX', 'PLX')},
    **{mn: frozenset('Y') for mn in ('LDY', 'TAY', 'TXY', 'INY', 'DEY', 'PLY')},
    'TCD': frozenset({'D'}), 'PLD': frozenset({'D'}),
    'PLB': frozenset({'DBR'}),
    'MVN': frozenset({'A', 'X', 'Y', 'DBR'}), 'MVP': frozenset({'A', 'X', 'Y', 'DBR'}),
}


def registers_written(op: Opcode) -> frozenset:
    """The registers among A, X, Y, D and DBR that *op* writes."""
    regs = REGISTER_WRITES.get(mnemonic(op), frozenset())
    return regs | {'A'} if op in ACCUMULATOR_OPCODES else regs


MODIFIES_A_OPCODES = frozenset(op for op in Opcode if 'A' in registers_written(op))

# The register whose new value N/Z reflect, at that register's width.
_NZ_REGISTER = {
    **{mn: 'A' for mn in ('LDA', 'AND', 'ORA', 'EOR', 'ADC', 'SBC', 'TXA', 'TYA', 'PLA')},
    **{mn: 'X' for mn in ('LDX', 'TAX', 'TSX', 'TYX', 'INX', 'DEX', 'PLX')},
    **{mn: 'Y' for mn in ('LDY', 'TAY', 'TXY', 'INY', 'DEY', 'PLY')},
}

# Instructions that leave N/Z as they were.
NZ_PRESERVING_MNEMONICS = frozenset({
    'STA', 'STX', 'STY', 'STZ',
    'PHA', 'PHX', 'PHY', 'PHB', 'PHD', 'PHK', 'PHP', 'PEA', 'PEI', 'PER',
    'TXS', 'TCS',
    'NOP', 'WDM',
    'BRA', 'BRL', 'BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS',
    'JMP',
    'CLC', 'SEC', 'CLI', 'SEI', 'CLD', 'SED', 'CLV',
    'REP', 'SEP',  # Only affect M/X bits (#$20/#$10) in compiler usage
})


def nz_register(op: Opcode, m16: "bool | None" = None) -> "str | None":
    """The register (A, X or Y) whose value N/Z reflect after *op*, at that
    register's width; None when they reflect something else - a compare,
    BIT, a memory read-modify-write, PLP, a call.

    XBA sets them from the new low byte and TSC/TDC from all 16 bits,
    whatever the accumulator width, so they count for A only at the width
    *m16* says (and not when it is unknown).
    """
    if op in ACCUMULATOR_OPCODES:
        return 'A'
    mn = mnemonic(op)
    if mn == 'XBA':
        return 'A' if m16 is False else None
    if mn in ('TSC', 'TDC'):
        return 'A' if m16 is True else None
    return _NZ_REGISTER.get(mn)


# Instructions whose N/Z flags are those of `CMP #0` on A, whatever the width.
NZ_OF_A_OPCODES = frozenset(op for op in Opcode if nz_register(op) == 'A')

# Instructions that read the carry, and those that replace it.
CARRY_READ_OPCODES = frozenset(
    op for op in Opcode
    if mnemonic(op) in ('ADC', 'SBC', 'ROL', 'ROR', 'BCC', 'BCS', 'XCE'))
CARRY_WRITE_OPCODES = CARRY_READ_OPCODES - {Opcode.BCC, Opcode.BCS} | frozenset(
    op for op in Opcode
    if mnemonic(op) in ('CLC', 'SEC', 'CMP', 'CPX', 'CPY', 'ASL', 'LSR', 'PLP'))


# ============================================================================
# Instruction Sizes
# ============================================================================

# Base instruction sizes (not counting accumulator/index width adjustments)
# Immediate instructions with * need +1 byte in 16-bit mode
OPCODE_SIZES: dict[Opcode, int] = {
    # ADC
    Opcode.ADC_IMMEDIATE: 2,        # +1 in m16
    Opcode.ADC_DP: 2,
    Opcode.ADC_DP_X: 2,
    Opcode.ADC_ABSOLUTE: 3,
    Opcode.ADC_ABSOLUTE_X: 3,
    Opcode.ADC_ABSOLUTE_Y: 3,
    Opcode.ADC_DP_INDIRECT: 2,
    Opcode.ADC_DP_INDIRECT_X: 2,
    Opcode.ADC_DP_INDIRECT_Y: 2,
    Opcode.ADC_DP_INDIRECT_LONG: 2,
    Opcode.ADC_DP_INDIRECT_LONG_Y: 2,
    Opcode.ADC_LONG: 4,
    Opcode.ADC_LONG_X: 4,
    Opcode.ADC_STACK: 2,
    Opcode.ADC_STACK_INDIRECT_Y: 2,

    # AND
    Opcode.AND_IMMEDIATE: 2,        # +1 in m16
    Opcode.AND_DP: 2,
    Opcode.AND_DP_X: 2,
    Opcode.AND_ABSOLUTE: 3,
    Opcode.AND_ABSOLUTE_X: 3,
    Opcode.AND_ABSOLUTE_Y: 3,
    Opcode.AND_DP_INDIRECT: 2,
    Opcode.AND_DP_INDIRECT_X: 2,
    Opcode.AND_DP_INDIRECT_Y: 2,
    Opcode.AND_DP_INDIRECT_LONG: 2,
    Opcode.AND_DP_INDIRECT_LONG_Y: 2,
    Opcode.AND_LONG: 4,
    Opcode.AND_LONG_X: 4,
    Opcode.AND_STACK: 2,
    Opcode.AND_STACK_INDIRECT_Y: 2,

    # ASL
    Opcode.ASL: 1,
    Opcode.ASL_DP: 2,
    Opcode.ASL_DP_X: 2,
    Opcode.ASL_ABSOLUTE: 3,
    Opcode.ASL_ABSOLUTE_X: 3,

    # Branches
    Opcode.BCC: 2,
    Opcode.BCS: 2,
    Opcode.BEQ: 2,
    Opcode.BMI: 2,
    Opcode.BNE: 2,
    Opcode.BPL: 2,
    Opcode.BRA: 2,
    Opcode.BRL: 3,
    Opcode.BVC: 2,
    Opcode.BVS: 2,

    # BIT
    Opcode.BIT_IMMEDIATE: 2,        # +1 in m16
    Opcode.BIT_DP: 2,
    Opcode.BIT_DP_X: 2,
    Opcode.BIT_ABSOLUTE: 3,
    Opcode.BIT_ABSOLUTE_X: 3,

    # BRK
    Opcode.BRK: 2,

    # Clear flags
    Opcode.CLC: 1,
    Opcode.CLD: 1,
    Opcode.CLI: 1,
    Opcode.CLV: 1,

    # CMP
    Opcode.CMP_IMMEDIATE: 2,        # +1 in m16
    Opcode.CMP_DP: 2,
    Opcode.CMP_DP_X: 2,
    Opcode.CMP_ABSOLUTE: 3,
    Opcode.CMP_ABSOLUTE_X: 3,
    Opcode.CMP_ABSOLUTE_Y: 3,
    Opcode.CMP_DP_INDIRECT: 2,
    Opcode.CMP_DP_INDIRECT_X: 2,
    Opcode.CMP_DP_INDIRECT_Y: 2,
    Opcode.CMP_DP_INDIRECT_LONG: 2,
    Opcode.CMP_DP_INDIRECT_LONG_Y: 2,
    Opcode.CMP_LONG: 4,
    Opcode.CMP_LONG_X: 4,
    Opcode.CMP_STACK: 2,
    Opcode.CMP_STACK_INDIRECT_Y: 2,

    # COP
    Opcode.COP: 2,

    # CPX
    Opcode.CPX_IMMEDIATE: 2,        # +1 in x16
    Opcode.CPX_DP: 2,
    Opcode.CPX_ABSOLUTE: 3,

    # CPY
    Opcode.CPY_IMMEDIATE: 2,        # +1 in x16
    Opcode.CPY_DP: 2,
    Opcode.CPY_ABSOLUTE: 3,

    # DEC
    Opcode.DEC: 1,
    Opcode.DEC_DP: 2,
    Opcode.DEC_DP_X: 2,
    Opcode.DEC_ABSOLUTE: 3,
    Opcode.DEC_ABSOLUTE_X: 3,

    # DEX, DEY
    Opcode.DEX: 1,
    Opcode.DEY: 1,

    # EOR
    Opcode.EOR_IMMEDIATE: 2,        # +1 in m16
    Opcode.EOR_DP: 2,
    Opcode.EOR_DP_X: 2,
    Opcode.EOR_ABSOLUTE: 3,
    Opcode.EOR_ABSOLUTE_X: 3,
    Opcode.EOR_ABSOLUTE_Y: 3,
    Opcode.EOR_DP_INDIRECT: 2,
    Opcode.EOR_DP_INDIRECT_X: 2,
    Opcode.EOR_DP_INDIRECT_Y: 2,
    Opcode.EOR_DP_INDIRECT_LONG: 2,
    Opcode.EOR_DP_INDIRECT_LONG_Y: 2,
    Opcode.EOR_LONG: 4,
    Opcode.EOR_LONG_X: 4,
    Opcode.EOR_STACK: 2,
    Opcode.EOR_STACK_INDIRECT_Y: 2,

    # INC
    Opcode.INC: 1,
    Opcode.INC_DP: 2,
    Opcode.INC_DP_X: 2,
    Opcode.INC_ABSOLUTE: 3,
    Opcode.INC_ABSOLUTE_X: 3,

    # INX, INY
    Opcode.INX: 1,
    Opcode.INY: 1,

    # JMP
    Opcode.JMP_ABSOLUTE: 3,
    Opcode.JMP_INDIRECT: 3,
    Opcode.JMP_INDIRECT_X: 3,
    Opcode.JMP_INDIRECT_LONG: 3,
    Opcode.JMP_LONG: 4,

    # JSR/JSL
    Opcode.JSR: 3,
    Opcode.JSR_INDIRECT_X: 3,
    Opcode.JSL: 4,

    # LDA
    Opcode.LDA_IMMEDIATE: 2,        # +1 in m16
    Opcode.LDA_DP: 2,
    Opcode.LDA_DP_X: 2,
    Opcode.LDA_ABSOLUTE: 3,
    Opcode.LDA_ABSOLUTE_X: 3,
    Opcode.LDA_ABSOLUTE_Y: 3,
    Opcode.LDA_DP_INDIRECT: 2,
    Opcode.LDA_DP_INDIRECT_X: 2,
    Opcode.LDA_DP_INDIRECT_Y: 2,
    Opcode.LDA_DP_INDIRECT_LONG: 2,
    Opcode.LDA_DP_INDIRECT_LONG_Y: 2,
    Opcode.LDA_LONG: 4,
    Opcode.LDA_LONG_X: 4,
    Opcode.LDA_STACK: 2,
    Opcode.LDA_STACK_INDIRECT_Y: 2,

    # LDX
    Opcode.LDX_IMMEDIATE: 2,        # +1 in x16
    Opcode.LDX_DP: 2,
    Opcode.LDX_DP_Y: 2,
    Opcode.LDX_ABSOLUTE: 3,
    Opcode.LDX_ABSOLUTE_Y: 3,

    # LDY
    Opcode.LDY_IMMEDIATE: 2,        # +1 in x16
    Opcode.LDY_DP: 2,
    Opcode.LDY_DP_X: 2,
    Opcode.LDY_ABSOLUTE: 3,
    Opcode.LDY_ABSOLUTE_X: 3,

    # LSR
    Opcode.LSR: 1,
    Opcode.LSR_DP: 2,
    Opcode.LSR_DP_X: 2,
    Opcode.LSR_ABSOLUTE: 3,
    Opcode.LSR_ABSOLUTE_X: 3,

    # MVN, MVP
    Opcode.MVN: 3,
    Opcode.MVP: 3,

    # NOP
    Opcode.NOP: 1,

    # ORA
    Opcode.ORA_IMMEDIATE: 2,        # +1 in m16
    Opcode.ORA_DP: 2,
    Opcode.ORA_DP_X: 2,
    Opcode.ORA_ABSOLUTE: 3,
    Opcode.ORA_ABSOLUTE_X: 3,
    Opcode.ORA_ABSOLUTE_Y: 3,
    Opcode.ORA_DP_INDIRECT: 2,
    Opcode.ORA_DP_INDIRECT_X: 2,
    Opcode.ORA_DP_INDIRECT_Y: 2,
    Opcode.ORA_DP_INDIRECT_LONG: 2,
    Opcode.ORA_DP_INDIRECT_LONG_Y: 2,
    Opcode.ORA_LONG: 4,
    Opcode.ORA_LONG_X: 4,
    Opcode.ORA_STACK: 2,
    Opcode.ORA_STACK_INDIRECT_Y: 2,

    # PEA, PEI, PER
    Opcode.PEA: 3,
    Opcode.PEI: 2,
    Opcode.PER: 3,

    # Push
    Opcode.PHA: 1,
    Opcode.PHB: 1,
    Opcode.PHD: 1,
    Opcode.PHK: 1,
    Opcode.PHP: 1,
    Opcode.PHX: 1,
    Opcode.PHY: 1,

    # Pull
    Opcode.PLA: 1,
    Opcode.PLB: 1,
    Opcode.PLD: 1,
    Opcode.PLP: 1,
    Opcode.PLX: 1,
    Opcode.PLY: 1,

    # REP
    Opcode.REP_IMMEDIATE: 2,

    # ROL
    Opcode.ROL: 1,
    Opcode.ROL_DP: 2,
    Opcode.ROL_DP_X: 2,
    Opcode.ROL_ABSOLUTE: 3,
    Opcode.ROL_ABSOLUTE_X: 3,

    # ROR
    Opcode.ROR: 1,
    Opcode.ROR_DP: 2,
    Opcode.ROR_DP_X: 2,
    Opcode.ROR_ABSOLUTE: 3,
    Opcode.ROR_ABSOLUTE_X: 3,

    # Returns
    Opcode.RTI: 1,
    Opcode.RTL: 1,
    Opcode.RTS: 1,

    # SBC
    Opcode.SBC_IMMEDIATE: 2,        # +1 in m16
    Opcode.SBC_DP: 2,
    Opcode.SBC_DP_X: 2,
    Opcode.SBC_ABSOLUTE: 3,
    Opcode.SBC_ABSOLUTE_X: 3,
    Opcode.SBC_ABSOLUTE_Y: 3,
    Opcode.SBC_DP_INDIRECT: 2,
    Opcode.SBC_DP_INDIRECT_X: 2,
    Opcode.SBC_DP_INDIRECT_Y: 2,
    Opcode.SBC_DP_INDIRECT_LONG: 2,
    Opcode.SBC_DP_INDIRECT_LONG_Y: 2,
    Opcode.SBC_LONG: 4,
    Opcode.SBC_LONG_X: 4,
    Opcode.SBC_STACK: 2,
    Opcode.SBC_STACK_INDIRECT_Y: 2,

    # Set flags
    Opcode.SEC: 1,
    Opcode.SED: 1,
    Opcode.SEI: 1,

    # SEP
    Opcode.SEP_IMMEDIATE: 2,

    # STA
    Opcode.STA_DP: 2,
    Opcode.STA_DP_X: 2,
    Opcode.STA_ABSOLUTE: 3,
    Opcode.STA_ABSOLUTE_X: 3,
    Opcode.STA_ABSOLUTE_Y: 3,
    Opcode.STA_DP_INDIRECT: 2,
    Opcode.STA_DP_INDIRECT_X: 2,
    Opcode.STA_DP_INDIRECT_Y: 2,
    Opcode.STA_DP_INDIRECT_LONG: 2,
    Opcode.STA_DP_INDIRECT_LONG_Y: 2,
    Opcode.STA_LONG: 4,
    Opcode.STA_LONG_X: 4,
    Opcode.STA_STACK: 2,
    Opcode.STA_STACK_INDIRECT_Y: 2,

    # STP
    Opcode.STP: 1,

    # STX
    Opcode.STX_DP: 2,
    Opcode.STX_DP_Y: 2,
    Opcode.STX_ABSOLUTE: 3,

    # STY
    Opcode.STY_DP: 2,
    Opcode.STY_DP_X: 2,
    Opcode.STY_ABSOLUTE: 3,

    # STZ
    Opcode.STZ_DP: 2,
    Opcode.STZ_DP_X: 2,
    Opcode.STZ_ABSOLUTE: 3,
    Opcode.STZ_ABSOLUTE_X: 3,

    # Transfers
    Opcode.TAX: 1,
    Opcode.TAY: 1,
    Opcode.TCD: 1,
    Opcode.TCS: 1,
    Opcode.TDC: 1,
    Opcode.TSC: 1,
    Opcode.TSX: 1,
    Opcode.TXA: 1,
    Opcode.TXS: 1,
    Opcode.TXY: 1,
    Opcode.TYA: 1,
    Opcode.TYX: 1,

    # TRB, TSB
    Opcode.TRB_DP: 2,
    Opcode.TRB_ABSOLUTE: 3,
    Opcode.TSB_DP: 2,
    Opcode.TSB_ABSOLUTE: 3,

    # WAI
    Opcode.WAI: 1,

    # WDM
    Opcode.WDM: 2,

    # XBA
    Opcode.XBA: 1,

    # XCE
    Opcode.XCE: 1,
}


# Opcodes that need +1 byte for immediate operand in 16-bit accumulator mode
M16_IMMEDIATE_OPCODES = frozenset({
    Opcode.ADC_IMMEDIATE, Opcode.AND_IMMEDIATE, Opcode.BIT_IMMEDIATE,
    Opcode.CMP_IMMEDIATE, Opcode.EOR_IMMEDIATE, Opcode.LDA_IMMEDIATE,
    Opcode.ORA_IMMEDIATE, Opcode.SBC_IMMEDIATE,
})

# Opcodes that need +1 byte for immediate operand in 16-bit index mode
X16_IMMEDIATE_OPCODES = frozenset({
    Opcode.CPX_IMMEDIATE, Opcode.CPY_IMMEDIATE,
    Opcode.LDX_IMMEDIATE, Opcode.LDY_IMMEDIATE,
})


# ============================================================================
# Instruction Selection Mappings
# ============================================================================
# Consolidated opcode mappings used by instruction selectors.

# Mapping from base mnemonic to opcode variants by addressing mode
OPCODE_VARIANTS: dict[str, dict[str, Opcode]] = {}
for _op in Opcode:
    if addressing_mode(_op) is not None:
        OPCODE_VARIANTS.setdefault(mnemonic(_op), {})[addressing_mode(_op)] = _op
del _op

# Mappings from register name to push/pull opcodes
PUSH_OPCODES: dict[str, Opcode] = {
    'A': Opcode.PHA, 'X': Opcode.PHX, 'Y': Opcode.PHY,
    'STATUS': Opcode.PHP, 'P': Opcode.PHP,
    'D': Opcode.PHD, 'DBR': Opcode.PHB, 'B': Opcode.PHB,
}

PULL_OPCODES: dict[str, Opcode] = {
    'A': Opcode.PLA, 'X': Opcode.PLX, 'Y': Opcode.PLY,
    'STATUS': Opcode.PLP, 'P': Opcode.PLP,
    'D': Opcode.PLD, 'DBR': Opcode.PLB, 'B': Opcode.PLB,
}

# Mappings for register transfers
TRANSFER_OPCODES: dict[tuple[str, str], Opcode] = {
    ('A', 'X'): Opcode.TAX,
    ('A', 'Y'): Opcode.TAY,
    ('X', 'A'): Opcode.TXA,
    ('Y', 'A'): Opcode.TYA,
    ('X', 'Y'): Opcode.TXY,
    ('Y', 'X'): Opcode.TYX,
}

# Mappings for load immediate by register
LOAD_IMMEDIATE_OPCODES: dict[str, Opcode] = {
    'A': Opcode.LDA_IMMEDIATE,
    'X': Opcode.LDX_IMMEDIATE,
    'Y': Opcode.LDY_IMMEDIATE,
}

# Mappings for store to DP by register
STORE_DP_OPCODES: dict[str, Opcode] = {
    'A': Opcode.STA_DP,
    'X': Opcode.STX_DP,
    'Y': Opcode.STY_DP,
}

# Mappings for load from DP by register
LOAD_DP_OPCODES: dict[str, Opcode] = {
    'A': Opcode.LDA_DP,
    'X': Opcode.LDX_DP,
    'Y': Opcode.LDY_DP,
}

# Load mnemonic by register
LOAD_MNEMONICS: dict[str, str] = {
    'A': 'LDA', 'X': 'LDX', 'Y': 'LDY',
}

# Store mnemonic by register
STORE_MNEMONICS: dict[str, str] = {
    'A': 'STA', 'X': 'STX', 'Y': 'STY',
}

# Mappings for register increment/decrement
INCREMENT_OPCODES: dict[str, Opcode] = {'X': Opcode.INX, 'Y': Opcode.INY, 'A': Opcode.INC}
DECREMENT_OPCODES: dict[str, Opcode] = {'X': Opcode.DEX, 'Y': Opcode.DEY, 'A': Opcode.DEC}

# Power-of-2 to shift count mapping (for multiply/divide optimization)
POWER_OF_2_SHIFTS: dict[int, int] = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8}

def instruction_size(op: Opcode, m16: bool = False, x16: bool = False) -> int:
    """
    Get the size of an instruction in bytes.

    Args:
        op: The opcode
        m16: True if accumulator is 16-bit
        x16: True if index registers are 16-bit

    Returns:
        Size in bytes
    """
    size = OPCODE_SIZES[op]
    if m16 and op in M16_IMMEDIATE_OPCODES:
        size += 1
    if x16 and op in X16_IMMEDIATE_OPCODES:
        size += 1
    return size
