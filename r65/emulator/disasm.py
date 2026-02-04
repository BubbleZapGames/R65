"""
65816 Disassembler for trace output.
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .cpu import CPU65816
    from .memory import Memory


# Opcode info: (mnemonic, addressing_mode, base_size)
# Size may vary based on M/X flags for immediate modes
OPCODE_INFO: dict[int, Tuple[str, str, int]] = {
    0x00: ("BRK", "sig", 2),
    0x01: ("ORA", "dp_x_ind", 2),
    0x02: ("COP", "sig", 2),
    0x03: ("ORA", "sr", 2),
    0x04: ("TSB", "dp", 2),
    0x05: ("ORA", "dp", 2),
    0x06: ("ASL", "dp", 2),
    0x07: ("ORA", "dp_ind_long", 2),
    0x08: ("PHP", "imp", 1),
    0x09: ("ORA", "imm_m", 2),  # +1 if M=0
    0x0A: ("ASL", "acc", 1),
    0x0B: ("PHD", "imp", 1),
    0x0C: ("TSB", "abs", 3),
    0x0D: ("ORA", "abs", 3),
    0x0E: ("ASL", "abs", 3),
    0x0F: ("ORA", "long", 4),
    0x10: ("BPL", "rel8", 2),
    0x11: ("ORA", "dp_ind_y", 2),
    0x12: ("ORA", "dp_ind", 2),
    0x13: ("ORA", "sr_ind_y", 2),
    0x14: ("TRB", "dp", 2),
    0x15: ("ORA", "dp_x", 2),
    0x16: ("ASL", "dp_x", 2),
    0x17: ("ORA", "dp_ind_long_y", 2),
    0x18: ("CLC", "imp", 1),
    0x19: ("ORA", "abs_y", 3),
    0x1A: ("INC", "acc", 1),
    0x1B: ("TCS", "imp", 1),
    0x1C: ("TRB", "abs", 3),
    0x1D: ("ORA", "abs_x", 3),
    0x1E: ("ASL", "abs_x", 3),
    0x1F: ("ORA", "long_x", 4),
    0x20: ("JSR", "abs", 3),
    0x21: ("AND", "dp_x_ind", 2),
    0x22: ("JSL", "long", 4),
    0x23: ("AND", "sr", 2),
    0x24: ("BIT", "dp", 2),
    0x25: ("AND", "dp", 2),
    0x26: ("ROL", "dp", 2),
    0x27: ("AND", "dp_ind_long", 2),
    0x28: ("PLP", "imp", 1),
    0x29: ("AND", "imm_m", 2),
    0x2A: ("ROL", "acc", 1),
    0x2B: ("PLD", "imp", 1),
    0x2C: ("BIT", "abs", 3),
    0x2D: ("AND", "abs", 3),
    0x2E: ("ROL", "abs", 3),
    0x2F: ("AND", "long", 4),
    0x30: ("BMI", "rel8", 2),
    0x31: ("AND", "dp_ind_y", 2),
    0x32: ("AND", "dp_ind", 2),
    0x33: ("AND", "sr_ind_y", 2),
    0x34: ("BIT", "dp_x", 2),
    0x35: ("AND", "dp_x", 2),
    0x36: ("ROL", "dp_x", 2),
    0x37: ("AND", "dp_ind_long_y", 2),
    0x38: ("SEC", "imp", 1),
    0x39: ("AND", "abs_y", 3),
    0x3A: ("DEC", "acc", 1),
    0x3B: ("TSC", "imp", 1),
    0x3C: ("BIT", "abs_x", 3),
    0x3D: ("AND", "abs_x", 3),
    0x3E: ("ROL", "abs_x", 3),
    0x3F: ("AND", "long_x", 4),
    0x40: ("RTI", "imp", 1),
    0x41: ("EOR", "dp_x_ind", 2),
    0x42: ("WDM", "sig", 2),
    0x43: ("EOR", "sr", 2),
    0x44: ("MVP", "blk", 3),
    0x45: ("EOR", "dp", 2),
    0x46: ("LSR", "dp", 2),
    0x47: ("EOR", "dp_ind_long", 2),
    0x48: ("PHA", "imp", 1),
    0x49: ("EOR", "imm_m", 2),
    0x4A: ("LSR", "acc", 1),
    0x4B: ("PHK", "imp", 1),
    0x4C: ("JMP", "abs", 3),
    0x4D: ("EOR", "abs", 3),
    0x4E: ("LSR", "abs", 3),
    0x4F: ("EOR", "long", 4),
    0x50: ("BVC", "rel8", 2),
    0x51: ("EOR", "dp_ind_y", 2),
    0x52: ("EOR", "dp_ind", 2),
    0x53: ("EOR", "sr_ind_y", 2),
    0x54: ("MVN", "blk", 3),
    0x55: ("EOR", "dp_x", 2),
    0x56: ("LSR", "dp_x", 2),
    0x57: ("EOR", "dp_ind_long_y", 2),
    0x58: ("CLI", "imp", 1),
    0x59: ("EOR", "abs_y", 3),
    0x5A: ("PHY", "imp", 1),
    0x5B: ("TCD", "imp", 1),
    0x5C: ("JMP", "long", 4),
    0x5D: ("EOR", "abs_x", 3),
    0x5E: ("LSR", "abs_x", 3),
    0x5F: ("EOR", "long_x", 4),
    0x60: ("RTS", "imp", 1),
    0x61: ("ADC", "dp_x_ind", 2),
    0x62: ("PER", "rel16", 3),
    0x63: ("ADC", "sr", 2),
    0x64: ("STZ", "dp", 2),
    0x65: ("ADC", "dp", 2),
    0x66: ("ROR", "dp", 2),
    0x67: ("ADC", "dp_ind_long", 2),
    0x68: ("PLA", "imp", 1),
    0x69: ("ADC", "imm_m", 2),
    0x6A: ("ROR", "acc", 1),
    0x6B: ("RTL", "imp", 1),
    0x6C: ("JMP", "abs_ind", 3),
    0x6D: ("ADC", "abs", 3),
    0x6E: ("ROR", "abs", 3),
    0x6F: ("ADC", "long", 4),
    0x70: ("BVS", "rel8", 2),
    0x71: ("ADC", "dp_ind_y", 2),
    0x72: ("ADC", "dp_ind", 2),
    0x73: ("ADC", "sr_ind_y", 2),
    0x74: ("STZ", "dp_x", 2),
    0x75: ("ADC", "dp_x", 2),
    0x76: ("ROR", "dp_x", 2),
    0x77: ("ADC", "dp_ind_long_y", 2),
    0x78: ("SEI", "imp", 1),
    0x79: ("ADC", "abs_y", 3),
    0x7A: ("PLY", "imp", 1),
    0x7B: ("TDC", "imp", 1),
    0x7C: ("JMP", "abs_x_ind", 3),
    0x7D: ("ADC", "abs_x", 3),
    0x7E: ("ROR", "abs_x", 3),
    0x7F: ("ADC", "long_x", 4),
    0x80: ("BRA", "rel8", 2),
    0x81: ("STA", "dp_x_ind", 2),
    0x82: ("BRL", "rel16", 3),
    0x83: ("STA", "sr", 2),
    0x84: ("STY", "dp", 2),
    0x85: ("STA", "dp", 2),
    0x86: ("STX", "dp", 2),
    0x87: ("STA", "dp_ind_long", 2),
    0x88: ("DEY", "imp", 1),
    0x89: ("BIT", "imm_m", 2),
    0x8A: ("TXA", "imp", 1),
    0x8B: ("PHB", "imp", 1),
    0x8C: ("STY", "abs", 3),
    0x8D: ("STA", "abs", 3),
    0x8E: ("STX", "abs", 3),
    0x8F: ("STA", "long", 4),
    0x90: ("BCC", "rel8", 2),
    0x91: ("STA", "dp_ind_y", 2),
    0x92: ("STA", "dp_ind", 2),
    0x93: ("STA", "sr_ind_y", 2),
    0x94: ("STY", "dp_x", 2),
    0x95: ("STA", "dp_x", 2),
    0x96: ("STX", "dp_y", 2),
    0x97: ("STA", "dp_ind_long_y", 2),
    0x98: ("TYA", "imp", 1),
    0x99: ("STA", "abs_y", 3),
    0x9A: ("TXS", "imp", 1),
    0x9B: ("TXY", "imp", 1),
    0x9C: ("STZ", "abs", 3),
    0x9D: ("STA", "abs_x", 3),
    0x9E: ("STZ", "abs_x", 3),
    0x9F: ("STA", "long_x", 4),
    0xA0: ("LDY", "imm_x", 2),
    0xA1: ("LDA", "dp_x_ind", 2),
    0xA2: ("LDX", "imm_x", 2),
    0xA3: ("LDA", "sr", 2),
    0xA4: ("LDY", "dp", 2),
    0xA5: ("LDA", "dp", 2),
    0xA6: ("LDX", "dp", 2),
    0xA7: ("LDA", "dp_ind_long", 2),
    0xA8: ("TAY", "imp", 1),
    0xA9: ("LDA", "imm_m", 2),
    0xAA: ("TAX", "imp", 1),
    0xAB: ("PLB", "imp", 1),
    0xAC: ("LDY", "abs", 3),
    0xAD: ("LDA", "abs", 3),
    0xAE: ("LDX", "abs", 3),
    0xAF: ("LDA", "long", 4),
    0xB0: ("BCS", "rel8", 2),
    0xB1: ("LDA", "dp_ind_y", 2),
    0xB2: ("LDA", "dp_ind", 2),
    0xB3: ("LDA", "sr_ind_y", 2),
    0xB4: ("LDY", "dp_x", 2),
    0xB5: ("LDA", "dp_x", 2),
    0xB6: ("LDX", "dp_y", 2),
    0xB7: ("LDA", "dp_ind_long_y", 2),
    0xB8: ("CLV", "imp", 1),
    0xB9: ("LDA", "abs_y", 3),
    0xBA: ("TSX", "imp", 1),
    0xBB: ("TYX", "imp", 1),
    0xBC: ("LDY", "abs_x", 3),
    0xBD: ("LDA", "abs_x", 3),
    0xBE: ("LDX", "abs_y", 3),
    0xBF: ("LDA", "long_x", 4),
    0xC0: ("CPY", "imm_x", 2),
    0xC1: ("CMP", "dp_x_ind", 2),
    0xC2: ("REP", "imm8", 2),
    0xC3: ("CMP", "sr", 2),
    0xC4: ("CPY", "dp", 2),
    0xC5: ("CMP", "dp", 2),
    0xC6: ("DEC", "dp", 2),
    0xC7: ("CMP", "dp_ind_long", 2),
    0xC8: ("INY", "imp", 1),
    0xC9: ("CMP", "imm_m", 2),
    0xCA: ("DEX", "imp", 1),
    0xCB: ("WAI", "imp", 1),
    0xCC: ("CPY", "abs", 3),
    0xCD: ("CMP", "abs", 3),
    0xCE: ("DEC", "abs", 3),
    0xCF: ("CMP", "long", 4),
    0xD0: ("BNE", "rel8", 2),
    0xD1: ("CMP", "dp_ind_y", 2),
    0xD2: ("CMP", "dp_ind", 2),
    0xD3: ("CMP", "sr_ind_y", 2),
    0xD4: ("PEI", "dp_ind", 2),
    0xD5: ("CMP", "dp_x", 2),
    0xD6: ("DEC", "dp_x", 2),
    0xD7: ("CMP", "dp_ind_long_y", 2),
    0xD8: ("CLD", "imp", 1),
    0xD9: ("CMP", "abs_y", 3),
    0xDA: ("PHX", "imp", 1),
    0xDB: ("STP", "imp", 1),
    0xDC: ("JMP", "abs_ind_long", 3),
    0xDD: ("CMP", "abs_x", 3),
    0xDE: ("DEC", "abs_x", 3),
    0xDF: ("CMP", "long_x", 4),
    0xE0: ("CPX", "imm_x", 2),
    0xE1: ("SBC", "dp_x_ind", 2),
    0xE2: ("SEP", "imm8", 2),
    0xE3: ("SBC", "sr", 2),
    0xE4: ("CPX", "dp", 2),
    0xE5: ("SBC", "dp", 2),
    0xE6: ("INC", "dp", 2),
    0xE7: ("SBC", "dp_ind_long", 2),
    0xE8: ("INX", "imp", 1),
    0xE9: ("SBC", "imm_m", 2),
    0xEA: ("NOP", "imp", 1),
    0xEB: ("XBA", "imp", 1),
    0xEC: ("CPX", "abs", 3),
    0xED: ("SBC", "abs", 3),
    0xEE: ("INC", "abs", 3),
    0xEF: ("SBC", "long", 4),
    0xF0: ("BEQ", "rel8", 2),
    0xF1: ("SBC", "dp_ind_y", 2),
    0xF2: ("SBC", "dp_ind", 2),
    0xF3: ("SBC", "sr_ind_y", 2),
    0xF4: ("PEA", "abs", 3),
    0xF5: ("SBC", "dp_x", 2),
    0xF6: ("INC", "dp_x", 2),
    0xF7: ("SBC", "dp_ind_long_y", 2),
    0xF8: ("SED", "imp", 1),
    0xF9: ("SBC", "abs_y", 3),
    0xFA: ("PLX", "imp", 1),
    0xFB: ("XCE", "imp", 1),
    0xFC: ("JSR", "abs_x_ind", 3),
    0xFD: ("SBC", "abs_x", 3),
    0xFE: ("INC", "abs_x", 3),
    0xFF: ("SBC", "long_x", 4),
}


def get_instruction_size(opcode: int, flag_m: bool, flag_x: bool) -> int:
    """Get the size of an instruction in bytes."""
    if opcode not in OPCODE_INFO:
        return 1

    _, mode, base_size = OPCODE_INFO[opcode]

    if mode == "imm_m":
        return base_size if flag_m else base_size + 1
    elif mode == "imm_x":
        return base_size if flag_x else base_size + 1

    return base_size


def disassemble(mem: 'Memory', bank: int, addr: int,
                flag_m: bool = True, flag_x: bool = True) -> Tuple[str, int]:
    """
    Disassemble a single instruction.

    Args:
        mem: Memory object
        bank: Program bank
        addr: Program counter address
        flag_m: M flag state (True = 8-bit accumulator)
        flag_x: X flag state (True = 8-bit index)

    Returns:
        (disassembly_string, instruction_size)
    """
    ea = (bank << 16) | addr
    opcode = mem.read(ea)

    if opcode not in OPCODE_INFO:
        return (f"???  ${opcode:02X}", 1)

    mnemonic, mode, base_size = OPCODE_INFO[opcode]
    size = get_instruction_size(opcode, flag_m, flag_x)

    # Read operand bytes
    operand_bytes = []
    for i in range(1, size):
        operand_bytes.append(mem.read((bank << 16) | ((addr + i) & 0xFFFF)))

    # Format operand based on addressing mode
    operand = _format_operand(mode, operand_bytes, addr, size, flag_m, flag_x)

    if operand:
        return (f"{mnemonic} {operand}", size)
    else:
        return (mnemonic, size)


def _format_operand(mode: str, operand_bytes: list, addr: int, size: int,
                    flag_m: bool, flag_x: bool) -> str:
    """Format the operand string based on addressing mode."""

    def byte1() -> int:
        return operand_bytes[0] if len(operand_bytes) > 0 else 0

    def word() -> int:
        if len(operand_bytes) >= 2:
            return operand_bytes[0] | (operand_bytes[1] << 8)
        return 0

    def long_addr() -> int:
        if len(operand_bytes) >= 3:
            return operand_bytes[0] | (operand_bytes[1] << 8) | (operand_bytes[2] << 16)
        return 0

    if mode == "imp" or mode == "acc":
        return ""

    elif mode == "imm8":
        return f"#${byte1():02X}"

    elif mode == "imm_m":
        if flag_m:
            return f"#${byte1():02X}"
        else:
            return f"#${word():04X}"

    elif mode == "imm_x":
        if flag_x:
            return f"#${byte1():02X}"
        else:
            return f"#${word():04X}"

    elif mode == "dp":
        return f"${byte1():02X}"

    elif mode == "dp_x":
        return f"${byte1():02X},X"

    elif mode == "dp_y":
        return f"${byte1():02X},Y"

    elif mode == "dp_ind":
        return f"(${byte1():02X})"

    elif mode == "dp_ind_long":
        return f"[${byte1():02X}]"

    elif mode == "dp_x_ind":
        return f"(${byte1():02X},X)"

    elif mode == "dp_ind_y":
        return f"(${byte1():02X}),Y"

    elif mode == "dp_ind_long_y":
        return f"[${byte1():02X}],Y"

    elif mode == "abs":
        return f"${word():04X}"

    elif mode == "abs_x":
        return f"${word():04X},X"

    elif mode == "abs_y":
        return f"${word():04X},Y"

    elif mode == "abs_ind":
        return f"(${word():04X})"

    elif mode == "abs_ind_long":
        return f"[${word():04X}]"

    elif mode == "abs_x_ind":
        return f"(${word():04X},X)"

    elif mode == "long":
        return f"${long_addr():06X}"

    elif mode == "long_x":
        return f"${long_addr():06X},X"

    elif mode == "sr":
        return f"${byte1():02X},S"

    elif mode == "sr_ind_y":
        return f"(${byte1():02X},S),Y"

    elif mode == "rel8":
        # Calculate target address
        offset = byte1()
        if offset & 0x80:
            offset = offset - 256
        target = (addr + size + offset) & 0xFFFF
        return f"${target:04X}"

    elif mode == "rel16":
        offset = word()
        if offset & 0x8000:
            offset = offset - 65536
        target = (addr + size + offset) & 0xFFFF
        return f"${target:04X}"

    elif mode == "blk":
        # Block move: destination, source banks
        return f"${byte1():02X},${operand_bytes[1]:02X}" if len(operand_bytes) >= 2 else ""

    elif mode == "sig":
        return f"#${byte1():02X}"

    return ""


def disassemble_range(mem: 'Memory', bank: int, start: int, end: int,
                      flag_m: bool = True, flag_x: bool = True) -> list:
    """
    Disassemble a range of addresses.

    Returns list of (address, disassembly, size) tuples.
    """
    results = []
    addr = start

    while addr < end:
        disasm, size = disassemble(mem, bank, addr, flag_m, flag_x)
        results.append((addr, disasm, size))
        addr += size

    return results
