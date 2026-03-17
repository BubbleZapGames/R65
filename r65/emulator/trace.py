# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Execution Trace Logger for 65816 CPU.
"""

import sys
from typing import TYPE_CHECKING, TextIO, Optional

from .disasm import disassemble

if TYPE_CHECKING:
    from .cpu import CPU65816


class TraceLogger:
    """
    Logs CPU execution trace in a formatted output.

    Output format:
    BB:AAAA  MNEMONIC OPERAND   A:XXXX X:XXXX Y:XXXX SP:XXXX D:XXXX DB:XX PB:XX nvMXdizC
    """

    def __init__(self, output: Optional[TextIO] = None):
        """
        Initialize trace logger.

        Args:
            output: Output stream (defaults to stdout)
        """
        self.output = output or sys.stdout
        self.instruction_count = 0

    def log(self, cpu: 'CPU65816'):
        """Log the current CPU state before executing an instruction."""
        # Get disassembly
        disasm, _ = disassemble(
            cpu.memory, cpu.PBR, cpu.PC,
            cpu.flag_m, cpu.flag_x
        )

        # Format registers
        if cpu.flag_m:
            a_str = f"A:{cpu.A & 0xFF:02X}  "
        else:
            a_str = f"A:{cpu.A:04X}"

        if cpu.flag_x:
            x_str = f"X:{cpu.X:02X}  "
            y_str = f"Y:{cpu.Y:02X}  "
        else:
            x_str = f"X:{cpu.X:04X}"
            y_str = f"Y:{cpu.Y:04X}"

        # Format flags: lowercase = 0, uppercase = 1
        flags = ""
        flags += "N" if cpu.flag_n else "n"
        flags += "V" if cpu.flag_v else "v"
        flags += "M" if cpu.flag_m else "m"
        flags += "X" if cpu.flag_x else "x"
        flags += "D" if cpu.flag_d else "d"
        flags += "I" if cpu.flag_i else "i"
        flags += "Z" if cpu.flag_z else "z"
        flags += "C" if cpu.flag_c else "c"

        # Emulation mode indicator
        e_flag = "E" if cpu.emulation_mode else "e"

        # Build trace line
        line = (
            f"{cpu.PBR:02X}:{cpu.PC:04X}  "
            f"{disasm:<20s} "
            f"{a_str} {x_str} {y_str} "
            f"SP:{cpu.SP:04X} D:{cpu.D:04X} "
            f"DB:{cpu.DBR:02X} "
            f"{flags}{e_flag}"
        )

        self.output.write(line + "\n")
        self.instruction_count += 1

    def log_header(self):
        """Print a header line for the trace output."""
        header = (
            "ADDR     INSTRUCTION          "
            "A     X     Y     SP    D     DB  FLAGS"
        )
        self.output.write(header + "\n")
        self.output.write("-" * len(header) + "\n")

    def log_separator(self, label: str = ""):
        """Print a separator line."""
        if label:
            self.output.write(f"--- {label} ---\n")
        else:
            self.output.write("-" * 70 + "\n")

    def log_state(self, cpu: 'CPU65816', label: str = ""):
        """Log detailed CPU state."""
        if label:
            self.output.write(f"=== {label} ===\n")

        self.output.write(f"  A  = ${cpu.A:04X}  (lo=${cpu.A & 0xFF:02X}, hi=${(cpu.A >> 8) & 0xFF:02X})\n")
        self.output.write(f"  X  = ${cpu.X:04X}\n")
        self.output.write(f"  Y  = ${cpu.Y:04X}\n")
        self.output.write(f"  SP = ${cpu.SP:04X}\n")
        self.output.write(f"  PC = ${cpu.PBR:02X}:{cpu.PC:04X}\n")
        self.output.write(f"  D  = ${cpu.D:04X}\n")
        self.output.write(f"  DBR = ${cpu.DBR:02X}\n")
        self.output.write(f"  P  = ${cpu.P:02X} (")

        flags = []
        if cpu.flag_n: flags.append("N")
        if cpu.flag_v: flags.append("V")
        if cpu.flag_m: flags.append("M")
        if cpu.flag_x: flags.append("X")
        if cpu.flag_d: flags.append("D")
        if cpu.flag_i: flags.append("I")
        if cpu.flag_z: flags.append("Z")
        if cpu.flag_c: flags.append("C")

        self.output.write(",".join(flags) if flags else "none")
        self.output.write(")\n")

        self.output.write(f"  Emulation mode: {cpu.emulation_mode}\n")
        self.output.write(f"  Cycles: {cpu.cycles}\n")

    def log_memory(self, cpu: 'CPU65816', bank: int, start: int, length: int = 16):
        """Log a region of memory."""
        self.output.write(f"Memory ${bank:02X}:{start:04X}-${bank:02X}:{(start + length - 1) & 0xFFFF:04X}:\n")

        for i in range(0, length, 16):
            addr = (start + i) & 0xFFFF
            self.output.write(f"  ${bank:02X}:{addr:04X}: ")

            # Hex bytes
            hex_part = ""
            ascii_part = ""
            for j in range(16):
                if i + j < length:
                    byte = cpu.memory.read((bank << 16) | ((addr + j) & 0xFFFF))
                    hex_part += f"{byte:02X} "
                    ascii_part += chr(byte) if 0x20 <= byte <= 0x7E else "."
                else:
                    hex_part += "   "

            self.output.write(hex_part)
            self.output.write(f" |{ascii_part}|\n")

    def get_stats(self) -> dict:
        """Get trace statistics."""
        return {
            "instruction_count": self.instruction_count
        }


class CompactTraceLogger(TraceLogger):
    """Compact trace format - one line per instruction, minimal info."""

    def log(self, cpu: 'CPU65816'):
        disasm, _ = disassemble(
            cpu.memory, cpu.PBR, cpu.PC,
            cpu.flag_m, cpu.flag_x
        )

        line = f"{cpu.PBR:02X}:{cpu.PC:04X} {disasm}"
        self.output.write(line + "\n")
        self.instruction_count += 1


class NullTraceLogger(TraceLogger):
    """No-op logger for when tracing is disabled."""

    def log(self, cpu: 'CPU65816'):
        self.instruction_count += 1

    def log_header(self):
        pass

    def log_separator(self, label: str = ""):
        pass

    def log_state(self, cpu: 'CPU65816', label: str = ""):
        pass

    def log_memory(self, cpu: 'CPU65816', bank: int, start: int, length: int = 16):
        pass
