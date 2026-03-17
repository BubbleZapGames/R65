# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
ROM Execution Comparison Tool.

Compares instruction-by-instruction execution between two ROMs to find divergence.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TextIO, Tuple, List
import sys

from .cpu import CPU65816, StopExecution, WaitForInterrupt
from .memory import SNESMemory, detect_mapping
from .disasm import disassemble, get_instruction_size, OPCODE_INFO

if TYPE_CHECKING:
    pass


@dataclass
class NormalizedTrace:
    """
    Captures instruction state without absolute addresses.

    This allows comparison between ROMs with different code locations
    while still detecting behavioral differences.
    """
    instruction_number: int
    opcode: int
    mnemonic: str

    # Register state AFTER instruction execution
    A: int
    X: int
    Y: int
    SP: int
    P: int  # Status flags

    # Branch/jump tracking
    branch_taken: Optional[bool]  # True/False for branches, None for non-branch

    # For debugging - original address (not used for comparison)
    original_bank: int
    original_addr: int

    # Memory access info (for debugging)
    memory_read_addr: Optional[int] = None
    memory_read_value: Optional[int] = None

    def flags_str(self) -> str:
        """Format flags as string (lowercase=0, uppercase=1)."""
        flags = ""
        flags += "N" if self.P & 0x80 else "n"
        flags += "V" if self.P & 0x40 else "v"
        flags += "M" if self.P & 0x20 else "m"
        flags += "X" if self.P & 0x10 else "x"
        flags += "D" if self.P & 0x08 else "d"
        flags += "I" if self.P & 0x04 else "i"
        flags += "Z" if self.P & 0x02 else "z"
        flags += "C" if self.P & 0x01 else "c"
        return flags


class CompareTraceLogger:
    """
    Captures normalized traces from CPU execution for comparison.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.traces: List[NormalizedTrace] = []
        self.instruction_count = 0

        # Pre-instruction state snapshot
        self._pre_pc: int = 0
        self._pre_pbr: int = 0
        self._pre_sp: int = 0

    def capture_pre_state(self, cpu: CPU65816) -> dict:
        """Snapshot state before instruction execution."""
        return {
            'pc': cpu.PC,
            'pbr': cpu.PBR,
            'sp': cpu.SP,
            'opcode': cpu.memory.read((cpu.PBR << 16) | cpu.PC),
        }

    def capture_post_state(self, cpu: CPU65816, pre: dict) -> NormalizedTrace:
        """Create normalized trace after instruction execution."""
        opcode = pre['opcode']

        # Get mnemonic
        if opcode in OPCODE_INFO:
            mnemonic = OPCODE_INFO[opcode][0]
        else:
            mnemonic = "???"

        # Determine if branch was taken
        branch_taken = None
        if mnemonic in ('BRA', 'BRL'):
            branch_taken = True  # Always taken
        elif mnemonic in ('BEQ', 'BNE', 'BCS', 'BCC', 'BMI', 'BPL', 'BVS', 'BVC'):
            # Check if branch was taken by comparing PC
            # If branch taken, PC won't be sequential
            old_pc = pre['pc']
            # Get instruction size
            flag_m = bool(cpu.P & 0x20)
            flag_x = bool(cpu.P & 0x10)
            inst_size = get_instruction_size(opcode, flag_m, flag_x)
            expected_pc = (old_pc + inst_size) & 0xFFFF
            branch_taken = (cpu.PC != expected_pc)

        trace = NormalizedTrace(
            instruction_number=self.instruction_count,
            opcode=opcode,
            mnemonic=mnemonic,
            A=cpu.A,
            X=cpu.X,
            Y=cpu.Y,
            SP=cpu.SP,
            P=cpu.P,
            branch_taken=branch_taken,
            original_bank=pre['pbr'],
            original_addr=pre['pc'],
        )

        self.traces.append(trace)
        self.instruction_count += 1

        return trace


@dataclass
class Divergence:
    """Describes a divergence point between two ROMs."""
    instruction_number: int
    trace1: NormalizedTrace
    trace2: NormalizedTrace
    differences: List[str]


class RomComparator:
    """
    Runs two ROMs in parallel and compares their execution traces.
    """

    def __init__(self, rom1_data: bytes, rom2_data: bytes,
                 rom1_name: str = "ROM 1", rom2_name: str = "ROM 2",
                 mapping: str = "auto"):
        """
        Initialize comparator with two ROM images.

        Args:
            rom1_data: First ROM data (typically original)
            rom2_data: Second ROM data (typically port)
            rom1_name: Display name for first ROM
            rom2_name: Display name for second ROM
            mapping: Memory mapping ("lorom", "hirom", or "auto")
        """
        self.rom1_name = rom1_name
        self.rom2_name = rom2_name

        # Detect mapping if auto
        if mapping == "auto":
            mapping1 = detect_mapping(rom1_data)
            mapping2 = detect_mapping(rom2_data)
            if mapping1 != mapping2:
                print(f"Warning: ROM mappings differ: {mapping1} vs {mapping2}",
                      file=sys.stderr)
            mapping = mapping1

        self.mapping = mapping

        # Create separate memory and CPU for each ROM
        self.mem1 = SNESMemory(rom1_data, mapping)
        self.mem2 = SNESMemory(rom2_data, mapping)

        self.cpu1 = CPU65816(self.mem1)
        self.cpu2 = CPU65816(self.mem2)

        # Trace loggers
        self.logger1 = CompareTraceLogger(rom1_name)
        self.logger2 = CompareTraceLogger(rom2_name)

        # Divergences found
        self.divergences: List[Divergence] = []

    def reset(self):
        """Reset both CPUs to initial state."""
        self.cpu1.reset()
        self.cpu2.reset()
        self.logger1 = CompareTraceLogger(self.rom1_name)
        self.logger2 = CompareTraceLogger(self.rom2_name)
        self.divergences = []

    def enable_nmi(self, enabled: bool = True):
        """Enable automatic NMI timing for both CPUs."""
        self.cpu1.enable_auto_nmi(enabled)
        self.cpu2.enable_auto_nmi(enabled)

    def compare_traces(self, t1: NormalizedTrace, t2: NormalizedTrace) -> List[str]:
        """
        Compare two traces and return list of differences.

        Returns empty list if traces match.
        """
        differences = []

        # Must execute same instruction type
        if t1.opcode != t2.opcode:
            differences.append(f"Opcode: ${t1.opcode:02X} ({t1.mnemonic}) vs ${t2.opcode:02X} ({t2.mnemonic})")
            return differences  # Can't meaningfully compare further

        # Check registers after execution
        if t1.A != t2.A:
            differences.append(f"A register: ${t1.A:04X} vs ${t2.A:04X}")

        if t1.X != t2.X:
            differences.append(f"X register: ${t1.X:04X} vs ${t2.X:04X}")

        if t1.Y != t2.Y:
            differences.append(f"Y register: ${t1.Y:04X} vs ${t2.Y:04X}")

        # Compare flags (but ignore exact SP value, compare deltas instead would be complex)
        if t1.P != t2.P:
            flags1 = t1.flags_str()
            flags2 = t2.flags_str()
            differences.append(f"Flags: {flags1} vs {flags2}")

        # Compare branch decisions
        if t1.branch_taken != t2.branch_taken:
            b1 = "taken" if t1.branch_taken else "not taken"
            b2 = "taken" if t2.branch_taken else "not taken"
            differences.append(f"Branch: {b1} vs {b2}")

        return differences

    def run(self, max_instructions: int = 1000,
            continue_on_diverge: bool = False,
            verbose: bool = False,
            output: TextIO = None) -> Optional[Divergence]:
        """
        Run both ROMs in lockstep and compare execution.

        Args:
            max_instructions: Maximum instructions to execute
            continue_on_diverge: Keep running after finding divergence
            verbose: Print parallel trace output
            output: Output stream (default stdout)

        Returns:
            First divergence found, or None if no divergence
        """
        if output is None:
            output = sys.stdout

        first_divergence = None

        try:
            for i in range(max_instructions):
                # Check if either CPU stopped
                if self.cpu1.stopped or self.cpu1.waiting:
                    output.write(f"\n{self.rom1_name} stopped at instruction #{i}\n")
                    break
                if self.cpu2.stopped or self.cpu2.waiting:
                    output.write(f"\n{self.rom2_name} stopped at instruction #{i}\n")
                    break

                # Capture pre-state
                pre1 = self.logger1.capture_pre_state(self.cpu1)
                pre2 = self.logger2.capture_pre_state(self.cpu2)

                # Execute one instruction on each CPU
                self.cpu1.step()
                self.cpu2.step()

                # Capture post-state
                t1 = self.logger1.capture_post_state(self.cpu1, pre1)
                t2 = self.logger2.capture_post_state(self.cpu2, pre2)

                # Verbose output
                if verbose:
                    disasm1, _ = disassemble(self.mem1, t1.original_bank, t1.original_addr,
                                            bool(self.cpu1.P & 0x20), bool(self.cpu1.P & 0x10))
                    disasm2, _ = disassemble(self.mem2, t2.original_bank, t2.original_addr,
                                            bool(self.cpu2.P & 0x20), bool(self.cpu2.P & 0x10))
                    output.write(f"#{i:6d}  {t1.original_bank:02X}:{t1.original_addr:04X} {disasm1:<20s} | "
                                f"{t2.original_bank:02X}:{t2.original_addr:04X} {disasm2:<20s}\n")

                # Compare traces
                differences = self.compare_traces(t1, t2)

                if differences:
                    divergence = Divergence(
                        instruction_number=i,
                        trace1=t1,
                        trace2=t2,
                        differences=differences
                    )
                    self.divergences.append(divergence)

                    if first_divergence is None:
                        first_divergence = divergence

                    if not continue_on_diverge:
                        return first_divergence

        except StopExecution as e:
            output.write(f"\nCPU stopped: {e}\n")
        except WaitForInterrupt:
            output.write(f"\nCPU waiting for interrupt\n")
        except KeyboardInterrupt:
            output.write(f"\nInterrupted by user\n")

        return first_divergence

    def format_divergence(self, div: Divergence,
                          context_before: int = 5,
                          output: TextIO = None):
        """
        Format divergence information for display.

        Args:
            div: Divergence to format
            context_before: Number of trace entries to show before divergence
            output: Output stream (default stdout)
        """
        if output is None:
            output = sys.stdout

        output.write("\n")
        output.write("=" * 70 + "\n")
        output.write(f"DIVERGENCE at instruction #{div.instruction_number}\n")
        output.write("=" * 70 + "\n\n")

        # Show context (preceding instructions)
        start_idx = max(0, div.instruction_number - context_before)

        if start_idx < div.instruction_number:
            output.write("Context (preceding instructions):\n")
            output.write("-" * 70 + "\n")

            for i in range(start_idx, div.instruction_number):
                t1 = self.logger1.traces[i]
                t2 = self.logger2.traces[i]

                disasm1, _ = disassemble(self.mem1, t1.original_bank, t1.original_addr,
                                        bool(t1.P & 0x20), bool(t1.P & 0x10))
                disasm2, _ = disassemble(self.mem2, t2.original_bank, t2.original_addr,
                                        bool(t2.P & 0x20), bool(t2.P & 0x10))

                output.write(f"  #{i:6d}  {t1.original_bank:02X}:{t1.original_addr:04X} {disasm1:<20s}")
                output.write(f" | {t2.original_bank:02X}:{t2.original_addr:04X} {disasm2:<20s}\n")

            output.write("\n")

        # Show divergent instruction
        t1 = div.trace1
        t2 = div.trace2

        disasm1, _ = disassemble(self.mem1, t1.original_bank, t1.original_addr,
                                bool(t1.P & 0x20), bool(t1.P & 0x10))
        disasm2, _ = disassemble(self.mem2, t2.original_bank, t2.original_addr,
                                bool(t2.P & 0x20), bool(t2.P & 0x10))

        output.write(f"Divergent instruction:\n")
        output.write(f"  {self.rom1_name}:\n")
        output.write(f"    {t1.original_bank:02X}:{t1.original_addr:04X}  {disasm1}\n")
        output.write(f"    A=${t1.A:04X} X=${t1.X:04X} Y=${t1.Y:04X} SP=${t1.SP:04X} {t1.flags_str()}\n")
        output.write(f"\n")
        output.write(f"  {self.rom2_name}:\n")
        output.write(f"    {t2.original_bank:02X}:{t2.original_addr:04X}  {disasm2}\n")
        output.write(f"    A=${t2.A:04X} X=${t2.X:04X} Y=${t2.Y:04X} SP=${t2.SP:04X} {t2.flags_str()}\n")
        output.write(f"\n")

        output.write(f"Differences:\n")
        for diff in div.differences:
            output.write(f"  - {diff}\n")
        output.write("\n")


def load_rom_with_header_detection(path: str) -> bytes:
    """
    Load ROM file, auto-detecting and stripping SMC header if present.

    Args:
        path: Path to ROM file

    Returns:
        ROM data with header stripped if detected
    """
    from pathlib import Path

    data = Path(path).read_bytes()

    # Detect 512-byte copier header (file size % 1024 == 512)
    if len(data) % 1024 == 512:
        return data[512:]
    return data
