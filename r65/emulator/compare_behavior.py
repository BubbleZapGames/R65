# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Behavioral Comparison Tool for ROMs.

Compares what ROMs DO (memory writes, HW register writes, function returns)
rather than comparing instruction-by-instruction. This allows comparing
ROMs compiled by different compilers that have different code but should
have equivalent behavior.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, TextIO, List, Dict, Set, Tuple
from enum import Enum
import sys

from .cpu import CPU65816, StopExecution, WaitForInterrupt
from .memory import SNESMemory, detect_mapping
from .disasm import disassemble, OPCODE_INFO

if TYPE_CHECKING:
    pass


class EventType(Enum):
    MEMORY_WRITE = "mem_write"
    HW_WRITE = "hw_write"
    FUNCTION_CALL = "call"
    FUNCTION_RETURN = "return"


@dataclass
class BehaviorEvent:
    """A single behavioral event (memory write, HW write, function call/return)."""
    event_type: EventType
    instruction_num: int

    # For memory/HW writes
    address: Optional[int] = None
    bank: Optional[int] = None
    value: Optional[int] = None
    size: int = 1  # 1 or 2 bytes

    # For function calls/returns
    target_addr: Optional[int] = None
    return_addr: Optional[int] = None
    a_reg: Optional[int] = None
    x_reg: Optional[int] = None
    y_reg: Optional[int] = None

    def __str__(self) -> str:
        if self.event_type == EventType.MEMORY_WRITE:
            if self.size == 2:
                return f"MEM[${self.bank:02X}:{self.address:04X}] = ${self.value:04X}"
            return f"MEM[${self.bank:02X}:{self.address:04X}] = ${self.value:02X}"
        elif self.event_type == EventType.HW_WRITE:
            return f"HW[${self.address:04X}] = ${self.value:02X}"
        elif self.event_type == EventType.FUNCTION_CALL:
            return f"CALL ${self.target_addr:04X} (A=${self.a_reg:04X} X=${self.x_reg:04X} Y=${self.y_reg:04X})"
        elif self.event_type == EventType.FUNCTION_RETURN:
            return f"RET -> ${self.return_addr:04X} (A=${self.a_reg:04X})"
        return str(self.event_type)


class InstrumentedMemory(SNESMemory):
    """Memory that logs writes for behavioral comparison."""

    def __init__(self, rom_data: bytes, mapping: str = "lorom"):
        super().__init__(rom_data, mapping)
        self.write_log: List[BehaviorEvent] = []
        self.instruction_num = 0
        self.log_enabled = True

        # Track which addresses to monitor (None = all)
        self.monitored_ranges: Optional[List[Tuple[int, int, int, int]]] = None  # [(bank_start, bank_end, addr_start, addr_end), ...]

        # HW registers of interest
        self.hw_registers: Set[int] = {
            0x2100, 0x2101, 0x2102, 0x2103,  # PPU
            0x2105, 0x2106, 0x2107, 0x2108, 0x2109, 0x210A, 0x210B, 0x210C,
            0x210D, 0x210E, 0x210F, 0x2110, 0x2111, 0x2112, 0x2113, 0x2114,
            0x2115, 0x2116, 0x2117, 0x2118, 0x2119, 0x211A, 0x211B, 0x211C,
            0x211D, 0x211E, 0x211F, 0x2120, 0x2121, 0x2122,
            0x2123, 0x2124, 0x2125, 0x2126, 0x2127, 0x2128, 0x2129, 0x212A,
            0x212B, 0x212C, 0x212D, 0x212E, 0x212F, 0x2130, 0x2131, 0x2132, 0x2133,
            0x4200, 0x4201, 0x4202, 0x4203, 0x4204, 0x4205, 0x4206,  # CPU
            0x4207, 0x4208, 0x4209, 0x420A, 0x420B, 0x420C, 0x420D,
        }

    def write(self, addr: int, value: int):
        """Override write to log behavioral events."""
        bank = (addr >> 16) & 0xFF
        offset = addr & 0xFFFF
        value &= 0xFF

        if self.log_enabled:
            # Check if this is a HW register write
            if (bank <= 0x3F or 0x80 <= bank <= 0xBF) and 0x2100 <= offset <= 0x44FF:
                if offset in self.hw_registers:
                    event = BehaviorEvent(
                        event_type=EventType.HW_WRITE,
                        instruction_num=self.instruction_num,
                        address=offset,
                        bank=bank,
                        value=value
                    )
                    self.write_log.append(event)

            # Check if this is WRAM write (game state)
            elif self._is_wram_write(bank, offset):
                if self._should_log_address(bank, offset):
                    event = BehaviorEvent(
                        event_type=EventType.MEMORY_WRITE,
                        instruction_num=self.instruction_num,
                        address=offset,
                        bank=bank,
                        value=value
                    )
                    self.write_log.append(event)

        # Do the actual write
        super().write(addr, value)

    def write16(self, addr: int, value: int):
        """Override 16-bit write to log as single event."""
        bank = (addr >> 16) & 0xFF
        offset = addr & 0xFFFF
        value &= 0xFFFF

        if self.log_enabled and self._is_wram_write(bank, offset):
            if self._should_log_address(bank, offset):
                event = BehaviorEvent(
                    event_type=EventType.MEMORY_WRITE,
                    instruction_num=self.instruction_num,
                    address=offset,
                    bank=bank,
                    value=value,
                    size=2
                )
                self.write_log.append(event)

        # Do the actual write
        super().write16(addr, value)

    def _is_wram_write(self, bank: int, addr: int) -> bool:
        """Check if address is WRAM."""
        if bank == 0x7E or bank == 0x7F:
            return True
        if (bank <= 0x3F or 0x80 <= bank <= 0xBF) and addr < 0x2000:
            return True
        return False

    def _should_log_address(self, bank: int, addr: int) -> bool:
        """Check if address should be logged."""
        if self.monitored_ranges is None:
            return True
        for bs, be, as_, ae in self.monitored_ranges:
            if bs <= bank <= be and as_ <= addr <= ae:
                return True
        return False


@dataclass
class FunctionMapping:
    """Maps a function between two ROMs."""
    name: str
    addr1: int  # Address in ROM1
    addr2: int  # Address in ROM2
    bank1: int = 0
    bank2: int = 0


class BehaviorComparator:
    """
    Compares ROM behavior by tracking what they DO rather than
    what instructions they execute.
    """

    def __init__(self, rom1_data: bytes, rom2_data: bytes,
                 rom1_name: str = "ROM 1", rom2_name: str = "ROM 2",
                 mapping: str = "auto"):
        self.rom1_name = rom1_name
        self.rom2_name = rom2_name

        # Detect mapping
        if mapping == "auto":
            mapping = detect_mapping(rom1_data)
        self.mapping = mapping

        # Create instrumented memory
        self.mem1 = InstrumentedMemory(rom1_data, mapping)
        self.mem2 = InstrumentedMemory(rom2_data, mapping)

        # Create CPUs
        self.cpu1 = CPU65816(self.mem1)
        self.cpu2 = CPU65816(self.mem2)

        # Function mappings (equivalent functions between ROMs)
        self.function_mappings: List[FunctionMapping] = []

        # Call stacks for tracking function calls/returns
        self._call_stack1: List[int] = []
        self._call_stack2: List[int] = []

    def reset(self):
        """Reset both CPUs."""
        self.cpu1.reset()
        self.cpu2.reset()
        self.mem1.write_log = []
        self.mem2.write_log = []
        self._call_stack1 = []
        self._call_stack2 = []

    def add_function_mapping(self, name: str, addr1: int, addr2: int,
                            bank1: int = 0, bank2: int = 0):
        """Add a mapping between equivalent functions."""
        self.function_mappings.append(FunctionMapping(
            name=name, addr1=addr1, addr2=addr2, bank1=bank1, bank2=bank2
        ))

    def set_monitored_ranges(self, ranges: List[Tuple[int, int, int, int]]):
        """Set memory ranges to monitor for writes."""
        self.mem1.monitored_ranges = ranges
        self.mem2.monitored_ranges = ranges

    def run_until_event_count(self, max_events: int = 100,
                              max_instructions: int = 100000) -> Tuple[List[BehaviorEvent], List[BehaviorEvent]]:
        """
        Run both ROMs until we have enough behavioral events to compare.

        Returns:
            Tuple of (events1, events2)
        """
        # Run ROM1
        self.mem1.write_log = []
        self.mem1.instruction_num = 0
        try:
            for _ in range(max_instructions):
                if self.cpu1.stopped or self.cpu1.waiting:
                    break
                if len(self.mem1.write_log) >= max_events:
                    break
                self.cpu1.step()
                self.mem1.instruction_num += 1
        except (StopExecution, WaitForInterrupt):
            pass

        events1 = self.mem1.write_log.copy()

        # Run ROM2
        self.mem2.write_log = []
        self.mem2.instruction_num = 0
        try:
            for _ in range(max_instructions):
                if self.cpu2.stopped or self.cpu2.waiting:
                    break
                if len(self.mem2.write_log) >= max_events:
                    break
                self.cpu2.step()
                self.mem2.instruction_num += 1
        except (StopExecution, WaitForInterrupt):
            pass

        events2 = self.mem2.write_log.copy()

        return events1, events2

    def compare_hw_writes(self, events1: List[BehaviorEvent],
                          events2: List[BehaviorEvent]) -> List[str]:
        """
        Compare hardware register writes between two event logs.

        Returns list of differences.
        """
        # Extract HW writes
        hw1 = [(e.address, e.value) for e in events1 if e.event_type == EventType.HW_WRITE]
        hw2 = [(e.address, e.value) for e in events2 if e.event_type == EventType.HW_WRITE]

        differences = []

        # Compare in order
        min_len = min(len(hw1), len(hw2))
        for i in range(min_len):
            addr1, val1 = hw1[i]
            addr2, val2 = hw2[i]
            if addr1 != addr2:
                differences.append(f"HW write #{i}: {self.rom1_name} writes ${addr1:04X}, {self.rom2_name} writes ${addr2:04X}")
            elif val1 != val2:
                differences.append(f"HW write #{i} to ${addr1:04X}: {self.rom1_name}=${val1:02X}, {self.rom2_name}=${val2:02X}")

        if len(hw1) != len(hw2):
            differences.append(f"HW write count: {self.rom1_name}={len(hw1)}, {self.rom2_name}={len(hw2)}")

        return differences

    def compare_memory_state(self) -> Dict[int, Tuple[int, int]]:
        """
        Compare WRAM state between both CPUs.

        Returns dict of {address: (value1, value2)} for differing addresses.
        """
        differences = {}

        # Compare low WRAM ($0000-$1FFF)
        for addr in range(0x2000):
            v1 = self.mem1.wram[addr]
            v2 = self.mem2.wram[addr]
            if v1 != v2:
                differences[addr] = (v1, v2)

        return differences

    def format_hw_comparison(self, events1: List[BehaviorEvent],
                             events2: List[BehaviorEvent],
                             output: TextIO = None):
        """Format HW register write comparison."""
        if output is None:
            output = sys.stdout

        hw1 = [e for e in events1 if e.event_type == EventType.HW_WRITE]
        hw2 = [e for e in events2 if e.event_type == EventType.HW_WRITE]

        output.write("\n")
        output.write("=" * 70 + "\n")
        output.write("Hardware Register Write Comparison\n")
        output.write("=" * 70 + "\n\n")

        output.write(f"{'#':>4}  {'Address':>8}  {self.rom1_name:>10}  {self.rom2_name:>10}  {'Match':>6}\n")
        output.write("-" * 50 + "\n")

        max_len = max(len(hw1), len(hw2))
        for i in range(min(max_len, 100)):  # Show first 100
            addr1 = hw1[i].address if i < len(hw1) else None
            val1 = hw1[i].value if i < len(hw1) else None
            addr2 = hw2[i].address if i < len(hw2) else None
            val2 = hw2[i].value if i < len(hw2) else None

            # Check match
            if addr1 == addr2 and val1 == val2:
                match = "OK"
            elif addr1 == addr2:
                match = "VALUE"
            else:
                match = "ADDR"

            addr_str = f"${addr1:04X}" if addr1 is not None else "----"
            if addr2 is not None and addr2 != addr1:
                addr_str = f"${addr1:04X}/${addr2:04X}" if addr1 else f"----/${addr2:04X}"

            val1_str = f"${val1:02X}" if val1 is not None else "--"
            val2_str = f"${val2:02X}" if val2 is not None else "--"

            output.write(f"{i:>4}  {addr_str:>8}  {val1_str:>10}  {val2_str:>10}  {match:>6}\n")

        if max_len > 100:
            output.write(f"... and {max_len - 100} more writes\n")

        output.write("\n")
        output.write(f"Total HW writes: {self.rom1_name}={len(hw1)}, {self.rom2_name}={len(hw2)}\n")


def load_symbols(path: str) -> Dict[str, Tuple[int, int]]:
    """
    Load symbols from WLA-DX .sym file.

    Returns:
        Dict mapping symbol name to (bank, address)
    """
    symbols = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                # Format: "BB:ADDR name"
                parts = line.split()
                if len(parts) >= 2 and ':' in parts[0]:
                    bank_addr = parts[0].split(':')
                    try:
                        bank = int(bank_addr[0], 16)
                        addr = int(bank_addr[1], 16)
                        name = parts[1]
                        symbols[name] = (bank, addr)
                    except ValueError:
                        continue
    except IOError:
        pass
    return symbols


def load_rom_with_header_detection(path: str) -> bytes:
    """Load ROM file, auto-detecting and stripping SMC header if present."""
    from pathlib import Path
    data = Path(path).read_bytes()
    if len(data) % 1024 == 512:
        return data[512:]
    return data
