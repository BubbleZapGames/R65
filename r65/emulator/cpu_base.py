"""
65816 CPU Emulator Base Class.

Abstract base class defining the CPU interface.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Callable
from dataclasses import dataclass

if TYPE_CHECKING:
    from .memory import Memory


# Status register bit positions
FLAG_C = 0  # Carry
FLAG_Z = 1  # Zero
FLAG_I = 2  # IRQ Disable
FLAG_D = 3  # Decimal Mode
FLAG_X = 4  # Index Register Size (0=16-bit, 1=8-bit)
FLAG_M = 5  # Accumulator Size (0=16-bit, 1=8-bit)
FLAG_V = 6  # Overflow
FLAG_N = 7  # Negative


@dataclass
class CPUState:
    """Snapshot of CPU state for debugging."""
    A: int
    X: int
    Y: int
    SP: int
    PC: int
    PBR: int
    DBR: int
    D: int
    P: int
    cycles: int
    emulation_mode: bool


class StopExecution(Exception):
    """Raised when CPU should stop (STP instruction or error)."""
    pass


class WaitForInterrupt(Exception):
    """Raised when CPU executes WAI instruction."""
    pass


class CPU65816Base(ABC):
    """
    65816 CPU emulator base class with full instruction set support.

    Registers:
        A:   16-bit accumulator (use low 8 bits when M=1)
        X:   16-bit X index register (use low 8 bits when X=1)
        Y:   16-bit Y index register (use low 8 bits when X=1)
        SP:  16-bit stack pointer
        PC:  16-bit program counter
        PBR: 8-bit program bank register
        DBR: 8-bit data bank register
        D:   16-bit direct page register
        P:   8-bit processor status (NVMXDIZC)
    """

    def __init__(self, memory: 'Memory'):
        self.memory = memory

        # Registers
        self.A: int = 0       # Accumulator (16-bit)
        self.X: int = 0       # X index (16-bit)
        self.Y: int = 0       # Y index (16-bit)
        self.SP: int = 0x01FF # Stack pointer
        self.PC: int = 0      # Program counter
        self.PBR: int = 0     # Program bank register
        self.DBR: int = 0     # Data bank register
        self.D: int = 0       # Direct page register
        self.P: int = 0x34    # Status: M=1, X=1, I=1 (8-bit mode, IRQ disabled)

        # Emulation mode flag (separate from P register)
        self.emulation_mode: bool = True

        # Cycle counter
        self.cycles: int = 0

        # Stopped/waiting state
        self.stopped: bool = False
        self.waiting: bool = False

    def reset(self):
        """Reset the CPU to initial state."""
        self.A = 0
        self.X = 0
        self.Y = 0
        self.SP = 0x01FF
        self.D = 0
        self.DBR = 0
        self.PBR = 0
        self.P = 0x34  # M=1, X=1, I=1
        self.emulation_mode = True
        self.stopped = False
        self.waiting = False
        self.cycles = 0

        # Load reset vector
        self.PC = self.memory.get_reset_vector()

    def get_state(self) -> CPUState:
        """Get current CPU state snapshot."""
        return CPUState(
            A=self.A,
            X=self.X,
            Y=self.Y,
            SP=self.SP,
            PC=self.PC,
            PBR=self.PBR,
            DBR=self.DBR,
            D=self.D,
            P=self.P,
            cycles=self.cycles,
            emulation_mode=self.emulation_mode
        )

    # Status flag properties
    @property
    def flag_c(self) -> bool:
        """Carry flag."""
        return bool(self.P & (1 << FLAG_C))

    @flag_c.setter
    def flag_c(self, value: bool):
        if value:
            self.P |= (1 << FLAG_C)
        else:
            self.P &= ~(1 << FLAG_C)

    @property
    def flag_z(self) -> bool:
        """Zero flag."""
        return bool(self.P & (1 << FLAG_Z))

    @flag_z.setter
    def flag_z(self, value: bool):
        if value:
            self.P |= (1 << FLAG_Z)
        else:
            self.P &= ~(1 << FLAG_Z)

    @property
    def flag_i(self) -> bool:
        """IRQ disable flag."""
        return bool(self.P & (1 << FLAG_I))

    @flag_i.setter
    def flag_i(self, value: bool):
        if value:
            self.P |= (1 << FLAG_I)
        else:
            self.P &= ~(1 << FLAG_I)

    @property
    def flag_d(self) -> bool:
        """Decimal mode flag."""
        return bool(self.P & (1 << FLAG_D))

    @flag_d.setter
    def flag_d(self, value: bool):
        if value:
            self.P |= (1 << FLAG_D)
        else:
            self.P &= ~(1 << FLAG_D)

    @property
    def flag_x(self) -> bool:
        """Index register size (True = 8-bit, False = 16-bit)."""
        return bool(self.P & (1 << FLAG_X))

    @flag_x.setter
    def flag_x(self, value: bool):
        if value:
            self.P |= (1 << FLAG_X)
            # Truncate X and Y to 8 bits when switching to 8-bit mode
            self.X &= 0xFF
            self.Y &= 0xFF
        else:
            self.P &= ~(1 << FLAG_X)

    @property
    def flag_m(self) -> bool:
        """Accumulator size (True = 8-bit, False = 16-bit)."""
        return bool(self.P & (1 << FLAG_M))

    @flag_m.setter
    def flag_m(self, value: bool):
        if value:
            self.P |= (1 << FLAG_M)
        else:
            self.P &= ~(1 << FLAG_M)

    @property
    def flag_v(self) -> bool:
        """Overflow flag."""
        return bool(self.P & (1 << FLAG_V))

    @flag_v.setter
    def flag_v(self, value: bool):
        if value:
            self.P |= (1 << FLAG_V)
        else:
            self.P &= ~(1 << FLAG_V)

    @property
    def flag_n(self) -> bool:
        """Negative flag."""
        return bool(self.P & (1 << FLAG_N))

    @flag_n.setter
    def flag_n(self, value: bool):
        if value:
            self.P |= (1 << FLAG_N)
        else:
            self.P &= ~(1 << FLAG_N)

    # Register size helpers
    @property
    def acc_size(self) -> int:
        """Accumulator size in bytes (1 or 2)."""
        return 1 if self.flag_m else 2

    @property
    def idx_size(self) -> int:
        """Index register size in bytes (1 or 2)."""
        return 1 if self.flag_x else 2

    @property
    def acc_mask(self) -> int:
        """Mask for accumulator value."""
        return 0xFF if self.flag_m else 0xFFFF

    @property
    def idx_mask(self) -> int:
        """Mask for index register value."""
        return 0xFF if self.flag_x else 0xFFFF

    # Memory access helpers
    def fetch_byte(self) -> int:
        """Fetch byte at PC and increment PC."""
        value = self.memory.read(self.PBR, self.PC)
        self.PC = (self.PC + 1) & 0xFFFF
        return value

    def fetch_word(self) -> int:
        """Fetch 16-bit word at PC and increment PC by 2."""
        lo = self.fetch_byte()
        hi = self.fetch_byte()
        return lo | (hi << 8)

    def fetch_long(self) -> int:
        """Fetch 24-bit long at PC and increment PC by 3."""
        lo = self.fetch_byte()
        mid = self.fetch_byte()
        hi = self.fetch_byte()
        return lo | (mid << 8) | (hi << 16)

    def push_byte(self, value: int):
        """Push byte onto stack."""
        self.memory.write(0, self.SP, value & 0xFF)
        if self.emulation_mode:
            # In emulation mode, stack wraps within page 1
            self.SP = 0x0100 | ((self.SP - 1) & 0xFF)
        else:
            self.SP = (self.SP - 1) & 0xFFFF

    def push_word(self, value: int):
        """Push 16-bit word onto stack (high byte first)."""
        self.push_byte((value >> 8) & 0xFF)
        self.push_byte(value & 0xFF)

    def pull_byte(self) -> int:
        """Pull byte from stack."""
        if self.emulation_mode:
            self.SP = 0x0100 | ((self.SP + 1) & 0xFF)
        else:
            self.SP = (self.SP + 1) & 0xFFFF
        return self.memory.read(0, self.SP)

    def pull_word(self) -> int:
        """Pull 16-bit word from stack (low byte first)."""
        lo = self.pull_byte()
        hi = self.pull_byte()
        return lo | (hi << 8)

    def set_nz_flags(self, value: int, is_16bit: bool = False):
        """Set N and Z flags based on value."""
        if is_16bit:
            self.flag_z = (value & 0xFFFF) == 0
            self.flag_n = bool(value & 0x8000)
        else:
            self.flag_z = (value & 0xFF) == 0
            self.flag_n = bool(value & 0x80)

    @abstractmethod
    def step(self) -> int:
        """
        Execute one instruction.

        Returns:
            Number of cycles used.
        """
        pass

    def run(self, max_cycles: Optional[int] = None,
            trace_callback: Optional[Callable[['CPU65816Base'], None]] = None) -> int:
        """
        Run until stopped or max cycles reached.

        Args:
            max_cycles: Maximum cycles to execute (None = unlimited)
            trace_callback: Called before each instruction for tracing

        Returns:
            Total cycles executed.
        """
        start_cycles = self.cycles

        try:
            while max_cycles is None or self.cycles - start_cycles < max_cycles:
                if trace_callback:
                    trace_callback(self)
                self.step()
        except (StopExecution, WaitForInterrupt):
            pass

        return self.cycles - start_cycles

    def trigger_nmi(self):
        """Trigger NMI interrupt."""
        self.waiting = False

        if self.emulation_mode:
            self.push_word(self.PC)
            self.push_byte(self.P | 0x20)  # B flag set for software interrupt
        else:
            self.push_byte(self.PBR)
            self.push_word(self.PC)
            self.push_byte(self.P)

        self.flag_i = True
        self.flag_d = False
        self.PBR = 0
        self.PC = self.memory.get_nmi_vector()

    def trigger_irq(self):
        """Trigger IRQ interrupt (if enabled)."""
        if self.flag_i:
            return  # IRQ disabled

        self.waiting = False

        if self.emulation_mode:
            self.push_word(self.PC)
            self.push_byte(self.P & ~0x10)  # B flag clear for hardware interrupt
        else:
            self.push_byte(self.PBR)
            self.push_word(self.PC)
            self.push_byte(self.P)

        self.flag_i = True
        self.flag_d = False
        self.PBR = 0
        self.PC = self.memory.get_irq_vector()
