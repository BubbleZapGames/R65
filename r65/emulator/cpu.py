# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
65816 CPU Emulator.

Full instruction set implementation with generic handlers and data-driven dispatch.
"""

from typing import TYPE_CHECKING, Callable, List, Optional
from dataclasses import dataclass
from . import addressing as addr
from . import operations as ops

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

# SNES timing constants (NTSC)
MASTER_CLOCK = 21477272  # Hz
SCANLINES_PER_FRAME_NTSC = 262
SCANLINES_PER_FRAME_PAL = 312
VBLANK_START_SCANLINE = 225  # NMI triggers at start of scanline 225
CYCLES_PER_SCANLINE = 186  # Average CPU cycles per scanline (~1364 master / 6-8)


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


class CPU65816:
    """
    65816 CPU emulator with full instruction set support.

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

    # Interrupt vector addresses
    RESET_VECTOR = 0xFFFC
    NMI_VECTOR = 0xFFEA       # Native mode
    NMI_VECTOR_EMU = 0xFFFA   # Emulation mode
    IRQ_VECTOR = 0xFFEE       # Native mode
    IRQ_VECTOR_EMU = 0xFFFE   # Emulation mode

    def __init__(self, memory: 'Memory'):
        self.memory = memory
        # Set back-reference for RDNMI reads
        if hasattr(memory, 'set_cpu'):
            memory.set_cpu(self)

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

        # Timing state for vblank NMI
        self.scanline: int = 0  # Current scanline (0-261 NTSC)
        self.scanline_cycles: int = 0  # Cycles accumulated in current scanline
        self.frame: int = 0  # Frame counter
        self.nmi_enabled: bool = False  # NMITIMEN bit 7 ($4200)
        self.vblank_flag: bool = False  # RDNMI bit 7 ($4210), set at vblank
        self.nmi_pending: bool = False  # NMI waiting to be triggered
        self.auto_nmi: bool = False  # Enable automatic NMI timing
        self.scanlines_per_frame: int = SCANLINES_PER_FRAME_NTSC

        # Build instruction dispatch table
        self._instructions: List[Callable[[], int]] = self._build_instruction_table()

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

        # Reset timing state
        self.scanline = 0
        self.scanline_cycles = 0
        self.frame = 0
        self.vblank_flag = False
        self.nmi_pending = False

        # Load reset vector
        self.PC = self.memory.read16(self.RESET_VECTOR)

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

    # ==================== STATUS FLAG PROPERTIES ====================

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

    # ==================== REGISTER SIZE HELPERS ====================

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

    # ==================== MEMORY ACCESS HELPERS ====================

    def fetch_byte(self) -> int:
        """Fetch byte at PC and increment PC."""
        value = self.memory.read((self.PBR << 16) | self.PC)
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
        self.memory.write(self.SP, value & 0xFF)
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
        return self.memory.read(self.SP)

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

    # ==================== EXECUTION ====================

    def step(self) -> int:
        """Execute one instruction and return cycles consumed."""
        if self.stopped:
            raise StopExecution("CPU stopped")
        if self.waiting:
            raise WaitForInterrupt("CPU waiting for interrupt")

        opcode = self.fetch_byte()
        handler = self._instructions[opcode]
        cycles = handler()
        self.cycles += cycles

        # Update timing and check for vblank NMI
        if self.auto_nmi:
            self._update_timing(cycles)

        return cycles

    def _update_timing(self, cycles: int):
        """Update scanline timing and trigger NMI at vblank."""
        self.scanline_cycles += cycles

        # Check for scanline completion
        while self.scanline_cycles >= CYCLES_PER_SCANLINE:
            self.scanline_cycles -= CYCLES_PER_SCANLINE
            self.scanline += 1

            # Check for vblank start
            if self.scanline == VBLANK_START_SCANLINE:
                self.vblank_flag = True
                if self.nmi_enabled:
                    self.nmi_pending = True

            # Check for frame completion
            if self.scanline >= self.scanlines_per_frame:
                self.scanline = 0
                self.vblank_flag = False
                self.frame += 1

        # Trigger pending NMI
        if self.nmi_pending:
            self.nmi_pending = False
            self.trigger_nmi()

    def run(self, max_cycles: Optional[int] = None,
            trace_callback: Optional[Callable[['CPU65816'], None]] = None) -> int:
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

    # ==================== INTERRUPTS ====================

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
        self.PC = self.memory.read16(self.NMI_VECTOR)

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
        self.PC = self.memory.read16(self.IRQ_VECTOR)

    # ==================== TIMING CONFIGURATION ====================

    def enable_auto_nmi(self, enabled: bool = True):
        """
        Enable automatic vblank NMI timing.

        When enabled, the CPU tracks scanlines and triggers NMI
        at the start of vblank (scanline 225) if nmi_enabled is True.
        """
        self.auto_nmi = enabled

    def set_nmi_enabled(self, enabled: bool):
        """
        Enable/disable NMI (simulates NMITIMEN $4200 bit 7).

        NMI will only trigger at vblank if this is True.
        """
        self.nmi_enabled = enabled

    def set_region(self, pal: bool = False):
        """
        Set video region timing (NTSC or PAL).

        Args:
            pal: True for PAL (312 scanlines), False for NTSC (262 scanlines)
        """
        self.scanlines_per_frame = SCANLINES_PER_FRAME_PAL if pal else SCANLINES_PER_FRAME_NTSC

    def read_nmi_flag(self) -> bool:
        """
        Read and clear vblank flag (simulates RDNMI $4210 bit 7).

        Returns True if in vblank, then clears the flag.
        """
        flag = self.vblank_flag
        self.vblank_flag = False
        return flag

    @property
    def in_vblank(self) -> bool:
        """Check if currently in vblank period (scanlines 225-261)."""
        return self.scanline >= VBLANK_START_SCANLINE

    # ==================== GENERIC HANDLERS ====================

    def _load_imm(self, op8, op16, use_m: bool) -> int:
        """Generic immediate load (LDA/LDX/LDY)."""
        flag = self.flag_m if use_m else self.flag_x
        if use_m:
            value, _ = addr.immediate_acc(self)
        else:
            value, _ = addr.immediate_idx(self)
        if flag:
            op8(self, value)
            return 2
        else:
            op16(self, value)
            return 3

    def _load(self, addr_fn, op8, op16, cycles8: int, use_m: bool = True) -> int:
        """Generic memory load (LDA/LDX/LDY with addressing mode)."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        flag = self.flag_m if use_m else self.flag_x
        if flag:
            op8(self, self.memory.read(ea))
            return cycles8 + extra
        else:
            op16(self, self.memory.read16(ea))
            return cycles8 + 1 + extra

    def _store(self, addr_fn, reg_fn, cycles8: int, use_m: bool = True, no_extra: bool = False) -> int:
        """Generic memory store (STA/STX/STY)."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if no_extra:
            extra = 0
        flag = self.flag_m if use_m else self.flag_x
        value = reg_fn()
        if flag:
            self.memory.write(ea, value & 0xFF)
            return cycles8 + extra
        else:
            self.memory.write16(ea, value & 0xFFFF)
            return cycles8 + 1 + extra

    def _store_zero(self, addr_fn, cycles8: int, no_extra: bool = False) -> int:
        """Generic STZ instruction."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if no_extra:
            extra = 0
        if self.flag_m:
            self.memory.write(ea, 0)
            return cycles8 + extra
        else:
            self.memory.write16(ea, 0)
            return cycles8 + 1 + extra

    def _alu_imm(self, op8, op16) -> int:
        """Generic immediate ALU (ADC/SBC/AND/ORA/EOR/CMP)."""
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            op8(self, value)
            return 2
        else:
            op16(self, value)
            return 3

    def _alu(self, addr_fn, op8, op16, cycles8: int) -> int:
        """Generic memory ALU (ADC/SBC/AND/ORA/EOR/CMP with addressing mode)."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if self.flag_m:
            op8(self, self.memory.read(ea))
            return cycles8 + extra
        else:
            op16(self, self.memory.read16(ea))
            return cycles8 + 1 + extra

    def _cmp_idx_imm(self, reg_fn, op8, op16) -> int:
        """Generic immediate index compare (CPX/CPY)."""
        value, _ = addr.immediate_idx(self)
        if self.flag_x:
            op8(self, reg_fn() & 0xFF, value)
            return 2
        else:
            op16(self, reg_fn(), value)
            return 3

    def _cmp_idx(self, addr_fn, reg_fn, op8, op16, cycles8: int) -> int:
        """Generic memory index compare (CPX/CPY with addressing mode)."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if self.flag_x:
            op8(self, reg_fn() & 0xFF, self.memory.read(ea))
            return cycles8 + extra
        else:
            op16(self, reg_fn(), self.memory.read16(ea))
            return cycles8 + 1 + extra

    def _bit_imm(self) -> int:
        """BIT immediate (doesn't set N/V from value)."""
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.bit8(self, value, set_nv=False)
            return 2
        else:
            ops.bit16(self, value, set_nv=False)
            return 3

    def _bit(self, addr_fn, cycles8: int) -> int:
        """Generic BIT with addressing mode (sets N/V from value)."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if self.flag_m:
            ops.bit8(self, self.memory.read(ea), set_nv=True)
            return cycles8 + extra
        else:
            ops.bit16(self, self.memory.read16(ea), set_nv=True)
            return cycles8 + 1 + extra

    def _rmw_acc(self, op8, op16) -> int:
        """Generic accumulator RMW (ASL/LSR/ROL/ROR/INC/DEC A)."""
        if self.flag_m:
            result = op8(self, self.A & 0xFF)
            self.A = (self.A & 0xFF00) | result
            return 2
        else:
            self.A = op16(self, self.A)
            return 2

    def _rmw(self, addr_fn, op8, op16, cycles8: int) -> int:
        """Generic memory RMW (ASL/LSR/ROL/ROR/INC/DEC with addressing mode)."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if self.flag_m:
            value = self.memory.read(ea)
            result = op8(self, value)
            self.memory.write(ea, result)
            return cycles8 + extra
        else:
            value = self.memory.read16(ea)
            result = op16(self, value)
            self.memory.write16(ea, result)
            return cycles8 + 2 + extra

    def _tsb(self, addr_fn, cycles8: int) -> int:
        """Test and Set Bits."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if self.flag_m:
            value = self.memory.read(ea)
            self.flag_z = (self.A & value) == 0
            self.memory.write(ea, value | (self.A & 0xFF))
            return cycles8 + extra
        else:
            value = self.memory.read16(ea)
            self.flag_z = (self.A & value) == 0
            self.memory.write16(ea, value | self.A)
            return cycles8 + 2 + extra

    def _trb(self, addr_fn, cycles8: int) -> int:
        """Test and Reset Bits."""
        bank, address, extra = addr_fn(self)
        ea = (bank << 16) | address
        if self.flag_m:
            value = self.memory.read(ea)
            self.flag_z = (self.A & value) == 0
            self.memory.write(ea, value & ~(self.A & 0xFF))
            return cycles8 + extra
        else:
            value = self.memory.read16(ea)
            self.flag_z = (self.A & value) == 0
            self.memory.write16(ea, value & ~self.A)
            return cycles8 + 2 + extra

    def _branch(self, condition: bool) -> int:
        """Execute conditional branch."""
        target, extra = addr.relative_8(self)
        if condition:
            self.PC = target
            return 3 + extra
        return 2

    # ==================== TRANSFER INSTRUCTIONS ====================

    def _tax(self) -> int:
        if self.flag_x:
            self.X = self.A & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = self.A & 0xFFFF
            self.set_nz_flags(self.X, True)
        return 2

    def _tay(self) -> int:
        if self.flag_x:
            self.Y = self.A & 0xFF
            self.set_nz_flags(self.Y, False)
        else:
            self.Y = self.A & 0xFFFF
            self.set_nz_flags(self.Y, True)
        return 2

    def _txa(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | (self.X & 0xFF)
            self.set_nz_flags(self.A & 0xFF, False)
        else:
            self.A = self.X & 0xFFFF
            self.set_nz_flags(self.A, True)
        return 2

    def _tya(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | (self.Y & 0xFF)
            self.set_nz_flags(self.A & 0xFF, False)
        else:
            self.A = self.Y & 0xFFFF
            self.set_nz_flags(self.A, True)
        return 2

    def _tsx(self) -> int:
        if self.flag_x:
            self.X = self.SP & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = self.SP
            self.set_nz_flags(self.X, True)
        return 2

    def _txy(self) -> int:
        if self.flag_x:
            self.Y = self.X & 0xFF
            self.set_nz_flags(self.Y, False)
        else:
            self.Y = self.X
            self.set_nz_flags(self.Y, True)
        return 2

    def _tyx(self) -> int:
        if self.flag_x:
            self.X = self.Y & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = self.Y
            self.set_nz_flags(self.X, True)
        return 2

    def _txs(self) -> int:
        if self.emulation_mode:
            self.SP = 0x0100 | (self.X & 0xFF)
        else:
            self.SP = self.X & 0xFFFF
        return 2

    def _tcd(self) -> int:
        self.D = self.A & 0xFFFF
        self.set_nz_flags(self.D, True)
        return 2

    def _tdc(self) -> int:
        self.A = self.D
        self.set_nz_flags(self.A, True)
        return 2

    def _tcs(self) -> int:
        if self.emulation_mode:
            self.SP = 0x0100 | (self.A & 0xFF)
        else:
            self.SP = self.A & 0xFFFF
        return 2

    def _tsc(self) -> int:
        self.A = self.SP
        self.set_nz_flags(self.A, True)
        return 2

    # ==================== STACK INSTRUCTIONS ====================

    def _pha(self) -> int:
        if self.flag_m:
            self.push_byte(self.A & 0xFF)
            return 3
        else:
            self.push_word(self.A)
            return 4

    def _phx(self) -> int:
        if self.flag_x:
            self.push_byte(self.X & 0xFF)
            return 3
        else:
            self.push_word(self.X)
            return 4

    def _phy(self) -> int:
        if self.flag_x:
            self.push_byte(self.Y & 0xFF)
            return 3
        else:
            self.push_word(self.Y)
            return 4

    def _pla(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | self.pull_byte()
            self.set_nz_flags(self.A & 0xFF, False)
            return 4
        else:
            self.A = self.pull_word()
            self.set_nz_flags(self.A, True)
            return 5

    def _plx(self) -> int:
        if self.flag_x:
            self.X = self.pull_byte()
            self.set_nz_flags(self.X, False)
            return 4
        else:
            self.X = self.pull_word()
            self.set_nz_flags(self.X, True)
            return 5

    def _ply(self) -> int:
        if self.flag_x:
            self.Y = self.pull_byte()
            self.set_nz_flags(self.Y, False)
            return 4
        else:
            self.Y = self.pull_word()
            self.set_nz_flags(self.Y, True)
            return 5

    def _php(self) -> int:
        self.push_byte(self.P)
        return 3

    def _plp(self) -> int:
        self.P = self.pull_byte()
        if self.emulation_mode:
            self.flag_m = True
            self.flag_x = True
        return 4

    def _phb(self) -> int:
        self.push_byte(self.DBR)
        return 3

    def _plb(self) -> int:
        self.DBR = self.pull_byte()
        self.set_nz_flags(self.DBR, False)
        return 4

    def _phd(self) -> int:
        self.push_word(self.D)
        return 4

    def _pld(self) -> int:
        self.D = self.pull_word()
        self.set_nz_flags(self.D, True)
        return 5

    def _phk(self) -> int:
        self.push_byte(self.PBR)
        return 3

    def _pea(self) -> int:
        address = self.fetch_word()
        self.push_word(address)
        return 5

    def _pei(self) -> int:
        bank, address, extra = addr.direct(self)
        ea = (bank << 16) | address
        value = self.memory.read16(ea)
        self.push_word(value)
        return 6 + extra

    def _per(self) -> int:
        offset = self.fetch_word()
        if offset & 0x8000:
            offset = offset - 65536
        address = (self.PC + offset) & 0xFFFF
        self.push_word(address)
        return 6

    # ==================== INCREMENT/DECREMENT INDEX ====================

    def _inx(self) -> int:
        if self.flag_x:
            self.X = (self.X + 1) & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = (self.X + 1) & 0xFFFF
            self.set_nz_flags(self.X, True)
        return 2

    def _iny(self) -> int:
        if self.flag_x:
            self.Y = (self.Y + 1) & 0xFF
            self.set_nz_flags(self.Y, False)
        else:
            self.Y = (self.Y + 1) & 0xFFFF
            self.set_nz_flags(self.Y, True)
        return 2

    def _dex(self) -> int:
        if self.flag_x:
            self.X = (self.X - 1) & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = (self.X - 1) & 0xFFFF
            self.set_nz_flags(self.X, True)
        return 2

    def _dey(self) -> int:
        if self.flag_x:
            self.Y = (self.Y - 1) & 0xFF
            self.set_nz_flags(self.Y, False)
        else:
            self.Y = (self.Y - 1) & 0xFFFF
            self.set_nz_flags(self.Y, True)
        return 2

    # ==================== BRANCH INSTRUCTIONS ====================

    def _bra(self) -> int:
        target, extra = addr.relative_8(self)
        self.PC = target
        return 3 + extra

    def _brl(self) -> int:
        target, _ = addr.relative_16(self)
        self.PC = target
        return 4

    def _beq(self) -> int:
        return self._branch(self.flag_z)

    def _bne(self) -> int:
        return self._branch(not self.flag_z)

    def _bcs(self) -> int:
        return self._branch(self.flag_c)

    def _bcc(self) -> int:
        return self._branch(not self.flag_c)

    def _bmi(self) -> int:
        return self._branch(self.flag_n)

    def _bpl(self) -> int:
        return self._branch(not self.flag_n)

    def _bvs(self) -> int:
        return self._branch(self.flag_v)

    def _bvc(self) -> int:
        return self._branch(not self.flag_v)

    # ==================== JUMP INSTRUCTIONS ====================

    def _jmp_abs(self) -> int:
        self.PC = self.fetch_word()
        return 3

    def _jmp_long(self) -> int:
        address = self.fetch_word()
        bank = self.fetch_byte()
        self.PBR = bank
        self.PC = address
        return 4

    def _jmp_ind(self) -> int:
        _, address, _ = addr.absolute_indirect(self)
        self.PC = address
        return 5

    def _jmp_ind_long(self) -> int:
        bank, address, _ = addr.absolute_indirect_long(self)
        self.PBR = bank
        self.PC = address
        return 6

    def _jmp_indexed_ind(self) -> int:
        _, address, _ = addr.absolute_indexed_indirect(self)
        self.PC = address
        return 6

    def _jsr_abs(self) -> int:
        address = self.fetch_word()
        self.push_word(self.PC - 1)
        self.PC = address
        return 6

    def _jsr_long(self) -> int:
        address = self.fetch_word()
        bank = self.fetch_byte()
        self.push_byte(self.PBR)
        self.push_word(self.PC - 1)
        self.PBR = bank
        self.PC = address
        return 8

    def _jsr_indexed_ind(self) -> int:
        _, address, _ = addr.absolute_indexed_indirect(self)
        self.push_word(self.PC - 1)
        self.PC = address
        return 8

    def _rts(self) -> int:
        self.PC = (self.pull_word() + 1) & 0xFFFF
        return 6

    def _rtl(self) -> int:
        self.PC = (self.pull_word() + 1) & 0xFFFF
        self.PBR = self.pull_byte()
        return 6

    def _rti(self) -> int:
        self.P = self.pull_byte()
        self.PC = self.pull_word()
        if not self.emulation_mode:
            self.PBR = self.pull_byte()
            return 7
        return 6

    # ==================== FLAG INSTRUCTIONS ====================

    def _clc(self) -> int:
        self.flag_c = False
        return 2

    def _sec(self) -> int:
        self.flag_c = True
        return 2

    def _cli(self) -> int:
        self.flag_i = False
        return 2

    def _sei(self) -> int:
        self.flag_i = True
        return 2

    def _cld(self) -> int:
        self.flag_d = False
        return 2

    def _sed(self) -> int:
        self.flag_d = True
        return 2

    def _clv(self) -> int:
        self.flag_v = False
        return 2

    def _sep(self) -> int:
        bits = self.fetch_byte()
        self.P |= bits
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 3

    def _rep(self) -> int:
        bits = self.fetch_byte()
        self.P &= ~bits
        return 3

    def _xce(self) -> int:
        old_c = self.flag_c
        self.flag_c = self.emulation_mode
        self.emulation_mode = old_c
        if self.emulation_mode:
            self.flag_m = True
            self.flag_x = True
            self.X &= 0xFF
            self.Y &= 0xFF
            self.SP = 0x0100 | (self.SP & 0xFF)
        return 2

    # ==================== MISCELLANEOUS ====================

    def _nop(self) -> int:
        return 2

    def _wdm(self) -> int:
        self.fetch_byte()  # Skip signature byte
        return 2

    def _xba(self) -> int:
        lo = self.A & 0xFF
        hi = (self.A >> 8) & 0xFF
        self.A = (lo << 8) | hi
        self.set_nz_flags(self.A & 0xFF, False)
        return 3

    def _stp(self) -> int:
        self.stopped = True
        return 3

    def _wai(self) -> int:
        self.waiting = True
        return 3

    def _brk(self) -> int:
        self.fetch_byte()  # Skip signature byte
        if self.emulation_mode:
            self.push_word(self.PC)
            self.push_byte(self.P | 0x10)
            self.flag_i = True
            self.flag_d = False
            self.PC = self.memory.read16(0xFFFE)
            return 7
        else:
            self.push_byte(self.PBR)
            self.push_word(self.PC)
            self.push_byte(self.P)
            self.flag_i = True
            self.flag_d = False
            self.PBR = 0
            self.PC = self.memory.read16(0xFFE6)
            return 8

    def _cop(self) -> int:
        self.fetch_byte()  # Skip signature byte
        if self.emulation_mode:
            self.push_word(self.PC)
            self.push_byte(self.P)
            self.flag_i = True
            self.flag_d = False
            self.PC = self.memory.read16(0xFFF4)
            return 7
        else:
            self.push_byte(self.PBR)
            self.push_word(self.PC)
            self.push_byte(self.P)
            self.flag_i = True
            self.flag_d = False
            self.PBR = 0
            self.PC = self.memory.read16(0xFFE4)
            return 8

    def _mvn(self) -> int:
        dest_bank, src_bank, _ = addr.block_move(self)
        self.DBR = dest_bank
        src = self.memory.read((src_bank << 16) | self.X)
        self.memory.write((dest_bank << 16) | self.Y, src)
        self.X = (self.X + 1) & self.idx_mask
        self.Y = (self.Y + 1) & self.idx_mask
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def _mvp(self) -> int:
        dest_bank, src_bank, _ = addr.block_move(self)
        self.DBR = dest_bank
        src = self.memory.read((src_bank << 16) | self.X)
        self.memory.write((dest_bank << 16) | self.Y, src)
        self.X = (self.X - 1) & self.idx_mask
        self.Y = (self.Y - 1) & self.idx_mask
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def _illegal(self) -> int:
        return 2

    # ==================== INSTRUCTION TABLE ====================

    def _build_instruction_table(self) -> List[Callable[[], int]]:
        """Build the opcode dispatch table mapping 0x00-0xFF to handlers."""
        t: List[Callable[[], int]] = [self._illegal] * 256

        # Helper references for operations
        lda8, lda16 = ops.lda8, ops.lda16
        ldx8, ldx16 = ops.ldx8, ops.ldx16
        ldy8, ldy16 = ops.ldy8, ops.ldy16
        adc8, adc16 = ops.adc8, ops.adc16
        sbc8, sbc16 = ops.sbc8, ops.sbc16
        and8, and16 = ops.and8, ops.and16
        ora8, ora16 = ops.ora8, ops.ora16
        eor8, eor16 = ops.eor8, ops.eor16
        cmp8, cmp16 = ops.cmp8, ops.cmp16
        asl8, asl16 = ops.asl8, ops.asl16
        lsr8, lsr16 = ops.lsr8, ops.lsr16
        rol8, rol16 = ops.rol8, ops.rol16
        ror8, ror16 = ops.ror8, ops.ror16
        inc8, inc16 = ops.inc8, ops.inc16
        dec8, dec16 = ops.dec8, ops.dec16

        # Register getters for store instructions
        get_a = lambda: self.A
        get_x = lambda: self.X
        get_y = lambda: self.Y

        # ===== LDA =====
        t[0xA9] = lambda: self._load_imm(lda8, lda16, True)
        t[0xA5] = lambda: self._load(addr.direct, lda8, lda16, 3)
        t[0xB5] = lambda: self._load(addr.direct_x, lda8, lda16, 4)
        t[0xAD] = lambda: self._load(addr.absolute, lda8, lda16, 4)
        t[0xBD] = lambda: self._load(addr.absolute_x, lda8, lda16, 4)
        t[0xB9] = lambda: self._load(addr.absolute_y, lda8, lda16, 4)
        t[0xAF] = lambda: self._load(addr.absolute_long, lda8, lda16, 5)
        t[0xBF] = lambda: self._load(addr.absolute_long_x, lda8, lda16, 5)
        t[0xA1] = lambda: self._load(addr.direct_x_indirect, lda8, lda16, 6)
        t[0xB1] = lambda: self._load(addr.direct_indirect_y, lda8, lda16, 5)
        t[0xB2] = lambda: self._load(addr.direct_indirect, lda8, lda16, 5)
        t[0xA7] = lambda: self._load(addr.direct_indirect_long, lda8, lda16, 6)
        t[0xB7] = lambda: self._load(addr.direct_indirect_long_y, lda8, lda16, 6)
        t[0xA3] = lambda: self._load(addr.stack_relative, lda8, lda16, 4)
        t[0xB3] = lambda: self._load(addr.stack_relative_indirect_y, lda8, lda16, 7)

        # ===== LDX =====
        t[0xA2] = lambda: self._load_imm(ldx8, ldx16, False)
        t[0xA6] = lambda: self._load(addr.direct, ldx8, ldx16, 3, False)
        t[0xB6] = lambda: self._load(addr.direct_y, ldx8, ldx16, 4, False)
        t[0xAE] = lambda: self._load(addr.absolute, ldx8, ldx16, 4, False)
        t[0xBE] = lambda: self._load(addr.absolute_y, ldx8, ldx16, 4, False)

        # ===== LDY =====
        t[0xA0] = lambda: self._load_imm(ldy8, ldy16, False)
        t[0xA4] = lambda: self._load(addr.direct, ldy8, ldy16, 3, False)
        t[0xB4] = lambda: self._load(addr.direct_x, ldy8, ldy16, 4, False)
        t[0xAC] = lambda: self._load(addr.absolute, ldy8, ldy16, 4, False)
        t[0xBC] = lambda: self._load(addr.absolute_x, ldy8, ldy16, 4, False)

        # ===== STA =====
        t[0x85] = lambda: self._store(addr.direct, get_a, 3)
        t[0x95] = lambda: self._store(addr.direct_x, get_a, 4)
        t[0x8D] = lambda: self._store(addr.absolute, get_a, 4, True, True)
        t[0x9D] = lambda: self._store(addr.absolute_x_no_penalty, get_a, 5, True, True)
        t[0x99] = lambda: self._store(addr.absolute_y, get_a, 5, True, True)
        t[0x8F] = lambda: self._store(addr.absolute_long, get_a, 5, True, True)
        t[0x9F] = lambda: self._store(addr.absolute_long_x, get_a, 5, True, True)
        t[0x81] = lambda: self._store(addr.direct_x_indirect, get_a, 6)
        t[0x91] = lambda: self._store(addr.direct_indirect_y, get_a, 6)
        t[0x92] = lambda: self._store(addr.direct_indirect, get_a, 5)
        t[0x87] = lambda: self._store(addr.direct_indirect_long, get_a, 6)
        t[0x97] = lambda: self._store(addr.direct_indirect_long_y, get_a, 6)
        t[0x83] = lambda: self._store(addr.stack_relative, get_a, 4, True, True)
        t[0x93] = lambda: self._store(addr.stack_relative_indirect_y, get_a, 7, True, True)

        # ===== STX =====
        t[0x86] = lambda: self._store(addr.direct, get_x, 3, False)
        t[0x96] = lambda: self._store(addr.direct_y, get_x, 4, False)
        t[0x8E] = lambda: self._store(addr.absolute, get_x, 4, False, True)

        # ===== STY =====
        t[0x84] = lambda: self._store(addr.direct, get_y, 3, False)
        t[0x94] = lambda: self._store(addr.direct_x, get_y, 4, False)
        t[0x8C] = lambda: self._store(addr.absolute, get_y, 4, False, True)

        # ===== STZ =====
        t[0x64] = lambda: self._store_zero(addr.direct, 3)
        t[0x74] = lambda: self._store_zero(addr.direct_x, 4)
        t[0x9C] = lambda: self._store_zero(addr.absolute, 4, True)
        t[0x9E] = lambda: self._store_zero(addr.absolute_x_no_penalty, 5, True)

        # ===== ADC =====
        t[0x69] = lambda: self._alu_imm(adc8, adc16)
        t[0x65] = lambda: self._alu(addr.direct, adc8, adc16, 3)
        t[0x75] = lambda: self._alu(addr.direct_x, adc8, adc16, 4)
        t[0x6D] = lambda: self._alu(addr.absolute, adc8, adc16, 4)
        t[0x7D] = lambda: self._alu(addr.absolute_x, adc8, adc16, 4)
        t[0x79] = lambda: self._alu(addr.absolute_y, adc8, adc16, 4)
        t[0x6F] = lambda: self._alu(addr.absolute_long, adc8, adc16, 5)
        t[0x7F] = lambda: self._alu(addr.absolute_long_x, adc8, adc16, 5)
        t[0x61] = lambda: self._alu(addr.direct_x_indirect, adc8, adc16, 6)
        t[0x71] = lambda: self._alu(addr.direct_indirect_y, adc8, adc16, 5)
        t[0x72] = lambda: self._alu(addr.direct_indirect, adc8, adc16, 5)
        t[0x67] = lambda: self._alu(addr.direct_indirect_long, adc8, adc16, 6)
        t[0x77] = lambda: self._alu(addr.direct_indirect_long_y, adc8, adc16, 6)
        t[0x63] = lambda: self._alu(addr.stack_relative, adc8, adc16, 4)
        t[0x73] = lambda: self._alu(addr.stack_relative_indirect_y, adc8, adc16, 7)

        # ===== SBC =====
        t[0xE9] = lambda: self._alu_imm(sbc8, sbc16)
        t[0xE5] = lambda: self._alu(addr.direct, sbc8, sbc16, 3)
        t[0xF5] = lambda: self._alu(addr.direct_x, sbc8, sbc16, 4)
        t[0xED] = lambda: self._alu(addr.absolute, sbc8, sbc16, 4)
        t[0xFD] = lambda: self._alu(addr.absolute_x, sbc8, sbc16, 4)
        t[0xF9] = lambda: self._alu(addr.absolute_y, sbc8, sbc16, 4)
        t[0xEF] = lambda: self._alu(addr.absolute_long, sbc8, sbc16, 5)
        t[0xFF] = lambda: self._alu(addr.absolute_long_x, sbc8, sbc16, 5)
        t[0xE1] = lambda: self._alu(addr.direct_x_indirect, sbc8, sbc16, 6)
        t[0xF1] = lambda: self._alu(addr.direct_indirect_y, sbc8, sbc16, 5)
        t[0xF2] = lambda: self._alu(addr.direct_indirect, sbc8, sbc16, 5)
        t[0xE7] = lambda: self._alu(addr.direct_indirect_long, sbc8, sbc16, 6)
        t[0xF7] = lambda: self._alu(addr.direct_indirect_long_y, sbc8, sbc16, 6)
        t[0xE3] = lambda: self._alu(addr.stack_relative, sbc8, sbc16, 4)
        t[0xF3] = lambda: self._alu(addr.stack_relative_indirect_y, sbc8, sbc16, 7)

        # ===== AND =====
        t[0x29] = lambda: self._alu_imm(and8, and16)
        t[0x25] = lambda: self._alu(addr.direct, and8, and16, 3)
        t[0x35] = lambda: self._alu(addr.direct_x, and8, and16, 4)
        t[0x2D] = lambda: self._alu(addr.absolute, and8, and16, 4)
        t[0x3D] = lambda: self._alu(addr.absolute_x, and8, and16, 4)
        t[0x39] = lambda: self._alu(addr.absolute_y, and8, and16, 4)
        t[0x2F] = lambda: self._alu(addr.absolute_long, and8, and16, 5)
        t[0x3F] = lambda: self._alu(addr.absolute_long_x, and8, and16, 5)
        t[0x21] = lambda: self._alu(addr.direct_x_indirect, and8, and16, 6)
        t[0x31] = lambda: self._alu(addr.direct_indirect_y, and8, and16, 5)
        t[0x32] = lambda: self._alu(addr.direct_indirect, and8, and16, 5)
        t[0x27] = lambda: self._alu(addr.direct_indirect_long, and8, and16, 6)
        t[0x37] = lambda: self._alu(addr.direct_indirect_long_y, and8, and16, 6)
        t[0x23] = lambda: self._alu(addr.stack_relative, and8, and16, 4)
        t[0x33] = lambda: self._alu(addr.stack_relative_indirect_y, and8, and16, 7)

        # ===== ORA =====
        t[0x09] = lambda: self._alu_imm(ora8, ora16)
        t[0x05] = lambda: self._alu(addr.direct, ora8, ora16, 3)
        t[0x15] = lambda: self._alu(addr.direct_x, ora8, ora16, 4)
        t[0x0D] = lambda: self._alu(addr.absolute, ora8, ora16, 4)
        t[0x1D] = lambda: self._alu(addr.absolute_x, ora8, ora16, 4)
        t[0x19] = lambda: self._alu(addr.absolute_y, ora8, ora16, 4)
        t[0x0F] = lambda: self._alu(addr.absolute_long, ora8, ora16, 5)
        t[0x1F] = lambda: self._alu(addr.absolute_long_x, ora8, ora16, 5)
        t[0x01] = lambda: self._alu(addr.direct_x_indirect, ora8, ora16, 6)
        t[0x11] = lambda: self._alu(addr.direct_indirect_y, ora8, ora16, 5)
        t[0x12] = lambda: self._alu(addr.direct_indirect, ora8, ora16, 5)
        t[0x07] = lambda: self._alu(addr.direct_indirect_long, ora8, ora16, 6)
        t[0x17] = lambda: self._alu(addr.direct_indirect_long_y, ora8, ora16, 6)
        t[0x03] = lambda: self._alu(addr.stack_relative, ora8, ora16, 4)
        t[0x13] = lambda: self._alu(addr.stack_relative_indirect_y, ora8, ora16, 7)

        # ===== EOR =====
        t[0x49] = lambda: self._alu_imm(eor8, eor16)
        t[0x45] = lambda: self._alu(addr.direct, eor8, eor16, 3)
        t[0x55] = lambda: self._alu(addr.direct_x, eor8, eor16, 4)
        t[0x4D] = lambda: self._alu(addr.absolute, eor8, eor16, 4)
        t[0x5D] = lambda: self._alu(addr.absolute_x, eor8, eor16, 4)
        t[0x59] = lambda: self._alu(addr.absolute_y, eor8, eor16, 4)
        t[0x4F] = lambda: self._alu(addr.absolute_long, eor8, eor16, 5)
        t[0x5F] = lambda: self._alu(addr.absolute_long_x, eor8, eor16, 5)
        t[0x41] = lambda: self._alu(addr.direct_x_indirect, eor8, eor16, 6)
        t[0x51] = lambda: self._alu(addr.direct_indirect_y, eor8, eor16, 5)
        t[0x52] = lambda: self._alu(addr.direct_indirect, eor8, eor16, 5)
        t[0x47] = lambda: self._alu(addr.direct_indirect_long, eor8, eor16, 6)
        t[0x57] = lambda: self._alu(addr.direct_indirect_long_y, eor8, eor16, 6)
        t[0x43] = lambda: self._alu(addr.stack_relative, eor8, eor16, 4)
        t[0x53] = lambda: self._alu(addr.stack_relative_indirect_y, eor8, eor16, 7)

        # ===== CMP =====
        # CMP uses a wrapper since cmp8/cmp16 take (cpu, reg, value) not (cpu, value)
        def cmp8_wrap(cpu, val): cmp8(cpu, cpu.A & 0xFF, val)
        def cmp16_wrap(cpu, val): cmp16(cpu, cpu.A, val)
        t[0xC9] = lambda: self._alu_imm(cmp8_wrap, cmp16_wrap)
        t[0xC5] = lambda: self._alu(addr.direct, cmp8_wrap, cmp16_wrap, 3)
        t[0xD5] = lambda: self._alu(addr.direct_x, cmp8_wrap, cmp16_wrap, 4)
        t[0xCD] = lambda: self._alu(addr.absolute, cmp8_wrap, cmp16_wrap, 4)
        t[0xDD] = lambda: self._alu(addr.absolute_x, cmp8_wrap, cmp16_wrap, 4)
        t[0xD9] = lambda: self._alu(addr.absolute_y, cmp8_wrap, cmp16_wrap, 4)
        t[0xCF] = lambda: self._alu(addr.absolute_long, cmp8_wrap, cmp16_wrap, 5)
        t[0xDF] = lambda: self._alu(addr.absolute_long_x, cmp8_wrap, cmp16_wrap, 5)
        t[0xC1] = lambda: self._alu(addr.direct_x_indirect, cmp8_wrap, cmp16_wrap, 6)
        t[0xD1] = lambda: self._alu(addr.direct_indirect_y, cmp8_wrap, cmp16_wrap, 5)
        t[0xD2] = lambda: self._alu(addr.direct_indirect, cmp8_wrap, cmp16_wrap, 5)
        t[0xC7] = lambda: self._alu(addr.direct_indirect_long, cmp8_wrap, cmp16_wrap, 6)
        t[0xD7] = lambda: self._alu(addr.direct_indirect_long_y, cmp8_wrap, cmp16_wrap, 6)
        t[0xC3] = lambda: self._alu(addr.stack_relative, cmp8_wrap, cmp16_wrap, 4)
        t[0xD3] = lambda: self._alu(addr.stack_relative_indirect_y, cmp8_wrap, cmp16_wrap, 7)

        # ===== CPX =====
        t[0xE0] = lambda: self._cmp_idx_imm(get_x, cmp8, cmp16)
        t[0xE4] = lambda: self._cmp_idx(addr.direct, get_x, cmp8, cmp16, 3)
        t[0xEC] = lambda: self._cmp_idx(addr.absolute, get_x, cmp8, cmp16, 4)

        # ===== CPY =====
        t[0xC0] = lambda: self._cmp_idx_imm(get_y, cmp8, cmp16)
        t[0xC4] = lambda: self._cmp_idx(addr.direct, get_y, cmp8, cmp16, 3)
        t[0xCC] = lambda: self._cmp_idx(addr.absolute, get_y, cmp8, cmp16, 4)

        # ===== BIT =====
        t[0x89] = self._bit_imm
        t[0x24] = lambda: self._bit(addr.direct, 3)
        t[0x34] = lambda: self._bit(addr.direct_x, 4)
        t[0x2C] = lambda: self._bit(addr.absolute, 4)
        t[0x3C] = lambda: self._bit(addr.absolute_x, 4)

        # ===== ASL =====
        t[0x0A] = lambda: self._rmw_acc(asl8, asl16)
        t[0x06] = lambda: self._rmw(addr.direct, asl8, asl16, 5)
        t[0x16] = lambda: self._rmw(addr.direct_x, asl8, asl16, 6)
        t[0x0E] = lambda: self._rmw(addr.absolute, asl8, asl16, 6)
        t[0x1E] = lambda: self._rmw(addr.absolute_x_no_penalty, asl8, asl16, 7)

        # ===== LSR =====
        t[0x4A] = lambda: self._rmw_acc(lsr8, lsr16)
        t[0x46] = lambda: self._rmw(addr.direct, lsr8, lsr16, 5)
        t[0x56] = lambda: self._rmw(addr.direct_x, lsr8, lsr16, 6)
        t[0x4E] = lambda: self._rmw(addr.absolute, lsr8, lsr16, 6)
        t[0x5E] = lambda: self._rmw(addr.absolute_x_no_penalty, lsr8, lsr16, 7)

        # ===== ROL =====
        t[0x2A] = lambda: self._rmw_acc(rol8, rol16)
        t[0x26] = lambda: self._rmw(addr.direct, rol8, rol16, 5)
        t[0x36] = lambda: self._rmw(addr.direct_x, rol8, rol16, 6)
        t[0x2E] = lambda: self._rmw(addr.absolute, rol8, rol16, 6)
        t[0x3E] = lambda: self._rmw(addr.absolute_x_no_penalty, rol8, rol16, 7)

        # ===== ROR =====
        t[0x6A] = lambda: self._rmw_acc(ror8, ror16)
        t[0x66] = lambda: self._rmw(addr.direct, ror8, ror16, 5)
        t[0x76] = lambda: self._rmw(addr.direct_x, ror8, ror16, 6)
        t[0x6E] = lambda: self._rmw(addr.absolute, ror8, ror16, 6)
        t[0x7E] = lambda: self._rmw(addr.absolute_x_no_penalty, ror8, ror16, 7)

        # ===== INC =====
        t[0x1A] = lambda: self._rmw_acc(inc8, inc16)
        t[0xE6] = lambda: self._rmw(addr.direct, inc8, inc16, 5)
        t[0xF6] = lambda: self._rmw(addr.direct_x, inc8, inc16, 6)
        t[0xEE] = lambda: self._rmw(addr.absolute, inc8, inc16, 6)
        t[0xFE] = lambda: self._rmw(addr.absolute_x_no_penalty, inc8, inc16, 7)

        # ===== DEC =====
        t[0x3A] = lambda: self._rmw_acc(dec8, dec16)
        t[0xC6] = lambda: self._rmw(addr.direct, dec8, dec16, 5)
        t[0xD6] = lambda: self._rmw(addr.direct_x, dec8, dec16, 6)
        t[0xCE] = lambda: self._rmw(addr.absolute, dec8, dec16, 6)
        t[0xDE] = lambda: self._rmw(addr.absolute_x_no_penalty, dec8, dec16, 7)

        # ===== TSB/TRB =====
        t[0x04] = lambda: self._tsb(addr.direct, 5)
        t[0x0C] = lambda: self._tsb(addr.absolute, 6)
        t[0x14] = lambda: self._trb(addr.direct, 5)
        t[0x1C] = lambda: self._trb(addr.absolute, 6)

        # ===== TRANSFERS =====
        t[0xAA] = self._tax
        t[0xA8] = self._tay
        t[0x8A] = self._txa
        t[0x98] = self._tya
        t[0xBA] = self._tsx
        t[0x9A] = self._txs
        t[0x9B] = self._txy
        t[0xBB] = self._tyx
        t[0x5B] = self._tcd
        t[0x7B] = self._tdc
        t[0x1B] = self._tcs
        t[0x3B] = self._tsc

        # ===== INDEX INC/DEC =====
        t[0xE8] = self._inx
        t[0xC8] = self._iny
        t[0xCA] = self._dex
        t[0x88] = self._dey

        # ===== STACK =====
        t[0x48] = self._pha
        t[0xDA] = self._phx
        t[0x5A] = self._phy
        t[0x68] = self._pla
        t[0xFA] = self._plx
        t[0x7A] = self._ply
        t[0x08] = self._php
        t[0x28] = self._plp
        t[0x8B] = self._phb
        t[0xAB] = self._plb
        t[0x0B] = self._phd
        t[0x2B] = self._pld
        t[0x4B] = self._phk
        t[0xF4] = self._pea
        t[0xD4] = self._pei
        t[0x62] = self._per

        # ===== BRANCHES =====
        t[0x80] = self._bra
        t[0x82] = self._brl
        t[0xF0] = self._beq
        t[0xD0] = self._bne
        t[0xB0] = self._bcs
        t[0x90] = self._bcc
        t[0x30] = self._bmi
        t[0x10] = self._bpl
        t[0x70] = self._bvs
        t[0x50] = self._bvc

        # ===== JUMPS =====
        t[0x4C] = self._jmp_abs
        t[0x5C] = self._jmp_long
        t[0x6C] = self._jmp_ind
        t[0xDC] = self._jmp_ind_long
        t[0x7C] = self._jmp_indexed_ind
        t[0x20] = self._jsr_abs
        t[0x22] = self._jsr_long
        t[0xFC] = self._jsr_indexed_ind
        t[0x60] = self._rts
        t[0x6B] = self._rtl
        t[0x40] = self._rti

        # ===== FLAGS =====
        t[0x18] = self._clc
        t[0x38] = self._sec
        t[0x58] = self._cli
        t[0x78] = self._sei
        t[0xD8] = self._cld
        t[0xF8] = self._sed
        t[0xB8] = self._clv
        t[0xE2] = self._sep
        t[0xC2] = self._rep
        t[0xFB] = self._xce

        # ===== MISCELLANEOUS =====
        t[0xEA] = self._nop
        t[0x42] = self._wdm
        t[0xEB] = self._xba
        t[0xDB] = self._stp
        t[0xCB] = self._wai
        t[0x00] = self._brk
        t[0x02] = self._cop
        t[0x54] = self._mvn
        t[0x44] = self._mvp

        return t
