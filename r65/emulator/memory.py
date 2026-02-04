"""
65816 Memory System with SNES LoROM/HiROM mapping.
"""

from typing import Optional
from enum import Enum


class RomMapping(Enum):
    LOROM = "lorom"
    HIROM = "hirom"


class Memory:
    """
    Base memory class for 65816 systems.

    All addresses are 24-bit (bank:offset combined as a single int).
    Provides read16/read24/write16 helpers that delegate to subclass
    read()/write() implementations.
    """

    def __init__(self, rom_data: bytes):
        """
        Initialize memory with ROM data.

        Args:
            rom_data: Raw ROM bytes
        """
        self.rom = bytearray(rom_data)

    def read(self, addr: int) -> int:
        """Read a byte from a 24-bit address."""
        raise NotImplementedError

    def write(self, addr: int, value: int):
        """Write a byte to a 24-bit address."""
        raise NotImplementedError

    def read16(self, addr: int) -> int:
        """Read a 16-bit value (little-endian). Wraps within the same bank."""
        lo = self.read(addr)
        hi = self.read((addr & 0xFF0000) | ((addr + 1) & 0xFFFF))
        return lo | (hi << 8)

    def read24(self, addr: int) -> int:
        """Read a 24-bit value (little-endian). Wraps within the same bank."""
        lo = self.read(addr)
        mid = self.read((addr & 0xFF0000) | ((addr + 1) & 0xFFFF))
        hi = self.read((addr & 0xFF0000) | ((addr + 2) & 0xFFFF))
        return lo | (mid << 8) | (hi << 16)

    def write16(self, addr: int, value: int):
        """Write a 16-bit value (little-endian). Wraps within the same bank."""
        self.write(addr, value & 0xFF)
        self.write((addr & 0xFF0000) | ((addr + 1) & 0xFFFF), (value >> 8) & 0xFF)


class SNESMemory(Memory):
    """
    SNES memory map implementation supporting LoROM and HiROM.

    Memory regions:
    - WRAM: 128KB at $7E0000-$7FFFFF, mirrored at $0000-$1FFF in banks $00-$3F, $80-$BF
    - ROM: Mapped according to LoROM or HiROM layout
    - Hardware registers: $2100-$21FF, $4200-$44FF

    Math hardware registers:
    - $4202 WRMPYA: Multiplicand A (write-only)
    - $4203 WRMPYB: Multiplicand B (write triggers 8x8 multiply)
    - $4204 WRDIVL: Dividend low byte (write-only)
    - $4205 WRDIVH: Dividend high byte (write-only)
    - $4206 WRDIVB: Divisor (write triggers 16/8 divide)
    - $4214 RDDIVL: Quotient low byte (read-only)
    - $4215 RDDIVH: Quotient high byte (read-only)
    - $4216 RDMPYL: Product/Remainder low byte (read-only)
    - $4217 RDMPYH: Product/Remainder high byte (read-only)
    """

    WRAM_SIZE = 128 * 1024  # 128KB

    # Math hardware register addresses
    NMITIMEN = 0x4200  # NMI/IRQ enable
    WRMPYA = 0x4202  # Multiplicand A
    WRMPYB = 0x4203  # Multiplicand B (write triggers multiply)
    WRDIVL = 0x4204  # Dividend low
    WRDIVH = 0x4205  # Dividend high
    WRDIVB = 0x4206  # Divisor (write triggers divide)
    RDNMI = 0x4210   # NMI flag (bit 7) and CPU version
    RDDIVL = 0x4214  # Quotient low (read-only)
    RDDIVH = 0x4215  # Quotient high (read-only)
    RDMPYL = 0x4216  # Product/Remainder low (read-only)
    RDMPYH = 0x4217  # Product/Remainder high (read-only)

    def __init__(self, rom_data: bytes, mapping: str = "lorom"):
        """
        Initialize memory with ROM data.

        Args:
            rom_data: Raw ROM bytes
            mapping: "lorom" or "hirom"
        """
        super().__init__(rom_data)
        self.wram = bytearray(self.WRAM_SIZE)
        self.mapping = RomMapping(mapping.lower())

        # Hardware register stubs (return 0 on read, ignore writes)
        self._hw_regs = {}

        # Math hardware state
        self._wrmpya = 0      # Multiplicand A
        self._wrdiv = 0       # 16-bit dividend
        self._rdmpy = 0       # 16-bit product/remainder
        self._rddiv = 0       # 16-bit quotient

        # APU I/O port state (echoes writes back to reads after handshake)
        self._apu_initialized = False
        self._apu_ports = [0xAA, 0xBB, 0x00, 0x00]  # Initial ready state

        # CPU reference for RDNMI ($4210) reads
        self._cpu = None

    def read(self, addr: int) -> int:
        """Read a byte from a 24-bit address."""
        bank = (addr >> 16) & 0xFF
        offset = addr & 0xFFFF

        # WRAM banks $7E-$7F
        if bank == 0x7E or bank == 0x7F:
            wram_offset = ((bank - 0x7E) << 16) | offset
            if wram_offset < self.WRAM_SIZE:
                return self.wram[wram_offset]
            return 0

        # Banks $00-$3F and $80-$BF
        if bank <= 0x3F or (0x80 <= bank <= 0xBF):
            # WRAM mirror at $0000-$1FFF
            if offset < 0x2000:
                return self.wram[offset]

            # PPU registers $2100-$21FF (stub)
            if 0x2100 <= offset <= 0x21FF:
                return self._read_hw_reg(bank, offset)

            # CPU registers $4200-$44FF (stub)
            if 0x4200 <= offset <= 0x44FF:
                return self._read_hw_reg(bank, offset)

            # ROM area
            if offset >= 0x8000:
                return self._read_rom(bank, offset)

            # Unmapped
            return 0

        # Banks $40-$7D (HiROM full banks, LoROM mirrors)
        if 0x40 <= bank <= 0x7D:
            if self.mapping == RomMapping.HIROM:
                return self._read_rom(bank, offset)
            else:
                # LoROM: upper half is ROM
                if offset >= 0x8000:
                    return self._read_rom(bank, offset)
                return 0

        # Banks $C0-$FF (ROM)
        if 0xC0 <= bank <= 0xFF:
            return self._read_rom(bank, offset)

        return 0

    def write(self, addr: int, value: int):
        """Write a byte to a 24-bit address."""
        bank = (addr >> 16) & 0xFF
        offset = addr & 0xFFFF
        value &= 0xFF

        # WRAM banks $7E-$7F
        if bank == 0x7E or bank == 0x7F:
            wram_offset = ((bank - 0x7E) << 16) | offset
            if wram_offset < self.WRAM_SIZE:
                self.wram[wram_offset] = value
            return

        # Banks $00-$3F and $80-$BF
        if bank <= 0x3F or (0x80 <= bank <= 0xBF):
            # WRAM mirror at $0000-$1FFF
            if offset < 0x2000:
                self.wram[offset] = value
                return

            # PPU registers $2100-$21FF (stub)
            if 0x2100 <= offset <= 0x21FF:
                self._write_hw_reg(bank, offset, value)
                return

            # CPU registers $4200-$44FF (stub)
            if 0x4200 <= offset <= 0x44FF:
                self._write_hw_reg(bank, offset, value)
                return

            # ROM area - ignore writes
            return

        # ROM banks - ignore writes
        return

    def _read_rom(self, bank: int, addr: int) -> int:
        """Read from ROM using appropriate mapping."""
        offset = self._rom_offset(bank, addr)
        if offset is not None and offset < len(self.rom):
            return self.rom[offset]
        return 0

    def _rom_offset(self, bank: int, addr: int) -> Optional[int]:
        """Calculate ROM file offset from bank:addr."""
        if self.mapping == RomMapping.LOROM:
            # LoROM: $8000-$FFFF in each bank = 32KB per bank
            # Banks $00-$7D, $80-$FF (mirrored)
            if addr < 0x8000:
                return None
            effective_bank = bank & 0x7F
            offset = (effective_bank * 0x8000) + (addr - 0x8000)
            return offset
        else:
            # HiROM: Full 64KB banks
            # Banks $40-$7D, $C0-$FF map directly
            # Banks $00-$3F, $80-$BF only upper half ($8000-$FFFF)
            if bank <= 0x3F or (0x80 <= bank <= 0xBF):
                if addr < 0x8000:
                    return None
                effective_bank = bank & 0x3F
                offset = (effective_bank * 0x10000) + addr
                return offset
            elif 0x40 <= bank <= 0x7D:
                effective_bank = bank - 0x40
                offset = (effective_bank * 0x10000) + addr
                return offset
            elif 0xC0 <= bank <= 0xFF:
                effective_bank = bank - 0xC0
                offset = (effective_bank * 0x10000) + addr
                return offset
            return None

    def set_cpu(self, cpu) -> None:
        """Set CPU reference for RDNMI reads."""
        self._cpu = cpu

    def _read_hw_reg(self, bank: int, addr: int) -> int:
        """Read from hardware register."""
        # APU I/O ports $2140-$2143 - SPC700 communication
        # Echoes back what was written (simulates SPC700 acknowledgment)
        if 0x2140 <= addr <= 0x2143:
            port = addr - 0x2140
            return self._apu_ports[port]

        # RDNMI ($4210) - NMI flag in bit 7, CPU version in bits 0-3
        if addr == self.RDNMI:
            if self._cpu:
                # Read and clear vblank flag, return with CPU version (2)
                vblank = self._cpu.read_nmi_flag()
                return (0x80 if vblank else 0x00) | 0x02  # Version 2
            return 0x02  # Just CPU version if no CPU reference

        # Math result registers (read-only)
        if addr == self.RDDIVL:
            return self._rddiv & 0xFF
        elif addr == self.RDDIVH:
            return (self._rddiv >> 8) & 0xFF
        elif addr == self.RDMPYL:
            return self._rdmpy & 0xFF
        elif addr == self.RDMPYH:
            return (self._rdmpy >> 8) & 0xFF

        # Return stored value or 0 for other registers
        return self._hw_regs.get(addr, 0)

    def _write_hw_reg(self, bank: int, addr: int, value: int):
        """Write to hardware register."""
        value &= 0xFF

        # APU I/O ports $2140-$2143 - SPC700 communication
        # Echo writes back for read to simulate APU acknowledgment
        if 0x2140 <= addr <= 0x2143:
            port = addr - 0x2140
            self._apu_ports[port] = value
            # Mark APU as initialized once we get past handshake
            if port == 0 and not self._apu_initialized:
                self._apu_initialized = True
            return

        # NMITIMEN - NMI/IRQ enable
        if addr == self.NMITIMEN:
            self._hw_regs[addr] = value
            # Update CPU's NMI enable flag based on bit 7
            if self._cpu:
                self._cpu.set_nmi_enabled((value & 0x80) != 0)
            return

        # Math registers
        if addr == self.WRMPYA:
            # Store multiplicand A
            self._wrmpya = value
        elif addr == self.WRMPYB:
            # Writing multiplicand B triggers 8x8 unsigned multiplication
            # Result = WRMPYA * WRMPYB (16-bit product)
            self._rdmpy = (self._wrmpya * value) & 0xFFFF
        elif addr == self.WRDIVL:
            # Store dividend low byte
            self._wrdiv = (self._wrdiv & 0xFF00) | value
        elif addr == self.WRDIVH:
            # Store dividend high byte
            self._wrdiv = (self._wrdiv & 0x00FF) | (value << 8)
        elif addr == self.WRDIVB:
            # Writing divisor triggers 16/8 unsigned division
            if value == 0:
                # Division by zero: quotient = 0xFFFF, remainder = dividend
                self._rddiv = 0xFFFF
                self._rdmpy = self._wrdiv
            else:
                # quotient = dividend / divisor
                # remainder = dividend % divisor
                self._rddiv = (self._wrdiv // value) & 0xFFFF
                self._rdmpy = (self._wrdiv % value) & 0xFFFF
        else:
            # Store other registers
            self._hw_regs[addr] = value


def detect_mapping(rom_data: bytes) -> str:
    """
    Auto-detect ROM mapping from header.

    Checks for valid header at LoROM ($7FC0) and HiROM ($FFC0) locations.
    """
    def score_header(offset: int) -> int:
        """Score likelihood of valid header at offset."""
        if offset + 32 > len(rom_data):
            return -1

        score = 0

        # Check map mode byte
        map_mode = rom_data[offset + 0x15] if offset + 0x15 < len(rom_data) else 0
        rom_type = rom_data[offset + 0x16] if offset + 0x16 < len(rom_data) else 0
        rom_size = rom_data[offset + 0x17] if offset + 0x17 < len(rom_data) else 0
        ram_size = rom_data[offset + 0x18] if offset + 0x18 < len(rom_data) else 0

        # Valid map modes
        if map_mode in (0x20, 0x21, 0x30, 0x31):  # LoROM variants
            score += 2
        elif map_mode in (0x21, 0x25, 0x31, 0x35):  # HiROM variants
            score += 2

        # Valid ROM sizes (8=256KB, 9=512KB, 10=1MB, 11=2MB, 12=4MB)
        if 8 <= rom_size <= 12:
            score += 2

        # Valid RAM sizes (0=none, 1=2KB, 3=8KB, 5=32KB)
        if ram_size in (0, 1, 3, 5):
            score += 1

        # Check for ASCII title (first 21 bytes)
        title_chars = 0
        for i in range(21):
            if offset + i < len(rom_data):
                c = rom_data[offset + i]
                if 0x20 <= c <= 0x7E:  # Printable ASCII
                    title_chars += 1
        if title_chars >= 10:
            score += 2

        # Checksum complement check
        if offset + 0x1F < len(rom_data):
            checksum = rom_data[offset + 0x1E] | (rom_data[offset + 0x1F] << 8)
            complement = rom_data[offset + 0x1C] | (rom_data[offset + 0x1D] << 8)
            if (checksum ^ complement) == 0xFFFF:
                score += 3

        return score

    lorom_score = score_header(0x7FC0)  # LoROM header location
    hirom_score = score_header(0xFFC0)  # HiROM header location

    if hirom_score > lorom_score:
        return "hirom"
    return "lorom"
