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
    WRMPYA = 0x4202  # Multiplicand A
    WRMPYB = 0x4203  # Multiplicand B (write triggers multiply)
    WRDIVL = 0x4204  # Dividend low
    WRDIVH = 0x4205  # Dividend high
    WRDIVB = 0x4206  # Divisor (write triggers divide)
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
        self.rom = bytearray(rom_data)
        self.wram = bytearray(self.WRAM_SIZE)
        self.mapping = RomMapping(mapping.lower())

        # Hardware register stubs (return 0 on read, ignore writes)
        self._hw_regs = {}

        # Math hardware state
        self._wrmpya = 0      # Multiplicand A
        self._wrdiv = 0       # 16-bit dividend
        self._rdmpy = 0       # 16-bit product/remainder
        self._rddiv = 0       # 16-bit quotient

    def read(self, bank: int, addr: int) -> int:
        """Read a byte from the 24-bit address space."""
        bank &= 0xFF
        addr &= 0xFFFF

        # WRAM banks $7E-$7F
        if bank == 0x7E or bank == 0x7F:
            offset = ((bank - 0x7E) << 16) | addr
            if offset < self.WRAM_SIZE:
                return self.wram[offset]
            return 0

        # Banks $00-$3F and $80-$BF
        if bank <= 0x3F or (0x80 <= bank <= 0xBF):
            # WRAM mirror at $0000-$1FFF
            if addr < 0x2000:
                return self.wram[addr]

            # PPU registers $2100-$21FF (stub)
            if 0x2100 <= addr <= 0x21FF:
                return self._read_hw_reg(bank, addr)

            # CPU registers $4200-$44FF (stub)
            if 0x4200 <= addr <= 0x44FF:
                return self._read_hw_reg(bank, addr)

            # ROM area
            if addr >= 0x8000:
                return self._read_rom(bank, addr)

            # Unmapped
            return 0

        # Banks $40-$7D (HiROM full banks, LoROM mirrors)
        if 0x40 <= bank <= 0x7D:
            if self.mapping == RomMapping.HIROM:
                return self._read_rom(bank, addr)
            else:
                # LoROM: upper half is ROM
                if addr >= 0x8000:
                    return self._read_rom(bank, addr)
                return 0

        # Banks $C0-$FF (ROM)
        if 0xC0 <= bank <= 0xFF:
            return self._read_rom(bank, addr)

        return 0

    def read16(self, bank: int, addr: int) -> int:
        """Read a 16-bit value (little-endian)."""
        lo = self.read(bank, addr)
        # Handle page wrap
        hi = self.read(bank, (addr + 1) & 0xFFFF)
        return lo | (hi << 8)

    def read24(self, bank: int, addr: int) -> int:
        """Read a 24-bit value (little-endian)."""
        lo = self.read(bank, addr)
        mid = self.read(bank, (addr + 1) & 0xFFFF)
        hi = self.read(bank, (addr + 2) & 0xFFFF)
        return lo | (mid << 8) | (hi << 16)

    def write(self, bank: int, addr: int, value: int):
        """Write a byte to the 24-bit address space."""
        bank &= 0xFF
        addr &= 0xFFFF
        value &= 0xFF

        # WRAM banks $7E-$7F
        if bank == 0x7E or bank == 0x7F:
            offset = ((bank - 0x7E) << 16) | addr
            if offset < self.WRAM_SIZE:
                self.wram[offset] = value
            return

        # Banks $00-$3F and $80-$BF
        if bank <= 0x3F or (0x80 <= bank <= 0xBF):
            # WRAM mirror at $0000-$1FFF
            if addr < 0x2000:
                self.wram[addr] = value
                return

            # PPU registers $2100-$21FF (stub)
            if 0x2100 <= addr <= 0x21FF:
                self._write_hw_reg(bank, addr, value)
                return

            # CPU registers $4200-$44FF (stub)
            if 0x4200 <= addr <= 0x44FF:
                self._write_hw_reg(bank, addr, value)
                return

            # ROM area - ignore writes
            return

        # ROM banks - ignore writes
        return

    def write16(self, bank: int, addr: int, value: int):
        """Write a 16-bit value (little-endian)."""
        self.write(bank, addr, value & 0xFF)
        self.write(bank, (addr + 1) & 0xFFFF, (value >> 8) & 0xFF)

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

    def _read_hw_reg(self, bank: int, addr: int) -> int:
        """Read from hardware register."""
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

    def get_reset_vector(self) -> int:
        """Get the reset vector address from ROM."""
        # Reset vector is at $00:FFFC (native mode) or $00:FFFC (emulation)
        # For 65816 native mode, use bank 0
        return self.read16(0x00, 0xFFFC)

    def get_nmi_vector(self) -> int:
        """Get the NMI vector address."""
        return self.read16(0x00, 0xFFEA)

    def get_irq_vector(self) -> int:
        """Get the IRQ vector address."""
        return self.read16(0x00, 0xFFEE)


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
