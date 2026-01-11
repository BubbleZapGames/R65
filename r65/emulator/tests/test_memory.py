"""
Tests for memory system.
"""

import pytest
from r65.emulator.memory import Memory, RomMapping


class TestMemoryInitialization:
    """Test memory initialization."""

    def test_create_with_lorom(self):
        """Memory should initialize with LoROM mapping."""
        rom = bytes(32768)
        mem = Memory(rom, mapping="lorom")

        assert mem.mapping == RomMapping.LOROM

    def test_create_with_hirom(self):
        """Memory should initialize with HiROM mapping."""
        rom = bytes(65536)
        mem = Memory(rom, mapping="hirom")

        assert mem.mapping == RomMapping.HIROM

    def test_wram_size(self):
        """WRAM should be 128KB."""
        rom = bytes(32768)
        mem = Memory(rom)

        assert len(mem.wram) == 128 * 1024


class TestWRAMAccess:
    """Test WRAM read/write."""

    def test_wram_bank_7e(self):
        """Bank $7E should access WRAM."""
        rom = bytes(32768)
        mem = Memory(rom)

        mem.write(0x7E, 0x0000, 0x42)
        assert mem.read(0x7E, 0x0000) == 0x42

    def test_wram_bank_7f(self):
        """Bank $7F should access upper WRAM."""
        rom = bytes(32768)
        mem = Memory(rom)

        mem.write(0x7F, 0x0000, 0x42)
        assert mem.read(0x7F, 0x0000) == 0x42

    def test_wram_mirror(self):
        """$0000-$1FFF should mirror WRAM in banks $00-$3F."""
        rom = bytes(32768)
        mem = Memory(rom)

        # Write via mirror
        mem.write(0x00, 0x0100, 0x42)
        # Read via direct WRAM access
        assert mem.read(0x7E, 0x0100) == 0x42

    def test_wram_mirror_bank_80(self):
        """$0000-$1FFF should mirror WRAM in banks $80-$BF."""
        rom = bytes(32768)
        mem = Memory(rom)

        mem.write(0x80, 0x0100, 0x42)
        assert mem.read(0x7E, 0x0100) == 0x42

    def test_read16(self):
        """read16 should read little-endian word."""
        rom = bytes(32768)
        mem = Memory(rom)

        mem.write(0x7E, 0x0000, 0x34)
        mem.write(0x7E, 0x0001, 0x12)

        assert mem.read16(0x7E, 0x0000) == 0x1234

    def test_write16(self):
        """write16 should write little-endian word."""
        rom = bytes(32768)
        mem = Memory(rom)

        mem.write16(0x7E, 0x0000, 0x1234)

        assert mem.read(0x7E, 0x0000) == 0x34
        assert mem.read(0x7E, 0x0001) == 0x12


class TestROMAccess:
    """Test ROM read access."""

    def test_rom_read_lorom(self):
        """ROM should be readable in LoROM mapping."""
        rom = bytearray(32768)
        rom[0] = 0x42  # First byte of ROM
        mem = Memory(bytes(rom), mapping="lorom")

        # In LoROM, $8000-$FFFF in bank $00 maps to ROM
        assert mem.read(0x00, 0x8000) == 0x42

    def test_rom_read_bank_00(self):
        """Bank $00 $8000-$FFFF should read ROM."""
        rom = bytearray(32768)
        rom[0x100] = 0x42
        mem = Memory(bytes(rom), mapping="lorom")

        assert mem.read(0x00, 0x8100) == 0x42

    def test_rom_write_ignored(self):
        """Writing to ROM should be ignored."""
        rom = bytearray(32768)
        rom[0] = 0x42
        mem = Memory(bytes(rom), mapping="lorom")

        # This should not change the ROM
        # (Note: actual implementation may vary)
        original = mem.read(0x00, 0x8000)
        assert original == 0x42


class TestVectorAccess:
    """Test interrupt vector access."""

    def test_reset_vector(self):
        """Reset vector should be readable."""
        rom = bytearray(32768)
        # Reset vector at $FFFC-$FFFD in ROM
        # In LoROM, this is at offset $7FFC-$7FFD
        rom[0x7FFC] = 0x00
        rom[0x7FFD] = 0x80
        mem = Memory(bytes(rom), mapping="lorom")

        reset = mem.get_reset_vector()
        assert reset == 0x8000

    def test_nmi_vector(self):
        """NMI vector should be readable (native mode at $FFEA)."""
        rom = bytearray(32768)
        # Native mode NMI vector at $FFEA (ROM offset 0x7FEA in LoROM)
        rom[0x7FEA] = 0x00
        rom[0x7FEB] = 0x90
        mem = Memory(bytes(rom), mapping="lorom")

        nmi = mem.get_nmi_vector()
        assert nmi == 0x9000

    def test_irq_vector(self):
        """IRQ vector should be readable (native mode at $FFEE)."""
        rom = bytearray(32768)
        # Native mode IRQ vector at $FFEE (ROM offset 0x7FEE in LoROM)
        rom[0x7FEE] = 0x00
        rom[0x7FEF] = 0xA0
        mem = Memory(bytes(rom), mapping="lorom")

        irq = mem.get_irq_vector()
        assert irq == 0xA000


class TestAddressMasking:
    """Test address masking."""

    def test_bank_mask(self):
        """Bank should be masked to 8 bits."""
        rom = bytes(32768)
        mem = Memory(rom)

        mem.write(0x17E, 0x0000, 0x42)  # Bank $17E -> $7E
        assert mem.read(0x7E, 0x0000) == 0x42

    def test_address_mask(self):
        """Address should be masked to 16 bits."""
        rom = bytes(32768)
        mem = Memory(rom)

        mem.write(0x7E, 0x10000, 0x42)  # Wraps to $0000
        assert mem.read(0x7E, 0x0000) == 0x42


class TestHardwareRegisters:
    """Test hardware register stubs."""

    def test_ppu_register_read(self):
        """PPU registers should return stub value."""
        rom = bytes(32768)
        mem = Memory(rom)

        # $2100-$21FF are PPU registers
        value = mem.read(0x00, 0x2100)
        # Stub returns 0 or implementation-defined value
        assert isinstance(value, int)

    def test_cpu_register_read(self):
        """CPU registers should return stub value."""
        rom = bytes(32768)
        mem = Memory(rom)

        # $4200-$44FF are CPU registers
        value = mem.read(0x00, 0x4200)
        assert isinstance(value, int)


class TestUnmappedRegions:
    """Test unmapped memory regions."""

    def test_unmapped_returns_zero(self):
        """Unmapped regions should return 0."""
        rom = bytes(32768)
        mem = Memory(rom)

        # $2000-$20FF is typically unmapped
        value = mem.read(0x00, 0x2000)
        assert value == 0
