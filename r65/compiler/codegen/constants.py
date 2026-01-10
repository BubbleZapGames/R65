"""
Memory layout constants for 65816/SNES code generation.

Defines memory region boundaries, stack defaults, and addressing mode
thresholds used throughout the code generation pipeline.
"""

# =============================================================================
# Low RAM Region ($0000-$1FFF)
# =============================================================================

# Low RAM contains direct page (zeropage), stack, and general-purpose RAM
LOWRAM_START = 0x0000
LOWRAM_END = 0x1FFF

# Direct page (zeropage) - uses fast DP addressing
ZEROPAGE_START = 0x0000
ZEROPAGE_END = 0x00FF

# =============================================================================
# Stack Defaults
# =============================================================================

# Default stack region if no #[stack] attribute specified
DEFAULT_STACK_LOWER = 0x0100
DEFAULT_STACK_UPPER = 0x01FF

# =============================================================================
# Main RAM Region (SNES Work RAM)
# =============================================================================

# Main RAM starts after low RAM mirror
RAM_START = 0x7E2000
RAM_END = 0x7FFFFF

# =============================================================================
# ROM Memory Map (LoROM/HiROM)
# =============================================================================

# LoROM configuration
LOROM_SLOT_SIZE = 0x8000   # 32KB
LOROM_SLOT_ADDR = 0x8000

# HiROM configuration
HIROM_SLOT_SIZE = 0x10000  # 64KB
HIROM_SLOT_ADDR = 0x0000

# =============================================================================
# Addressing Mode Thresholds
# =============================================================================

# Direct page addressing: $00-$FF
DP_MAX = 0xFF

# Absolute addressing: $0100-$FFFF
ABSOLUTE_MIN = 0x0100
ABSOLUTE_MAX = 0xFFFF

# Long addressing: > $FFFF (24-bit address)
LONG_MIN = 0x10000

# =============================================================================
# ROM Size Calculation
# =============================================================================

def calculate_rom_size(bank_count: int, is_hirom: bool = False) -> tuple:
    """
    Calculate minimum ROM size parameters for WLA-DX.

    Args:
        bank_count: Number of banks actually used
        is_hirom: True for HiROM (64KB banks), False for LoROM (32KB banks)

    Returns:
        Tuple of (rom_banks, romsize_value):
        - rom_banks: Power-of-2 number of banks for .ROMBANKMAP
        - romsize_value: SNES header ROMSIZE value ($08-$0D)

    ROMSIZE values (from SNES header spec):
        $08 = 256 KB  (8 LoROM banks / 4 HiROM banks)
        $09 = 512 KB  (16 LoROM banks / 8 HiROM banks)
        $0A = 1 MB    (32 LoROM banks / 16 HiROM banks)
        $0B = 2 MB    (64 LoROM banks / 32 HiROM banks)
        $0C = 4 MB    (128 LoROM banks / 64 HiROM banks)
        $0D = 8 MB    (reserved)
    """
    import math

    # Bank size in KB
    bank_size_kb = 64 if is_hirom else 32

    # Round bank count up to next power of 2
    if bank_count <= 0:
        bank_count = 1
    rom_banks = 1 << (bank_count - 1).bit_length()

    # Calculate total ROM size in KB
    rom_size_kb = rom_banks * bank_size_kb

    # Minimum ROM size is 256KB (ROMSIZE $08)
    min_rom_kb = 256
    if rom_size_kb < min_rom_kb:
        rom_size_kb = min_rom_kb
        rom_banks = min_rom_kb // bank_size_kb

    # Calculate ROMSIZE value: $08 = 256KB, each increment doubles size
    # ROMSIZE = 8 + log2(size_in_256KB)
    romsize_value = 8 + int(math.log2(rom_size_kb // 256))

    # Cap at $0D (8MB)
    romsize_value = min(romsize_value, 0x0D)

    return (rom_banks, romsize_value)
