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
# Processor Status Flag Masks
# =============================================================================

# 65816 processor status register bits for mode control
M_FLAG = 0x20         # Accumulator mode: 0=16-bit, 1=8-bit
X_FLAG = 0x10         # Index register mode: 0=16-bit, 1=8-bit
MX_FLAGS = 0x30       # Both M and X flags combined

# =============================================================================
# Bit Masks and Value Limits
# =============================================================================

# Common bit masks for value operations
BYTE_MASK = 0xFF      # 8-bit mask
WORD_MASK = 0xFFFF    # 16-bit mask

# Size boundary for direct page addressing
DP_BOUNDARY = 0x100   # Addresses >= this use absolute addressing

# =============================================================================
# SNES Memory Banks
# =============================================================================

# SNES Work RAM bank numbers (for block move instructions)
WRAM_BANK = 0x7E      # Work RAM bank ($7E0000-$7EFFFF)
WRAM_BANK2 = 0x7F     # Work RAM bank 2 ($7F0000-$7FFFFF)

# SNES Work RAM bank start addresses
WRAM_BANK_START = 0x7E0000   # Start of WRAM bank $7E
WRAM_BANK2_START = 0x7F0000  # Start of WRAM bank $7F

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
