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
