# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
SNES ROM bank sizes. Bank overflow itself is reported by wlalink, which lays
out every bank from the emitted `.BANK` directives.
"""

LOROM_BANK_SIZE = 0x8000   # 32KB
HIROM_BANK_SIZE = 0x10000  # 64KB


def rom_bank_size(is_hirom: bool) -> int:
    """Bytes in one ROM bank of the given mapping."""
    return HIROM_BANK_SIZE if is_hirom else LOROM_BANK_SIZE
