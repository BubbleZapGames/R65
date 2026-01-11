"""
65816 ALU Operations.

Pure operations separated from instruction decoding.
Each operation has 8-bit and 16-bit variants.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cpu import CPU65816


# ============== LOAD OPERATIONS ==============

def lda8(cpu: 'CPU65816', value: int):
    """Load accumulator (8-bit mode)."""
    cpu.A = (cpu.A & 0xFF00) | (value & 0xFF)
    cpu.set_nz_flags(value, False)


def lda16(cpu: 'CPU65816', value: int):
    """Load accumulator (16-bit mode)."""
    cpu.A = value & 0xFFFF
    cpu.set_nz_flags(value, True)


def ldx8(cpu: 'CPU65816', value: int):
    """Load X register (8-bit mode)."""
    cpu.X = value & 0xFF
    cpu.set_nz_flags(value, False)


def ldx16(cpu: 'CPU65816', value: int):
    """Load X register (16-bit mode)."""
    cpu.X = value & 0xFFFF
    cpu.set_nz_flags(value, True)


def ldy8(cpu: 'CPU65816', value: int):
    """Load Y register (8-bit mode)."""
    cpu.Y = value & 0xFF
    cpu.set_nz_flags(value, False)


def ldy16(cpu: 'CPU65816', value: int):
    """Load Y register (16-bit mode)."""
    cpu.Y = value & 0xFFFF
    cpu.set_nz_flags(value, True)


# ============== ARITHMETIC OPERATIONS ==============

def adc8(cpu: 'CPU65816', value: int):
    """Add with carry (8-bit mode)."""
    a = cpu.A & 0xFF
    if cpu.flag_d:  # Decimal mode
        lo = (a & 0x0F) + (value & 0x0F) + (1 if cpu.flag_c else 0)
        if lo > 9:
            lo = (lo - 10) | 0x10
        hi = (a >> 4) + (value >> 4) + (lo >> 4)
        cpu.flag_v = bool(~(a ^ value) & (a ^ (hi << 4)) & 0x80)
        if hi > 9:
            hi += 6
        cpu.flag_c = hi > 15
        result = ((hi & 0x0F) << 4) | (lo & 0x0F)
    else:
        result = a + value + (1 if cpu.flag_c else 0)
        cpu.flag_c = result > 0xFF
        cpu.flag_v = bool(~(a ^ value) & (a ^ result) & 0x80)
        result &= 0xFF
    cpu.A = (cpu.A & 0xFF00) | result
    cpu.set_nz_flags(result, False)


def adc16(cpu: 'CPU65816', value: int):
    """Add with carry (16-bit mode)."""
    a = cpu.A
    if cpu.flag_d:
        lo = (a & 0x000F) + (value & 0x000F) + (1 if cpu.flag_c else 0)
        if lo > 9:
            lo = (lo - 10) | 0x10
        m1 = (a & 0x00F0) + (value & 0x00F0) + (lo & 0xF0)
        if m1 > 0x90:
            m1 = (m1 - 0xA0) | 0x100
        m2 = (a & 0x0F00) + (value & 0x0F00) + (m1 & 0xF00)
        if m2 > 0x900:
            m2 = (m2 - 0xA00) | 0x1000
        hi = (a >> 12) + (value >> 12) + (m2 >> 12)
        cpu.flag_v = bool(~(a ^ value) & (a ^ (hi << 12)) & 0x8000)
        if hi > 9:
            hi += 6
        cpu.flag_c = hi > 15
        result = ((hi & 0x0F) << 12) | (m2 & 0x0F00) | (m1 & 0x00F0) | (lo & 0x000F)
    else:
        result = a + value + (1 if cpu.flag_c else 0)
        cpu.flag_c = result > 0xFFFF
        cpu.flag_v = bool(~(a ^ value) & (a ^ result) & 0x8000)
        result &= 0xFFFF
    cpu.A = result
    cpu.set_nz_flags(result, True)


def sbc8(cpu: 'CPU65816', value: int):
    """Subtract with borrow (8-bit mode)."""
    a = cpu.A & 0xFF
    if cpu.flag_d:  # Decimal mode
        lo = (a & 0x0F) - (value & 0x0F) - (0 if cpu.flag_c else 1)
        if lo < 0:
            lo = ((lo + 10) & 0x0F) - 0x10
        hi = (a >> 4) - (value >> 4) + (lo >> 4)
        if hi < 0:
            hi = ((hi + 10) & 0x0F) - 0x10
        cpu.flag_c = (a - value - (0 if cpu.flag_c else 1)) >= 0
        result = ((hi & 0x0F) << 4) | (lo & 0x0F)
        cpu.flag_v = bool((a ^ value) & (a ^ result) & 0x80)
    else:
        result = a - value - (0 if cpu.flag_c else 1)
        cpu.flag_c = result >= 0
        cpu.flag_v = bool((a ^ value) & (a ^ result) & 0x80)
        result &= 0xFF
    cpu.A = (cpu.A & 0xFF00) | result
    cpu.set_nz_flags(result, False)


def sbc16(cpu: 'CPU65816', value: int):
    """Subtract with borrow (16-bit mode)."""
    a = cpu.A
    if cpu.flag_d:
        # 16-bit BCD subtraction
        borrow = 0 if cpu.flag_c else 1
        lo = (a & 0x000F) - (value & 0x000F) - borrow
        if lo < 0:
            lo = ((lo + 10) & 0x0F) - 0x10
        m1 = ((a & 0x00F0) - (value & 0x00F0) + (lo & 0xFFF0)) // 16
        if m1 < 0:
            m1 = ((m1 + 10) & 0x0F) - 0x10
        m2 = ((a & 0x0F00) - (value & 0x0F00) + (m1 << 4)) // 256
        if m2 < 0:
            m2 = ((m2 + 10) & 0x0F) - 0x10
        hi = ((a >> 12) - (value >> 12) + m2) & 0x0F
        if hi < 0:
            hi = (hi + 10) & 0x0F
        cpu.flag_c = (a - value - borrow) >= 0
        result = (hi << 12) | ((m2 & 0x0F) << 8) | ((m1 & 0x0F) << 4) | (lo & 0x0F)
        cpu.flag_v = bool((a ^ value) & (a ^ result) & 0x8000)
    else:
        result = a - value - (0 if cpu.flag_c else 1)
        cpu.flag_c = result >= 0
        cpu.flag_v = bool((a ^ value) & (a ^ result) & 0x8000)
        result &= 0xFFFF
    cpu.A = result
    cpu.set_nz_flags(result, True)


# ============== COMPARE OPERATIONS ==============

def cmp8(cpu: 'CPU65816', reg: int, value: int):
    """Compare (8-bit mode)."""
    result = (reg & 0xFF) - (value & 0xFF)
    cpu.flag_c = result >= 0
    result &= 0xFF
    cpu.set_nz_flags(result, False)


def cmp16(cpu: 'CPU65816', reg: int, value: int):
    """Compare (16-bit mode)."""
    result = reg - value
    cpu.flag_c = result >= 0
    result &= 0xFFFF
    cpu.set_nz_flags(result, True)


# ============== LOGIC OPERATIONS ==============

def and8(cpu: 'CPU65816', value: int):
    """AND with accumulator (8-bit mode)."""
    result = (cpu.A & 0xFF) & value
    cpu.A = (cpu.A & 0xFF00) | result
    cpu.set_nz_flags(result, False)


def and16(cpu: 'CPU65816', value: int):
    """AND with accumulator (16-bit mode)."""
    cpu.A = cpu.A & value
    cpu.set_nz_flags(cpu.A, True)


def ora8(cpu: 'CPU65816', value: int):
    """OR with accumulator (8-bit mode)."""
    result = (cpu.A & 0xFF) | value
    cpu.A = (cpu.A & 0xFF00) | result
    cpu.set_nz_flags(result, False)


def ora16(cpu: 'CPU65816', value: int):
    """OR with accumulator (16-bit mode)."""
    cpu.A = cpu.A | value
    cpu.set_nz_flags(cpu.A, True)


def eor8(cpu: 'CPU65816', value: int):
    """XOR with accumulator (8-bit mode)."""
    result = (cpu.A & 0xFF) ^ value
    cpu.A = (cpu.A & 0xFF00) | result
    cpu.set_nz_flags(result, False)


def eor16(cpu: 'CPU65816', value: int):
    """XOR with accumulator (16-bit mode)."""
    cpu.A = cpu.A ^ value
    cpu.set_nz_flags(cpu.A, True)


def bit8(cpu: 'CPU65816', value: int, set_nv: bool = True):
    """Test bits (8-bit mode)."""
    result = (cpu.A & 0xFF) & value
    cpu.flag_z = result == 0
    if set_nv:
        cpu.flag_n = bool(value & 0x80)
        cpu.flag_v = bool(value & 0x40)


def bit16(cpu: 'CPU65816', value: int, set_nv: bool = True):
    """Test bits (16-bit mode)."""
    result = cpu.A & value
    cpu.flag_z = result == 0
    if set_nv:
        cpu.flag_n = bool(value & 0x8000)
        cpu.flag_v = bool(value & 0x4000)


# ============== SHIFT OPERATIONS ==============

def asl8(cpu: 'CPU65816', value: int) -> int:
    """Arithmetic shift left (8-bit mode). Returns result."""
    result = value << 1
    cpu.flag_c = bool(result & 0x100)
    result &= 0xFF
    cpu.set_nz_flags(result, False)
    return result


def asl16(cpu: 'CPU65816', value: int) -> int:
    """Arithmetic shift left (16-bit mode). Returns result."""
    result = value << 1
    cpu.flag_c = bool(result & 0x10000)
    result &= 0xFFFF
    cpu.set_nz_flags(result, True)
    return result


def lsr8(cpu: 'CPU65816', value: int) -> int:
    """Logical shift right (8-bit mode). Returns result."""
    cpu.flag_c = bool(value & 0x01)
    result = value >> 1
    cpu.set_nz_flags(result, False)
    return result


def lsr16(cpu: 'CPU65816', value: int) -> int:
    """Logical shift right (16-bit mode). Returns result."""
    cpu.flag_c = bool(value & 0x01)
    result = value >> 1
    cpu.set_nz_flags(result, True)
    return result


def rol8(cpu: 'CPU65816', value: int) -> int:
    """Rotate left (8-bit mode). Returns result."""
    carry_in = 1 if cpu.flag_c else 0
    result = (value << 1) | carry_in
    cpu.flag_c = bool(result & 0x100)
    result &= 0xFF
    cpu.set_nz_flags(result, False)
    return result


def rol16(cpu: 'CPU65816', value: int) -> int:
    """Rotate left (16-bit mode). Returns result."""
    carry_in = 1 if cpu.flag_c else 0
    result = (value << 1) | carry_in
    cpu.flag_c = bool(result & 0x10000)
    result &= 0xFFFF
    cpu.set_nz_flags(result, True)
    return result


def ror8(cpu: 'CPU65816', value: int) -> int:
    """Rotate right (8-bit mode). Returns result."""
    carry_in = 0x80 if cpu.flag_c else 0
    cpu.flag_c = bool(value & 0x01)
    result = (value >> 1) | carry_in
    cpu.set_nz_flags(result, False)
    return result


def ror16(cpu: 'CPU65816', value: int) -> int:
    """Rotate right (16-bit mode). Returns result."""
    carry_in = 0x8000 if cpu.flag_c else 0
    cpu.flag_c = bool(value & 0x01)
    result = (value >> 1) | carry_in
    cpu.set_nz_flags(result, True)
    return result


# ============== INCREMENT/DECREMENT OPERATIONS ==============

def inc8(cpu: 'CPU65816', value: int) -> int:
    """Increment (8-bit mode). Returns result."""
    result = (value + 1) & 0xFF
    cpu.set_nz_flags(result, False)
    return result


def inc16(cpu: 'CPU65816', value: int) -> int:
    """Increment (16-bit mode). Returns result."""
    result = (value + 1) & 0xFFFF
    cpu.set_nz_flags(result, True)
    return result


def dec8(cpu: 'CPU65816', value: int) -> int:
    """Decrement (8-bit mode). Returns result."""
    result = (value - 1) & 0xFF
    cpu.set_nz_flags(result, False)
    return result


def dec16(cpu: 'CPU65816', value: int) -> int:
    """Decrement (16-bit mode). Returns result."""
    result = (value - 1) & 0xFFFF
    cpu.set_nz_flags(result, True)
    return result
