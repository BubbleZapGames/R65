"""
65816 Addressing Mode Handlers.

Each addressing mode function returns (bank, addr, extra_cycles) for the effective address,
or (value, extra_cycles) for immediate modes.
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .cpu import CPU65816


def immediate_8(cpu: 'CPU65816') -> Tuple[int, int]:
    """Immediate 8-bit: #$nn"""
    value = cpu.fetch_byte()
    return (value, 0)


def immediate_16(cpu: 'CPU65816') -> Tuple[int, int]:
    """Immediate 16-bit: #$nnnn"""
    value = cpu.fetch_word()
    return (value, 0)


def immediate_acc(cpu: 'CPU65816') -> Tuple[int, int]:
    """Immediate based on accumulator size."""
    if cpu.flag_m:
        return immediate_8(cpu)
    else:
        return immediate_16(cpu)


def immediate_idx(cpu: 'CPU65816') -> Tuple[int, int]:
    """Immediate based on index register size."""
    if cpu.flag_x:
        return immediate_8(cpu)
    else:
        return immediate_16(cpu)


def absolute(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute: $nnnn"""
    addr = cpu.fetch_word()
    return (cpu.DBR, addr, 0)


def absolute_long(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Long: $nnnnnn"""
    addr = cpu.fetch_word()
    bank = cpu.fetch_byte()
    return (bank, addr, 0)


def direct(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page: $nn"""
    offset = cpu.fetch_byte()
    addr = (cpu.D + offset) & 0xFFFF
    # Extra cycle if D is not page-aligned
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    return (0, addr, extra)


def direct_x(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page Indexed X: $nn,X"""
    offset = cpu.fetch_byte()
    x = cpu.X & cpu.idx_mask
    addr = (cpu.D + offset + x) & 0xFFFF
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    return (0, addr, extra)


def direct_y(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page Indexed Y: $nn,Y"""
    offset = cpu.fetch_byte()
    y = cpu.Y & cpu.idx_mask
    addr = (cpu.D + offset + y) & 0xFFFF
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    return (0, addr, extra)


def absolute_x(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Indexed X: $nnnn,X"""
    base = cpu.fetch_word()
    x = cpu.X & cpu.idx_mask
    addr = (base + x) & 0xFFFF
    # Extra cycle if page boundary crossed
    extra = 1 if (base & 0xFF00) != (addr & 0xFF00) else 0
    return (cpu.DBR, addr, extra)


def absolute_x_no_penalty(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Indexed X (no page crossing penalty): $nnnn,X"""
    base = cpu.fetch_word()
    x = cpu.X & cpu.idx_mask
    addr = (base + x) & 0xFFFF
    return (cpu.DBR, addr, 0)


def absolute_y(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Indexed Y: $nnnn,Y"""
    base = cpu.fetch_word()
    y = cpu.Y & cpu.idx_mask
    addr = (base + y) & 0xFFFF
    extra = 1 if (base & 0xFF00) != (addr & 0xFF00) else 0
    return (cpu.DBR, addr, extra)


def absolute_long_x(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Long Indexed X: $nnnnnn,X"""
    addr = cpu.fetch_word()
    bank = cpu.fetch_byte()
    x = cpu.X & cpu.idx_mask
    full = (bank << 16) | addr
    full = (full + x) & 0xFFFFFF
    return (full >> 16, full & 0xFFFF, 0)


def direct_indirect(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page Indirect: ($nn)"""
    offset = cpu.fetch_byte()
    ptr_addr = (cpu.D + offset) & 0xFFFF
    addr = cpu.memory.read16(ptr_addr)
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    return (cpu.DBR, addr, extra)


def direct_indirect_long(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page Indirect Long: [$nn]"""
    offset = cpu.fetch_byte()
    ptr_addr = (cpu.D + offset) & 0xFFFF
    addr = cpu.memory.read16(ptr_addr)
    bank = cpu.memory.read((ptr_addr + 2) & 0xFFFF)
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    return (bank, addr, extra)


def direct_x_indirect(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page Indexed Indirect X: ($nn,X)"""
    offset = cpu.fetch_byte()
    x = cpu.X & cpu.idx_mask
    ptr_addr = (cpu.D + offset + x) & 0xFFFF
    addr = cpu.memory.read16(ptr_addr)
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    return (cpu.DBR, addr, extra)


def direct_indirect_y(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page Indirect Indexed Y: ($nn),Y"""
    offset = cpu.fetch_byte()
    ptr_addr = (cpu.D + offset) & 0xFFFF
    base = cpu.memory.read16(ptr_addr)
    y = cpu.Y & cpu.idx_mask
    addr = (base + y) & 0xFFFF
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    if (base & 0xFF00) != (addr & 0xFF00):
        extra += 1
    return (cpu.DBR, addr, extra)


def direct_indirect_long_y(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Direct Page Indirect Long Indexed Y: [$nn],Y"""
    offset = cpu.fetch_byte()
    ptr_addr = (cpu.D + offset) & 0xFFFF
    base = cpu.memory.read16(ptr_addr)
    bank = cpu.memory.read((ptr_addr + 2) & 0xFFFF)
    y = cpu.Y & cpu.idx_mask
    full = (bank << 16) | base
    full = (full + y) & 0xFFFFFF
    extra = 1 if (cpu.D & 0xFF) != 0 else 0
    return (full >> 16, full & 0xFFFF, extra)


def stack_relative(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Stack Relative: $nn,S"""
    offset = cpu.fetch_byte()
    addr = (cpu.SP + offset) & 0xFFFF
    return (0, addr, 0)


def stack_relative_indirect_y(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Stack Relative Indirect Indexed Y: ($nn,S),Y"""
    offset = cpu.fetch_byte()
    ptr_addr = (cpu.SP + offset) & 0xFFFF
    base = cpu.memory.read16(ptr_addr)
    y = cpu.Y & cpu.idx_mask
    addr = (base + y) & 0xFFFF
    return (cpu.DBR, addr, 0)


def absolute_indirect(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Indirect: ($nnnn) - used by JMP"""
    ptr = cpu.fetch_word()
    addr = cpu.memory.read16(ptr)
    return (cpu.PBR, addr, 0)


def absolute_indirect_long(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Indirect Long: [$nnnn] - used by JMP"""
    ptr = cpu.fetch_word()
    addr = cpu.memory.read16(ptr)
    bank = cpu.memory.read((ptr + 2) & 0xFFFF)
    return (bank, addr, 0)


def absolute_indexed_indirect(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Absolute Indexed Indirect: ($nnnn,X) - used by JMP/JSR"""
    base = cpu.fetch_word()
    x = cpu.X & cpu.idx_mask
    ptr = (base + x) & 0xFFFF
    addr = cpu.memory.read16((cpu.PBR << 16) | ptr)
    return (cpu.PBR, addr, 0)


def relative_8(cpu: 'CPU65816') -> Tuple[int, int]:
    """Relative 8-bit (branch): $rr"""
    offset = cpu.fetch_byte()
    # Sign extend
    if offset & 0x80:
        offset = offset - 256
    target = (cpu.PC + offset) & 0xFFFF
    # Extra cycle if page boundary crossed
    extra = 1 if (cpu.PC & 0xFF00) != (target & 0xFF00) else 0
    return (target, extra)


def relative_16(cpu: 'CPU65816') -> Tuple[int, int]:
    """Relative 16-bit (BRL): $rrrr"""
    offset = cpu.fetch_word()
    # Sign extend
    if offset & 0x8000:
        offset = offset - 65536
    target = (cpu.PC + offset) & 0xFFFF
    return (target, 0)


def block_move(cpu: 'CPU65816') -> Tuple[int, int, int]:
    """Block move operands: #$ss,#$dd (dest, src banks)"""
    dest_bank = cpu.fetch_byte()
    src_bank = cpu.fetch_byte()
    return (src_bank, dest_bank, 0)
