# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
SNES ROM bank sizes, and the check that every bank's code and data fit.

The check lays the final assembly out the way WLA-DX will
(`AddressCalculator`) and reports an overflowing bank as a compile error at
the function or static that crosses its end, naming a bank with room. Left
to the assembler, an overflow is a `MEM_INSERT` error with no R65 source -
and bank-0 bytes run into the ROM header are only a warning, linked into a
ROM with a broken header.
"""

from typing import Dict, List, Mapping, Optional, Tuple

LOROM_BANK_SIZE = 0x8000   # 32KB
HIROM_BANK_SIZE = 0x10000  # 64KB

# Where bank 0's ROM header starts: `.SNESHEADER` writes from the game code
# (`ID`, at $xFB2) through the vectors at $xFFF. The maker code before it
# is written only with MAKERCODE, which the compiler never emits.
LOROM_HEADER_START = 0x7FB2
HIROM_HEADER_START = 0xFFB2

# Where HiROM's vector trampolines go in bank 0
# (AssemblyEmitter.emit_hirom_vector_trampolines).
HIROM_TRAMPOLINE_ORG = 0xFF00


def rom_bank_size(is_hirom: bool) -> int:
    """Bytes in one ROM bank of the given mapping."""
    return HIROM_BANK_SIZE if is_hirom else LOROM_BANK_SIZE


def _bytes(n: int) -> str:
    return f"{n:,} byte{'s' if n != 1 else ''}"


def check_bank_overflow(nodes: list, mir_program, rom_banks: int, is_hirom: bool,
                        bank0_end: Optional[Tuple[int, str]] = None,
                        symbols: Optional[Mapping[str, int]] = None) -> None:
    """Raise CodegenError if any ROM bank's code and data pass its end.

    *nodes* are the banks' contents, without what codegen places at fixed
    addresses itself (the header, vector tables and HiROM trampolines).
    A bank holds `rom_bank_size` bytes; bank 0 stops at *bank0_end*, the
    address the first of those fixed parts starts at and what it is. A
    bank with an `include_asm!` file is measured without it, so it is
    reported only when what is measured already overflows, and never
    offered as room.
    """
    from r65.compiler.codegen.address_calc import AddressCalculator
    from r65.compiler.codegen.asm_nodes import Label
    from r65.compiler.codegen.constants import calculate_rom_size
    from r65.compiler.codegen.symbol_gen import auto_bank_number, rom_static_bank
    from r65.compiler.errors import CodegenError

    calc = AddressCalculator(0, symbols=symbols)
    addresses, _ = calc.calculate_with_labels(nodes)
    bank_size = rom_bank_size(is_hirom)

    def limit(bank: int) -> int:
        return bank0_end[0] if bank == 0 and bank0_end else bank_size

    over = sorted(b for b, end in calc.bank_ends.items() if end > limit(b))
    if not over:
        return
    bank = over[0]

    # Who owns each byte: the last function or ROM static label laid out in
    # the bank, so the error can point at a declaration and list the largest.
    owners: Dict[str, object] = {f.name: f for f in mir_program.functions}
    for decl in mir_program.statics:
        label = getattr(getattr(decl, 'symbol', None), 'rom_label', None)
        if label:
            owners[label] = decl
    sizes: Dict[str, int] = {}
    owner: Optional[str] = None
    owner_bank: Optional[int] = None
    crossing: Optional[str] = None
    for i, node in enumerate(nodes):
        if isinstance(node, Label) and node.name in owners:
            owner, owner_bank = node.name, calc.label_banks.get(node.name)
            continue
        if i not in addresses or calc.node_banks.get(i) != bank:
            continue
        name = owner if owner_bank == bank else None
        start, size = addresses[i]
        if name is not None:
            sizes[name] = sizes.get(name, 0) + size
        if crossing is None and start + size > limit(bank):
            crossing = name

    used = calc.bank_ends[bank]
    message = (f"ROM bank {bank} overflows by {_bytes(used - limit(bank))}: "
               f"its code and data take {used:,} of {limit(bank):,}")
    if bank == 0 and bank0_end:
        message += f" ({bank0_end[1]} at ${bank0_end[0]:04X})"

    too_big = [n for n, k in sizes.items() if k > bank_size]
    if too_big:
        n = max(too_big, key=lambda n: sizes[n])
        hints: List[str] = [f"{_display(owners[n])} is {_bytes(sizes[n])}, more than a "
                            f"bank holds ({bank_size:,}): split it into parts of at most "
                            f"{bank_size:,} bytes in banks of their own"]
    else:
        hints = [_room(bank, used - limit(bank), calc, rom_banks, bank_size,
                       is_hirom, calculate_rom_size)]
    # The compiler's own functions (`__init_start`) are not the user's to move.
    movable = [(n, k) for n, k in sizes.items() if not _display(owners[n]).startswith('fn __')]
    largest = sorted(movable, key=lambda kv: -kv[1])[:3]
    if largest:
        hints.append(f"largest in bank {bank}: " + ", ".join(
            f"{_display(owners[n])} {_bytes(k)}" for n, k in largest))
    if bank == auto_bank_number(mir_program) and _has_auto_items(mir_program, rom_static_bank):
        hints.append(f"#[bank(auto)] puts every auto item in bank {bank}: "
                     "give some an explicit #[bank(n)]")

    decl = owners.get(crossing) if crossing else None
    raise CodegenError(message, source_loc=getattr(decl, 'source_loc', None),
                       hint="; ".join(hints))


def _room(bank: int, needed: int, calc, rom_banks: int, bank_size: int,
          is_hirom: bool, calculate_rom_size) -> str:
    """Where the overflow could go: an empty bank, else the bank with the
    most room that takes it, else the next bank past the ROM's last."""
    usable = [b for b in range(1, rom_banks)
              if b != bank and b not in calc.unsized_banks]
    empty = [b for b in usable if b not in calc.bank_ends]
    move = "put a fn or ROM static there with"
    if empty:
        b = empty[0]
        return f"bank {b} is empty ({bank_size:,} bytes free): {move} #[bank({b})]"
    free = {b: bank_size - calc.bank_ends[b] for b in usable}
    fits = [b for b in free if free[b] >= needed]
    if fits:
        b = max(fits, key=lambda b: free[b])
        return f"bank {b} has {free[b]:,} bytes free: {move} #[bank({b})]"
    b = max(rom_banks, max(calc.bank_ends) + 1)
    banks, _ = calculate_rom_size(b + 1, is_hirom)
    return (f"bank {b} is free past the ROM's last bank (the ROM grows to "
            f"{banks * bank_size // 1024} KB): {move} #[bank({b})]")


def _display(decl) -> str:
    """`fn name` for a function, the static's name for its ROM data."""
    if hasattr(decl, 'blocks'):
        return f"fn {decl.name}"
    return decl.name


def _has_auto_items(mir_program, rom_static_bank) -> bool:
    if any(f.bank_attr is not None and f.bank_attr.bank_number is None
           for f in mir_program.functions):
        return True
    return any(getattr(d, 'bank_attr', None) is not None and rom_static_bank(d) is None
               for d in mir_program.statics)
