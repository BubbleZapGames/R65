# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Assembly emitter: generates WLA-DX assembly text.

Provides methods for emitting instructions, labels, comments, and sections
in WLA-DX assembly format for the 65816 processor.

Builds structured AsmNode objects which are converted to assembly text
via emit_nodes().
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from r65.compiler.errors import SourceLocation

from r65.compiler import __version__
from r65.compiler.errors import InternalCompilerError
from r65.compiler.codegen.opcodes import (
    Opcode, mnemonic, addressing_mode, BRANCH_OPCODES, MEMORY_WRITE_OPCODES,
    ACCUMULATOR_OPCODES, NZ_PRESERVING_MNEMONICS, nz_register, registers_written,
    rep_sep_preserves_nz,
)
from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Comment, Directive, ModeChange, BlankLine,
    RawAsm, Immediate, Address, StackOffset, BlockMove, Operand,
)


# =============================================================================
# Assembly Formatting Constants
# =============================================================================

# Width of section dividers
DIVIDER_WIDTH = 76

# Pre-computed divider strings
SECTION_DIVIDER = "=" * DIVIDER_WIDTH
SUBSECTION_DIVIDER = "-" * DIVIDER_WIDTH

# Column alignment for comments
COMMENT_COLUMN = 32

# Import memory constants (delay import to avoid circular dependency)
def _get_rom_constants():
    from r65.compiler.codegen.constants import (
        LOROM_SLOT_SIZE, LOROM_SLOT_ADDR, HIROM_SLOT_SIZE, HIROM_SLOT_ADDR
    )
    return LOROM_SLOT_SIZE, LOROM_SLOT_ADDR, HIROM_SLOT_SIZE, HIROM_SLOT_ADDR


# =============================================================================
# Node Emission to WLA-DX Assembly
# =============================================================================

def emit_node(node: AsmNode, indent: str = "    ") -> str:
    """
    Convert a single AsmNode to WLA-DX assembly text.

    Args:
        node: The node to emit
        indent: Indentation for instructions (default 4 spaces)

    Returns:
        Assembly text for this node
    """
    match node:
        case Instruction(opcode, operand, comment):
            text = f"{indent}{_emit_instruction(opcode, operand)}"
            if comment:
                text += f"  ; {comment}"
            return text

        case Label(name):
            return f"{name}:"

        case Comment(text, section_header=True):
            line = "=" * 76
            return f"; {line}\n; {text}\n; {line}"

        case Comment(text, section_header=False):
            return f"; {text}"

        case Directive(name, args):
            if args:
                return f"{name} {', '.join(args)}"
            return name

        case ModeChange(flag, bits):
            # Indented to match the file-wide convention for mode
            # directives. Column-0 directives are reserved for
            # top-level forms (`.BANK`, `.ORG`, `.MEMORYMAP`); tests
            # and section parsers (e.g. test_jml_d_indirect_call.py)
            # use column-0 `.X` as a section boundary marker.
            return f"{indent}.{flag} {bits}"

        case BlankLine():
            return ""

        case RawAsm(text):
            # Raw assembly is emitted as-is without additional indent
            return text

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def emit_nodes(nodes: list[AsmNode], indent: str = "    ") -> str:
    """
    Convert a list of AsmNodes to WLA-DX assembly text.

    Args:
        nodes: List of nodes to emit
        indent: Indentation for instructions

    Returns:
        Complete assembly text with newlines
    """
    return '\n'.join(emit_node(n, indent) for n in nodes)


def _emit_instruction(opcode: Opcode, operand: Operand | None) -> str:
    """Format an instruction with its operand."""
    mnem = mnemonic(opcode)
    mode = addressing_mode(opcode)

    # Branch instructions: no mode suffix but take an operand
    if opcode in BRANCH_OPCODES:
        if operand is None:
            return mnem  # Shouldn't happen
        match operand:
            case Address(value):
                return f"{mnem} {_format_value(value)}"
            case _:
                return f"{mnem} {operand}"

    # Call instructions: JSR/JSL have no mode suffix but take an operand
    if opcode in (Opcode.JSR, Opcode.JSL):
        if operand is None:
            return mnem
        match operand:
            case Address(value):
                return f"{mnem} {_format_value(value)}"
            case _:
                return f"{mnem} {operand}"

    # PEI: Push Effective Indirect — WLA-DX wants the direct-page
    # operand parenthesized, `PEI ($10)`.
    if opcode == Opcode.PEI:
        match operand:
            case Immediate(value) | Address(value):
                return f"{mnem} ({_format_value(value)})"
            case None:
                return mnem
            case _:
                return f"{mnem} ({operand})"

    # PEA: Push Effective Address — takes a 16-bit absolute operand
    if opcode == Opcode.PEA:
        if operand is not None:
            match operand:
                case Immediate(value) | Address(value):
                    return f"{mnem} {_format_value(value)}"
                case _:
                    return f"{mnem} {operand}"
        return mnem

    # JMP_LONG (0x5C) is written JML, as the hand-written stubs spell it.
    if opcode == Opcode.JMP_LONG and isinstance(operand, Address):
        return f"JML {_format_long_value(operand.value)}"

    # JMP_INDIRECT_LONG (0xDC) is JML [addr] in WLA-DX. The mnemonic()
    # helper strips the suffix and yields "JMP", but WLA-DX rejects
    # `JMP [addr]` for 0xDC — it requires the explicit "JML" form.
    if opcode == Opcode.JMP_INDIRECT_LONG:
        if operand is not None:
            match operand:
                case Address(value):
                    return f"JML [{_format_value(value)}]"
                case _:
                    return f"JML {operand}"
        return "JML"

    # Accumulator mode instructions (need explicit 'A' operand in WLA-DX)
    if opcode in ACCUMULATOR_OPCODES:
        return f"{mnem} A"

    # Block move instructions (MVN/MVP) - special operand format
    if isinstance(operand, BlockMove):
        return f"{mnem} {BlockMove._fmt(operand.src_bank)}, {BlockMove._fmt(operand.dst_bank)}"

    # Implied addressing (no operand)
    if mode is None:
        return mnem

    # Format operand based on addressing mode
    if operand is None:
        # Shouldn't happen for non-implied, but handle gracefully
        return mnem

    match (mode, operand):
        # Immediate
        case ("IMMEDIATE", Immediate(value)):
            return f"{mnem} #{_format_value(value)}"
        case ("IMMEDIATE", Address(value)):
            # Immediate with address label (e.g., LDX #label)
            return f"{mnem} #{_format_value(value)}"

        # Direct Page
        case ("DP", Address(value)):
            return f"{mnem} {_format_value(value)}"
        case ("DP_X", Address(value)):
            return f"{mnem} {_format_value(value)},X"
        case ("DP_Y", Address(value)):
            return f"{mnem} {_format_value(value)},Y"

        # Absolute and long. WLA-DX picks the smallest form an operand's value
        # fits, whatever its digits: `STA $0010` and `STA $000010` both
        # assemble as direct page, which under D = S writes the stack frame.
        # A label, or a value that would shrink, gets the size suffix.
        case ("ABSOLUTE", Address(value)):
            return f"{mnem}{_size_suffix(value, '.w', 0x100)} {_format_absolute(value)}"
        case ("ABSOLUTE_X", Address(value)):
            return f"{mnem}{_size_suffix(value, '.w', 0x100)} {_format_absolute(value)},X"
        case ("ABSOLUTE_Y", Address(value)):
            return f"{mnem}{_size_suffix(value, '.w', 0x100)} {_format_absolute(value)},Y"
        case ("LONG", Address(value)):
            return f"{mnem}{_size_suffix(value, '.l', 0x10000)} {_format_long_value(value)}"
        case ("LONG_X", Address(value)):
            return f"{mnem}{_size_suffix(value, '.l', 0x10000)} {_format_long_value(value)},X"

        # Indirect
        case ("INDIRECT", Address(value)):
            return f"{mnem} ({_format_value(value)})"
        case ("INDIRECT_X", Address(value)):
            return f"{mnem} ({_format_value(value)},X)"
        case ("INDIRECT_LONG", Address(value)):
            return f"{mnem} [{_format_value(value)}]"

        # Direct Page Indirect
        case ("DP_INDIRECT", Address(value)):
            return f"{mnem} ({_format_value(value)})"
        case ("DP_INDIRECT_X", Address(value)):
            return f"{mnem} ({_format_value(value)},X)"
        case ("DP_INDIRECT_Y", Address(value)):
            return f"{mnem} ({_format_value(value)}),Y"
        case ("DP_INDIRECT_LONG", Address(value)):
            return f"{mnem} [{_format_value(value)}]"
        case ("DP_INDIRECT_LONG_Y", Address(value)):
            return f"{mnem} [{_format_value(value)}],Y"

        # Stack Relative
        case ("STACK", StackOffset(offset)):
            return f"{mnem} {_format_value(offset)},S"
        case ("STACK_INDIRECT_Y", StackOffset(offset)):
            return f"{mnem} ({_format_value(offset)},S),Y"

        # Block Move
        case (_, BlockMove() as bm):
            return f"{mnem} {BlockMove._fmt(bm.src_bank)}, {BlockMove._fmt(bm.dst_bank)}"

        case _:
            # A raw operand string, printed verbatim. A bare int never gets
            # here: emit_instr refuses it rather than print it as an address.
            return f"{mnem} {operand}"


def _format_value(value: int | str) -> str:
    """Format an immediate or address value."""
    if isinstance(value, str):
        return value
    if value < 0x100:
        return f"${value:02X}"
    return f"${value:04X}"


def _size_suffix(value: int | str, suffix: str, shrinks_below: int) -> str:
    """The WLA-DX size suffix an operand needs to keep its addressing mode."""
    return suffix if isinstance(value, str) or value < shrinks_below else ""


def _format_absolute(value: int | str) -> str:
    """Format an absolute address value."""
    if isinstance(value, str):
        return value
    return f"${value:04X}"


def _format_long_value(value: int | str) -> str:
    """Format a 24-bit long address value."""
    if isinstance(value, str):
        return value
    return f"${value:06X}"


def _snapshot(obj) -> dict:
    """*obj*'s attributes, with its mutable containers copied."""
    names = (vars(obj) if hasattr(obj, '__dict__')
             else {n: None for n in type(obj).__slots__})
    return {n: (v.copy() if isinstance(v, (list, set, dict)) else v)
            for n in names if n != 'nodes' and hasattr(obj, n)
            for v in (getattr(obj, n),)}


def _restore(obj, snapshot: dict):
    if hasattr(obj, '__dict__'):
        for n in [n for n in vars(obj) if n != 'nodes' and n not in snapshot]:
            delattr(obj, n)
    for n, v in snapshot.items():
        setattr(obj, n, v)


class Capture:
    """The nodes emitted so far inside an `AssemblyEmitter.capture()` block."""

    def __init__(self, emitter: 'AssemblyEmitter', start: int):
        self._emitter = emitter
        self._start = start

    @property
    def nodes(self) -> list:
        return self._emitter.nodes[self._start:]


# =============================================================================
# Assembly Emitter Class
# =============================================================================

class AssemblyEmitter:
    """
    Emits WLA-DX assembly code.

    Builds structured AsmNode objects and provides high-level methods
    for generating properly formatted assembly sections and instructions.

    Call get_nodes() to retrieve the structured representation, or
    to_string() to get the final assembly text.
    """

    def __init__(self, source_file: Optional[str] = None):
        """
        Initialize assembly emitter.

        Args:
            source_file: Original R65 source file path (for header)
        """
        self.source_file = source_file or "unknown.r65"
        self.nodes: List[AsmNode] = []
        # Every .DEFINE / .EQU value emitted: WLA-DX sizes an `asm!` operand
        # naming one by its value (see asm_size.asm_line_size).
        self.asm_symbols: Dict[str, int] = {}
        # Track current processor modes for optimization
        self._current_accu_mode = 8  # Default m8 mode
        self._current_index_mode = 16  # Default x16 mode
        # Track which register (if any) the N/Z flags currently reflect, at
        # the width that register had when they were set: a REP/SEP that
        # changes that width (M for A, X for X/Y) invalidates it. Used to
        # elide redundant CMP/CPX/CPY #0 when flags are already valid.
        self.nz_valid_for: str | None = None  # 'A', 'X', 'Y', or None
        # True while X and Y are known to hold the same value — set by
        # TYX/TXY, cleared by anything that writes X or Y and at labels.
        # Lets a repeated TYX/TXY be skipped (see _update_xy_copy_tracking).
        self.x_equals_y: bool = False
        # What A and X hold, as `(stack slot, left shift)`, or None. A slot is
        # named by its offset minus the bytes pushed since the stack pointer was
        # last set (`stack_depth`), so it stays the same slot as S moves. Lets
        # an index reload that would give X the value it already has be
        # skipped (see _update_slot_tracking).
        self.a_slot = None
        self.x_slot = None
        self.stack_depth: int = 0
        # The one record of how far S has moved inside a function body:
        # `stack_depth` counts every push and pull emitted, `frame_base` is
        # its value once the prologue has run, and `note_stack` adds what no
        # opcode shows (a callee removing its arguments, a trampoline's
        # return address taken by an RTS). Every stack-relative offset is
        # frame-relative plus `displacement` (see stack_top/stack_operand on
        # the instruction selector).
        self.frame_base: int = 0
        self.peak_displacement: int = 0
        # Track banks that have had data/code emitted to avoid .ORG 0 overlap
        self._used_banks: set = set()
        # Base bank byte for ROM labels, emitted as `.BASE $xx` before each
        # `.BANK`. WLA-DX resolves a label's bank as `.BASE + its ROM bank`
        # (doc/asmdiv.rst), so this is what puts `:label`, `#:label`, `JSL`,
        # `LDA.l` and `MVN` bank operands into the right 65816 bank:
        #   $00 - LoROM SlowROM (banks $00-$3F); the default
        #   $80 - LoROM FastROM (banks $80-$BF); MEMSEL only speeds up $80-$FF
        #   $C0 - HiROM (banks $C0-$FF); already the FastROM-capable region
        self.rom_base: int = 0x00
        # True when #[snesrom(..., fastrom)] was given. Drives the MEMSEL
        # write, the vector trampolines, and the data bank below.
        self.fastrom: bool = False
        # The data bank: what DBR holds outside a far-pointer or databank
        # bracket, so absolute operands reach the hardware registers and low
        # RAM. $00, or $80 under FastROM (the mirror MEMSEL speeds up); both
        # map I/O and low RAM, which the HiROM code banks $C0-$FF do not.
        self.data_bank: int = 0x00
        # The 65816 bank of the code being emitted (`.BASE` + `.BANK`).
        self.code_bank: int = 0x00

    # ========================================================================
    # Trial Emission
    # ========================================================================

    @contextmanager
    def capture(self, *trackers):
        """Emit on trial: inside the block code is emitted as usual, and on
        exit it is gone as if it never was.

        Yields a `Capture` whose `nodes` are what the block emitted. On exit
        the node list is cut back to where it was, which takes the labels,
        mode directives and source locations (the debug info) with it, and
        every tracked attribute of the emitter - accumulator and index mode,
        N/Z validity, X = Y, the A/X stack slots and stack depth, used banks -
        is put back. *trackers* are other objects whose state the emission
        moves (XBA state, the stack displacement tracker); their attributes
        are put back too.
        """
        start = len(self.nodes)
        saved = [(obj, _snapshot(obj)) for obj in (self, *trackers)]
        try:
            yield Capture(self, start)
        finally:
            del self.nodes[start:]
            for obj, snapshot in saved:
                _restore(obj, snapshot)

    # ========================================================================
    # Low-Level Node Emission
    # ========================================================================

    def emit(self, node: AsmNode):
        """Emit a single node."""
        self.nodes.append(node)
        if isinstance(node, (Instruction, RawAsm)):
            self.x_equals_y = False  # untracked code: assume X/Y changed
            self.forget_slots()

    def emit_instr(self, opcode: Opcode, operand: Operand | None = None,
                   comment: str | None = None, source_loc: 'SourceLocation | None' = None):
        """
        Emit an instruction using structured Opcode enum.

        Args:
            opcode: The Opcode enum value (e.g., Opcode.LDA_IMMEDIATE)
            operand: The operand (Immediate, Address, StackOffset, etc.)
            comment: Optional comment
            source_loc: Optional source location for debug info

        A bare int is not an operand: it would print as an address, so an
        intended immediate would become a memory read. Wrap it in Immediate
        or Address. A 24-bit address needs a LONG opcode: WLA-DX would
        assemble it long anyway, a byte longer than OPCODE_SIZES counts.
        """
        if isinstance(operand, int):
            raise InternalCompilerError(
                f"{opcode.name}: bare int operand {operand!r}; "
                f"pass Immediate(...) or Address(...)", source_loc)
        if (isinstance(operand, Address) and isinstance(operand.value, int)
                and operand.value > 0xFFFF
                and addressing_mode(opcode) in ('ABSOLUTE', 'ABSOLUTE_X', 'ABSOLUTE_Y')):
            raise InternalCompilerError(
                f"{opcode.name}: 24-bit address ${operand.value:06X} in a 16-bit "
                f"absolute opcode; select the LONG form", source_loc)
        self.nodes.append(Instruction(opcode, operand, comment, source_loc))
        self._update_nz_tracking(opcode, operand)
        self._update_xy_copy_tracking(opcode, operand)
        self._update_slot_tracking(opcode, operand)

    def emit_raw(self, text: str, source_loc: 'SourceLocation | None' = None,
                 user_asm: bool = False):
        """Emit raw assembly text (*user_asm*: a line of `asm!`)."""
        self.nodes.append(RawAsm(text, source_loc, user_asm=user_asm))
        self.nz_valid_for = None  # Unknown effect on flags
        self.x_equals_y = False   # ...and on registers
        self.forget_slots()

    def emit_directive(self, text: str):
        """
        Emit a WLA-DX directive (e.g., .65816, .MEMORYMAP).

        Args:
            text: Directive text
        """
        self.nodes.append(RawAsm(text))

    def emit_blank_line(self):
        """Emit a blank line for spacing."""
        self.nodes.append(BlankLine())

    def emit_comment(self, text: str):
        """
        Emit a comment line.

        Args:
            text: Comment text (without ; prefix)
        """
        self.nodes.append(Comment(text))

    def emit_accu_mode(self, bits: int):
        """Emit a `.ACCU` mode-tracking directive and update tracked mode."""
        self.nodes.append(ModeChange("ACCU", bits))
        self._current_accu_mode = bits

    def emit_index_mode(self, bits: int):
        """Emit a `.INDEX` mode-tracking directive and update tracked mode."""
        self.nodes.append(ModeChange("INDEX", bits))
        self._current_index_mode = bits

    def set_accu_mode_tracking(self, bits: 'int | None'):
        """Update tracked accumulator mode without emitting .ACCU directive.

        Used when inline asm changes mode via REP/SEP - WLA-DX auto-tracks
        those instructions so no .ACCU directive is needed, but the compiler's
        internal tracker must stay in sync.

        Args:
            bits: 8 for m8, 16 for m16, None if mode is unknown
        """
        self._current_accu_mode = bits

    def get_accu_mode(self) -> 'int | None':
        """Get current tracked accumulator mode (8, 16, or None if unknown)."""
        return self._current_accu_mode

    def get_index_mode(self) -> int:
        """Get current tracked index mode (8 or 16)."""
        return self._current_index_mode

    # ========================================================================
    # Section Headers
    # ========================================================================

    def emit_section_header(self, title: str):
        """
        Emit a major section header with divider.

        Example:
            ; ============================================================================
            ; Direct Page Allocations
            ; ============================================================================

        Args:
            title: Section title
        """
        self.emit_comment(SECTION_DIVIDER)
        self.emit_comment(title)
        self.emit_comment(SECTION_DIVIDER)

    def emit_subsection_header(self, title: str):
        """
        Emit a subsection header with dashes.

        Example:
            ; ----------------------------------------------------------------------------
            ; Helper Functions
            ; ----------------------------------------------------------------------------

        Args:
            title: Subsection title
        """
        self.emit_comment(SUBSECTION_DIVIDER)
        self.emit_comment(title)
        self.emit_comment(SUBSECTION_DIVIDER)

    # ========================================================================
    # File Header
    # ========================================================================

    def emit_file_header(self):
        """
        Emit file header with metadata.

        Generated:
            ; ============================================================================
            ; Generated by R65 Compiler
            ; Source: main.r65
            ; Generated: 2026-01-01 14:30:00
            ; Compiler Version: 0.3.0
            ; ============================================================================
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.emit_section_header("Generated by R65 Compiler")
        self.emit_comment(f"Source: {self.source_file}")
        self.emit_comment(f"Generated: {now}")
        self.emit_comment(f"Compiler Version: {__version__}")
        self.emit_comment(SECTION_DIVIDER)
        self.emit_blank_line()

    # ========================================================================
    # Processor Directives
    # ========================================================================

    def emit_processor_directive(self):
        """
        Emit processor directive for 65816.

        Generated:
            .65816
        """
        self.emit_directive(".65816")
        self.emit_blank_line()

    # ========================================================================
    # Memory Map
    # ========================================================================

    def emit_memory_map(self, rom_type: str = "lorom", banks: int = 1,
                        has_wram_7f: bool = False, fastrom: bool = False):
        """
        Emit WLA-DX memory map and ROM bank map.

        Args:
            rom_type: "lorom" (32KB banks) or "hirom" (64KB banks)
            banks: Number of ROM banks
            fastrom: True when #[snesrom(..., fastrom)] was given. For LoROM
                this moves ROM labels into the $80-$BF mirror, the only region
                MEMSEL ($420D) actually speeds up. HiROM already sits at $C0+.

        Generated (LoROM; SLOT 2 only with has_wram_7f):
            .MEMORYMAP
                DEFAULTSLOT 0
                SLOTSIZE $8000
                SLOT 0 $8000
                SLOT 1 $2000 SIZE $E000
                SLOT 2 $0000 SIZE $10000
            .ENDME

            .ROMBANKMAP
                BANKSTOTAL 1
                BANKSIZE $8000
                BANKS 1
            .ENDRO
        """
        self.emit_section_header(f"Memory Map ({rom_type.upper()})")

        lorom_size, lorom_addr, hirom_size, hirom_addr = _get_rom_constants()
        if rom_type.lower() == "lorom":
            slot_size = lorom_size
            slot_addr = lorom_addr
        else:  # hirom
            slot_size = hirom_size
            slot_addr = hirom_addr

        # Memory map
        self.emit_directive(".MEMORYMAP")
        self.emit_directive("    DEFAULTSLOT 0")
        self.emit_directive(f"    SLOTSIZE ${slot_size:04X}")
        self.emit_directive(f"    SLOT 0 ${slot_addr:04X}")
        self.emit_directive("    SLOT 1 $2000 SIZE $E000  ; WRAM bank $7E ($2000-$FFFF)")
        if has_wram_7f:
            self.emit_directive("    SLOT 2 $0000 SIZE $10000 ; WRAM bank $7F ($0000-$FFFF)")
        self.emit_directive(".ENDME")
        self.emit_blank_line()

        # ROM bank map
        self.emit_directive(".ROMBANKMAP")
        self.emit_directive(f"    BANKSTOTAL {banks}")
        self.emit_directive(f"    BANKSIZE ${slot_size:04X}")
        self.emit_directive(f"    BANKS {banks}")
        self.emit_directive(".ENDRO")
        self.emit_blank_line()

        # Pick the ROM base that emit_bank_directive and emit_ramsection
        # restore with .BASE before each section.
        if rom_type.lower() != "lorom":
            # $C0-$FF is both the fully-ROM-mapped and the FastROM-capable
            # HiROM region, so HiROM needs no separate fastrom base.
            self.rom_base = 0xC0
        else:
            self.rom_base = 0x80 if fastrom else 0x00
        self.fastrom = fastrom
        self.data_bank = 0x80 if fastrom else 0x00
        self.code_bank = self.rom_base

    def data_bank_load(self) -> list:
        """(opcode, operand) pairs that set DBR to the data bank without
        touching a register: PHK/PLB when the code runs in the data bank,
        else PEA of the bank twice and two PLBs."""
        if self.code_bank == self.data_bank:
            return [(Opcode.PHK, None), (Opcode.PLB, None)]
        bank = self.data_bank
        return [(Opcode.PEA, Immediate((bank << 8) | bank)),
                (Opcode.PLB, None), (Opcode.PLB, None)]

    def emit_data_bank_load(self, comment: str,
                            source_loc: 'SourceLocation | None' = None):
        """Emit data_bank_load(), with `comment` on its final PLB."""
        ops = self.data_bank_load()
        for i, (opcode, operand) in enumerate(ops):
            self.emit_instr(opcode, operand,
                            comment if i == len(ops) - 1 else None, source_loc)

    # ========================================================================
    # Bank Directives
    # ========================================================================

    def emit_bank_directive(self, bank_num: int, slot: int = 0):
        """
        Emit bank directive and origin.

        Emits `.BASE $<rom_base>` first when a non-zero base is in effect, so
        `:label` / `#:label` bank bytes resolve into the right 65816 bank:
        $C0+ for HiROM (where $0000-$FFFF all map to ROM) and $80+ for LoROM
        FastROM. A zero base emits nothing.

        Args:
            bank_num: Bank number
            slot: Slot number (usually 0)
        """
        if self.rom_base:
            self.emit_directive(f".BASE ${self.rom_base:02X}")
        self.emit_directive(f".BANK {bank_num} SLOT {slot}")
        self.code_bank = self.rom_base + bank_num
        if bank_num not in self._used_banks:
            self.emit_directive(".ORG 0")
            self._used_banks.add(bank_num)
        self.emit_blank_line()

    # ========================================================================
    # RAM Sections
    # ========================================================================

    def emit_ramsection(self, section_name: str, bank: int, slot: int,
                        orga: int, entries: list):
        """
        Emit a .RAMSECTION block for RAM variable allocation.

        WLA-DX .RAMSECTION creates proper labels with correct bank metadata,
        making the #: (bank byte) operator return the correct value.

        Args:
            section_name: Section name (e.g., "ram.data")
            bank: Bank number (e.g., 0x7E)
            slot: Slot number (e.g., 1)
            orga: Origin address within the slot (e.g., 0x2000)
            entries: List of (name, size, comment) tuples
        """
        # Set .BASE to the target bank so #: resolves correctly.
        # RAMSECTION uses BANK $00 since .BASE provides the bank offset.
        self.emit_directive(f".BASE ${bank:02X}")
        self.emit_directive(
            f'.RAMSECTION "{section_name}" BANK $00 SLOT {slot} FORCE ORGA ${orga:04X}'
        )
        for name, size, comment in entries:
            line = f"    {name} dsb {size}"
            if comment:
                padding = max(1, COMMENT_COLUMN - len(line))
                line += " " * padding + f"; {comment}"
            self.emit_directive(line)
        self.emit_directive(".ENDS")
        # Restore the ROM base so subsequent ROM bank labels get correct bank
        # bytes. Without this, .BASE $7E persists and offsets all later :label
        # calculations (e.g., BANK 1 would resolve to $7F instead of $01).
        self.emit_directive(f".BASE ${self.rom_base:02X}")

    # ========================================================================
    # Labels
    # ========================================================================

    def emit_label(self, label: str, source_loc: 'SourceLocation | None' = None):
        """
        Emit a label.

        Args:
            label: Label name (with or without trailing :)
            source_loc: Optional source location for debug info

        Generated:
            label:
        """
        # Strip trailing colon (emit_node adds it back)
        label_name = label.rstrip(':')
        self.nodes.append(Label(label_name, source_loc))
        self.nz_valid_for = None  # Labels are merge points; flags ambiguous
        self.x_equals_y = False   # re-established by the caller when every
                                  # predecessor agrees (function_gen block loop)
        self.a_slot = None
        self.x_slot = None


    # ========================================================================
    # X/Y copy, stack-slot and N/Z tracking
    # ========================================================================

    # Beyond what `registers_written` lists: control leaving for code that
    # may write anything.
    _MAY_WRITE_XY = frozenset({'JSR', 'JSL', 'RTS', 'RTL', 'RTI', 'BRK', 'COP'})

    def _update_xy_copy_tracking(self, opcode: Opcode, operand):
        """Track whether X and Y hold the same value after *opcode*.

        TYX/TXY make them equal; any write to either register ends it (a
        width change to x8 included). Used to skip a TYX/TXY that would copy
        a value X already has.
        """
        if opcode in (Opcode.TYX, Opcode.TXY):
            self.x_equals_y = True
            return
        if (registers_written(opcode, operand) & {'X', 'Y'}
                or mnemonic(opcode) in self._MAY_WRITE_XY):
            self.x_equals_y = False

    # ------------------------------------------------------------------
    # Stack-slot contents of A and X
    # ------------------------------------------------------------------

    # Bytes each push/pull moves: an int, None for the accumulator width,
    # 'x' for the index width.
    _PUSH_BYTES = {'PHA': None, 'PHX': 'x', 'PHY': 'x', 'PHP': 1, 'PHB': 1,
                   'PHK': 1, 'PHD': 2, 'PEA': 2, 'PEI': 2, 'PER': 2}
    _PULL_BYTES = {'PLA': None, 'PLX': 'x', 'PLY': 'x', 'PLP': 1, 'PLB': 1,
                   'PLD': 2}

    @property
    def displacement(self) -> int:
        """Bytes S has moved down since the frame was set up."""
        return self.stack_depth - self.frame_base

    def start_body(self):
        """The prologue is done: S is at the frame base from here."""
        self.frame_base = self.stack_depth
        self.peak_displacement = 0

    def note_stack(self, delta: int):
        """S moved by *delta* bytes (positive: down) with no push or pull
        to show it."""
        self.stack_depth += delta
        self.peak_displacement = max(self.peak_displacement, self.displacement)

    def forget_slots(self):
        """Nothing is known about what A or X hold."""
        self.a_slot = None
        self.x_slot = None

    def slot_key(self, offset: int) -> int:
        """The stable name of the stack slot at `offset,S` right now."""
        return offset - self.stack_depth

    def _width(self, kind) -> int:
        if kind == 'x':
            return 1 if self._current_index_mode == 8 else 2
        if kind is None:
            return 1 if self._current_accu_mode == 8 else 2
        return kind

    def _update_slot_tracking(self, opcode: Opcode, operand):
        """Track which stack slot (shifted left k times) A and X hold.

        `LDA off,S` in 16-bit mode makes A the slot's value; `ASL A` in
        16-bit mode shifts it; `TAX` hands it to X. Any other write to A or X
        forgets it. A store that may reach the slot forgets both: one to the
        slot itself, or any store through direct page or a pointer (a D=S
        function addresses its stack slots through direct page). Absolute and
        long stores name statics, which never live on the stack.
        """
        mn = mnemonic(opcode)
        mode = addressing_mode(opcode)

        # Stack depth: pushes and pulls move S under the slot offsets.
        if mn in self._PUSH_BYTES:
            self.stack_depth += self._width(self._PUSH_BYTES[mn])
            self.peak_displacement = max(self.peak_displacement, self.displacement)
        elif mn in self._PULL_BYTES:
            self.stack_depth -= self._width(self._PULL_BYTES[mn])
        if mn in ('TCS', 'TXS'):
            # S set outright. Where by how much is known (frame allocation,
            # dropping pushed arguments) the emitting code notes it
            # (note_stack); the slots are forgotten either way.
            self.forget_slots()
            return
        if mn in ('JSR', 'JSL', 'RTS', 'RTL', 'RTI', 'BRK', 'COP', 'MVN', 'MVP',
                  'XCE', 'PLP', 'TCD', 'PLD', 'WDM'):
            self.forget_slots()
            return

        # Stores that may reach a tracked slot.
        if opcode in MEMORY_WRITE_OPCODES:
            if mode == 'STACK':
                key = self.slot_key(operand.offset)
                for attr in ('a_slot', 'x_slot'):
                    held = getattr(self, attr)
                    if held is not None and abs(held[0] - key) <= 1:
                        setattr(self, attr, None)
            elif not mode.startswith(('ABSOLUTE', 'LONG')):
                self.forget_slots()

        # A.
        if opcode == Opcode.LDA_STACK and self._current_accu_mode == 16:
            self.a_slot = (self.slot_key(operand.offset), 0)
        elif opcode == Opcode.ASL and self._current_accu_mode == 16 and self.a_slot:
            self.a_slot = (self.a_slot[0], self.a_slot[1] + 1)
        elif opcode == Opcode.TAX:
            # x16 TAX copies all of C; only a 16-bit load filled it.
            self.x_slot = self.a_slot if self._current_index_mode == 16 else None
        elif 'A' in registers_written(opcode):
            self.a_slot = None

        # X written by anything but TAX (x8 truncating it included).
        if (('X' in registers_written(opcode, operand) and opcode != Opcode.TAX)
                or mn in self._MAY_WRITE_XY):
            self.x_slot = None

    def _update_nz_tracking(self, opcode: Opcode, operand: Operand | None = None):
        """
        Update N/Z flag tracking after emitting an instruction.

        Tracks which register (A, X, or Y) the processor N/Z flags currently
        reflect, at that register's width when they were set. Used to elide
        redundant CMP/CPX/CPY #0 instructions when the flags are already
        valid from a prior load, transfer, or ALU op.
        """
        mn = mnemonic(opcode)
        if mn in ('REP', 'SEP'):
            mask = operand.value if isinstance(operand, Immediate) else None
            if (self.nz_valid_for is not None
                    and not rep_sep_preserves_nz(mask, self.nz_valid_for)):
                self.nz_valid_for = None
            return
        reg = nz_register(opcode, {16: True, 8: False}.get(self._current_accu_mode))
        if reg is not None:
            self.nz_valid_for = reg
        elif mn not in NZ_PRESERVING_MNEMONICS:
            # CMP/CPX/CPY, BIT, PLP, a call, a memory read-modify-write...
            self.nz_valid_for = None

    # ========================================================================
    # Symbol Definitions
    # ========================================================================

    def emit_define(self, name: str, value: int, comment: Optional[str] = None):
        """
        Emit a .DEFINE directive.

        Args:
            name: Symbol name
            value: Address value
            comment: Optional inline comment

        Generated:
            .DEFINE TEMP $0020      ; main.r65:5
        """
        self.asm_symbols[name] = value
        line = f".DEFINE {name} ${value:04X}"

        if comment:
            padding = max(1, COMMENT_COLUMN - len(line))
            line += " " * padding + f"; {comment}"

        self.emit_directive(line)

    def emit_equ(self, name: str, value: int, comment: Optional[str] = None):
        """
        Emit a .EQU directive for constants.

        Args:
            name: Constant name
            value: Constant value
            comment: Optional inline comment

        Generated:
            .EQU SCREEN_WIDTH 256   ; Constant

        A bool value is written as 1/0: WLA-DX has no True/False.
        """
        self.asm_symbols[name] = int(value)
        line = f".EQU {name} {int(value)}"

        if comment:
            padding = max(1, COMMENT_COLUMN - len(line))
            line += " " * padding + f"; {comment}"

        self.emit_directive(line)

    # ========================================================================
    # Data Directives
    # ========================================================================

    def emit_byte(self, values: List[int]):
        """
        Emit byte data (.DB directive).

        Args:
            values: List of byte values

        Generated:
            .DB $80, $83, $86, $89
        """
        # Format as hex bytes
        hex_values = [f"${v:02X}" for v in values]

        # Emit 8 values per line for readability
        for i in range(0, len(hex_values), 8):
            chunk = hex_values[i:i+8]
            self.emit_directive(f"    .DB {', '.join(chunk)}")

    def emit_word(self, values: List[int]):
        """
        Emit word data (.DW directive).

        Args:
            values: List of word values

        Generated:
            .DW $1234, $5678
        """
        hex_values = [f"${v:04X}" for v in values]

        # Emit 8 values per line
        for i in range(0, len(hex_values), 8):
            chunk = hex_values[i:i+8]
            self.emit_directive(f"    .DW {', '.join(chunk)}")

    def emit_space(self, size: int, fill_value: int = 0):
        """
        Emit space reservation (.DSB directive).

        Args:
            size: Number of bytes to reserve
            fill_value: Fill value (default 0)

        Generated:
            .DSB 256, 0
        """
        self.emit_directive(f"    .DSB {size}, {fill_value}")

    def emit_incbin(self, filepath: str, size: int, label: Optional[str] = None):
        """
        Emit binary file inclusion (.INCBIN directive).

        Args:
            filepath: Path to binary file to include, as the assembler finds it
            size: The file's size, which the node carries for layout
            label: Optional label for the data

        Generated:
            LABEL:
            .INCBIN "path/to/file.bin"
        """
        if label:
            self.emit_label(label)
        self.nodes.append(RawAsm(f'    .INCBIN "{filepath}"', size=size))

    # ========================================================================
    # SNES Header and Interrupt Vectors
    # ========================================================================

    def emit_snes_header(self, snesrom_config=None, romsize_value: int = 0x08):
        """
        Emit SNES ROM header using .SNESHEADER directive.

        Args:
            snesrom_config: the #[snesrom(...)] ast.SnesRomDirective, or None for defaults
            romsize_value: SNES header ROMSIZE value ($08=256KB, $09=512KB, etc.)

        Generated:
            .SNESHEADER
              ID "SNES"
              NAME "ROM NAME          "
              LOROM
              CARTRIDGETYPE $00
              ROMSIZE $08
              SRAMSIZE $00
              COUNTRY $01
              LICENSEECODE $00
              VERSION $00
            .ENDSNES

        Vectors are emitted separately by emit_interrupt_vectors
        (.SNESNATIVEVECTOR / .SNESEMUVECTOR).
        """
        self.emit_section_header("SNES ROM Header")

        # Use defaults if no config provided
        if snesrom_config is None:
            rom_name = "R65 ROM"
            rom_id = "SNES"
            cartridge_type = 0x00
            sram_size = 0x00
            country = 0x01
            version = 0x00
            lorom = True
            hirom = False
            exhirom = False
            slowrom = True
            fastrom = False
        else:
            rom_name = snesrom_config.name
            rom_id = snesrom_config.id
            cartridge_type = snesrom_config.cartridge_type
            sram_size = snesrom_config.sram_size
            country = snesrom_config.country
            version = snesrom_config.version
            lorom = snesrom_config.lorom
            hirom = snesrom_config.hirom
            exhirom = snesrom_config.exhirom
            slowrom = snesrom_config.slowrom
            fastrom = snesrom_config.fastrom

        # Pad or truncate ROM name to exactly 21 characters
        padded_name = rom_name[:21].ljust(21)

        self.emit_directive(".SNESHEADER")
        self.emit_directive(f'ID "{rom_id}"')
        self.emit_directive(f'NAME "{padded_name}"')

        # Memory mapping type (mutually exclusive)
        if hirom:
            self.emit_directive("HIROM")
        elif exhirom:
            self.emit_directive("EXHIROM")
        else:
            self.emit_directive("LOROM")

        # ROM speed
        if fastrom:
            self.emit_directive("FASTROM")
        else:
            self.emit_directive("SLOWROM")

        self.emit_directive(f"CARTRIDGETYPE ${cartridge_type:02X}")
        self.emit_directive(f"ROMSIZE ${romsize_value:02X}")
        self.emit_directive(f"SRAMSIZE ${sram_size:02X}")
        self.emit_directive(f"COUNTRY ${country:02X}")
        self.emit_directive("LICENSEECODE $00")
        self.emit_directive(f"VERSION ${version:02X}")
        self.emit_directive(".ENDSNES")
        self.emit_blank_line()

    def emit_empty_interrupt_handler(self):
        """
        Emit empty interrupt handler for unused vectors.

        Generated:
            __empty_handler:
                RTI
        """
        self.emit_label("__empty_handler")
        self.emit_instr(Opcode.RTI)
        self.emit_blank_line()

    def emit_lorom_fast_vector_trampolines(self, reset=None, **handlers):
        """
        Emit LoROM FastROM vector trampolines.

        MEMSEL ($420D) only speeds up banks $80-$FF, but the CPU always fetches
        interrupt vectors with PBR=$00, so entry code runs at SlowROM speed
        until something reloads PBR. These stubs sit in bank $00 (where the
        16-bit vector operand can reach them) and JML into the $80-$BF mirror,
        whose bank byte comes from the `.BASE $80` in effect for ROM banks.

        Emitted inside the bank-0 code window, so no .ORG and no reserved
        space is needed - any bank-0 address works.

        `handlers` maps a vector name (nmi, irq, brk, cop, abort) to its
        handler label; each gets a `__fast_<vector>` stub.

        Generated:
            __fast_reset: SEI / CLC / XCE / SEP #$20 / LDA #$01 / STA $420D
                          JML main
            __fast_nmi:   JML nmi_handler
            __fast_irq:   JML irq_handler
        """
        self.emit_section_header("FastROM Vector Trampolines")

        if reset:
            self.emit_label("__fast_reset")
            self.emit_instr(Opcode.SEI)
            self.emit_instr(Opcode.CLC)
            self.emit_instr(Opcode.XCE)
            self.emit_memsel_write()
            self.emit_raw(f"    JML {reset}")

        for vector, handler in handlers.items():
            if handler:
                self.emit_label(f"__fast_{vector}")
                self.emit_raw(f"    JML {handler}")

        self.emit_blank_line()

    def emit_memsel_write(self):
        """
        Emit the FastROM enable: MEMSEL ($420D) bit 0 = 1.

        Forces m8 first because a reset stub runs before any mode is
        established, and MEMSEL is an 8-bit register.
        """
        self.emit_raw("    SEP #$20")
        self.emit_accu_mode(8)
        self.emit_raw("    LDA #$01")
        self.emit_raw("    STA $420D")

    def emit_hirom_vector_trampolines(self, reset=None, fastrom: bool = False,
                                      **handlers):
        """
        Emit HiROM interrupt vector trampolines at addresses >= $8000.

        In HiROM, the CPU reads interrupt vectors from bank $00 where only
        $8000-$FFFF maps to ROM. Code at .ORG 0 gets addresses $0000-$7FFF
        which are WRAM/IO in bank $00. These trampolines at $FF00+ sit in
        ROM and jump to the real handlers via bank $C0.

        `handlers` maps a vector name (nmi, irq, brk, cop, abort) to its
        handler label; each gets a `__hirom_<vector>` stub.

        Generated at $FF00:
            __hirom_reset: SEI / CLC / XCE / [MEMSEL] / JML main
            __hirom_nmi:   JML nmi_handler
            __hirom_irq:   JML irq_handler
        """
        self.emit_section_header("HiROM Vector Trampolines")
        from r65.compiler.codegen.bank_size import HIROM_TRAMPOLINE_ORG
        self.emit_directive(".BANK 0 SLOT 0")
        self.emit_directive(f".ORG ${HIROM_TRAMPOLINE_ORG:04X}")

        if reset:
            self.emit_label("__hirom_reset")
            self.emit_instr(Opcode.SEI)
            self.emit_instr(Opcode.CLC)
            self.emit_instr(Opcode.XCE)
            if fastrom:
                self.emit_memsel_write()
            self.emit_raw(f"    JML {reset}")

        for vector, handler in handlers.items():
            if handler:
                self.emit_label(f"__hirom_{vector}")
                self.emit_raw(f"    JML {handler}")

        self.emit_blank_line()

    def emit_interrupt_vectors(self, nmi=None, irq=None, reset=None,
                               brk=None, cop=None, abort=None):
        """
        Emit interrupt vector table using .SNESNATIVEVECTOR and .SNESEMUVECTOR.

        Each argument is a handler label, or None for __empty_handler. The
        native table gets every handler; the emulation table only RESET and
        IRQ/BRK, since a handler is compiled for native mode and the CPU
        runs in emulation only until the reset code's XCE.

        Generated:
            .SNESNATIVEVECTOR
              COP cop_handler
              BRK brk_handler
              ABORT abort_handler
              NMI nmi_handler
              IRQ irq_handler
            .ENDNATIVEVECTOR

            .SNESEMUVECTOR
              COP __empty_handler
              ABORT __empty_handler
              NMI __empty_handler
              RESET main
              IRQBRK irq_handler
            .ENDEMUVECTOR
        """
        empty = "__empty_handler"

        # Native mode vectors (5 vectors, no UNUSED)
        self.emit_section_header("Interrupt Vectors (Native Mode)")
        self.emit_directive(".SNESNATIVEVECTOR")
        self.emit_directive(f"COP {cop or empty}")
        self.emit_directive(f"BRK {brk or empty}")
        self.emit_directive(f"ABORT {abort or empty}")
        self.emit_directive(f"NMI {nmi or empty}")
        self.emit_directive(f"IRQ {irq or empty}")
        self.emit_directive(".ENDNATIVEVECTOR")
        self.emit_blank_line()

        # Emulation mode vectors (5 vectors, no UNUSED)
        self.emit_section_header("Interrupt Vectors (Emulation Mode)")
        self.emit_directive(".SNESEMUVECTOR")
        self.emit_directive(f"COP {empty}")
        self.emit_directive(f"ABORT {empty}")
        self.emit_directive(f"NMI {empty}")
        self.emit_directive(f"RESET {reset or empty}")
        self.emit_directive(f"IRQBRK {irq or empty}")
        self.emit_directive(".ENDEMUVECTOR")
        self.emit_blank_line()

    # ========================================================================
    # Symbol Exports
    # ========================================================================

    def emit_export(self, symbol: str):
        """
        Emit symbol export directive for .DEFINE symbols.

        Note: In WLA-DX, .EXPORT is only for .DEFINE symbols.
        Labels (function entry points) are automatically visible
        within a single-file assembly and don't need explicit exports.

        Args:
            symbol: Symbol to export

        Generated:
            .EXPORT BUFFER_SIZE
        """
        self.emit_directive(f".EXPORT {symbol}")

    def emit_exports(self, symbols: List[str]):
        """
        Emit multiple symbol exports for .DEFINE symbols.

        Note: In WLA-DX, .EXPORT is only for .DEFINE symbols.
        Labels don't need exports - they're automatically visible.

        Args:
            symbols: List of .DEFINE symbols to export

        Generated:
            ; ============================================================================
            ; Symbol Exports
            ; ============================================================================
            .EXPORT BUFFER_SIZE
            .EXPORT MAX_ENEMIES
        """
        if not symbols:
            return

        self.emit_section_header("Symbol Exports")
        for symbol in symbols:
            self.emit_export(symbol)
        self.emit_blank_line()

    # ========================================================================
    # Output Generation
    # ========================================================================

    def clear(self):
        """
        Clear all emitted content.

        Resets the emitter to its initial empty state.
        """
        self.nodes.clear()

    def get_nodes(self) -> List[AsmNode]:
        """
        Get the structured AsmNode representation.

        Returns:
            List of AsmNode objects
        """
        return self.nodes

    def set_nodes(self, nodes: List[AsmNode]):
        """
        Set the nodes (after optimization passes have modified them).

        Args:
            nodes: List of AsmNode objects
        """
        self.nodes = nodes

    def to_string(self) -> str:
        """
        Generate complete assembly as string.

        Returns:
            Complete assembly text with newlines
        """
        return emit_nodes(self.nodes)

    def to_lines(self) -> List[str]:
        """
        Get output as list of lines.

        Returns:
            List of assembly lines
        """
        return self.to_string().split('\n')

    def write_to_file(self, filepath: str):
        """
        Write assembly to file.

        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            f.write(self.to_string())
