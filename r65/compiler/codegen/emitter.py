# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Assembly emitter: generates WLA-DX assembly text.

Provides methods for emitting instructions, labels, comments, and sections
in WLA-DX assembly format for the 65816 processor.

Builds structured AsmNode objects which are converted to assembly text
via emit_nodes().
"""

from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from r65.compiler.errors import SourceLocation

from r65.compiler.codegen.opcodes import (
    Opcode, mnemonic, addressing_mode, BRANCH_OPCODES,
)
from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Comment, Directive, BlankLine, RawAsm,
    Immediate, Address, StackOffset, BlockMove, Operand,
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

    # PEA: Push Effective Address — takes a 16-bit absolute operand
    if opcode == Opcode.PEA:
        if operand is not None:
            match operand:
                case Immediate(value) | Address(value):
                    return f"{mnem} {_format_value(value)}"
                case _:
                    return f"{mnem} {operand}"
        return mnem

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
    ACCUMULATOR_OPCODES = {Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR, Opcode.INC, Opcode.DEC}
    if opcode in ACCUMULATOR_OPCODES:
        return f"{mnem} A"

    # Block move instructions (MVN/MVP) - special operand format
    if isinstance(operand, BlockMove):
        return f"{mnem} ${operand.src_bank:02X}, ${operand.dst_bank:02X}"

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

        # Absolute
        # For labels, use .w suffix to force 16-bit addressing in WLA-DX
        case ("ABSOLUTE", Address(value)):
            suffix = ".w" if isinstance(value, str) else ""
            return f"{mnem}{suffix} {_format_absolute(value)}"
        case ("ABSOLUTE_X", Address(value)):
            suffix = ".w" if isinstance(value, str) else ""
            return f"{mnem}{suffix} {_format_absolute(value)},X"
        case ("ABSOLUTE_Y", Address(value)):
            suffix = ".w" if isinstance(value, str) else ""
            return f"{mnem}{suffix} {_format_absolute(value)},Y"

        # Long (24-bit)
        # For labels, use .l suffix to force 24-bit addressing in WLA-DX
        case ("LONG", Address(value)):
            suffix = ".l" if isinstance(value, str) else ""
            return f"{mnem}{suffix} {_format_long_value(value)}"
        case ("LONG_X", Address(value)):
            suffix = ".l" if isinstance(value, str) else ""
            return f"{mnem}{suffix} {_format_long_value(value)},X"

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
        case (_, BlockMove(src, dst)):
            return f"{mnem} ${src:02X}, ${dst:02X}"

        case _:
            # Fallback for unhandled cases
            return f"{mnem} {operand}"


def _format_value(value: int | str) -> str:
    """Format an immediate or address value."""
    if isinstance(value, str):
        return value
    if value < 0x100:
        return f"${value:02X}"
    return f"${value:04X}"


def _format_absolute(value: int | str) -> str:
    """Format an absolute address value, forcing 16-bit addressing for labels."""
    if isinstance(value, str):
        # For WLA-DX, labels need absolute addressing but the .w suffix approach
        # requires modifying the mnemonic, so we just return the label and ensure
        # the opcode used is correct (ABSOLUTE mode, not DP mode)
        return value
    return f"${value:04X}"


def _format_long_value(value: int | str) -> str:
    """Format a 24-bit long address value."""
    if isinstance(value, str):
        return value
    return f"${value:06X}"


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
        # Track current processor modes for optimization
        self._current_accu_mode = 8  # Default m8 mode
        self._current_index_mode = 16  # Default x16 mode
        # Track which register (if any) the N/Z flags currently reflect.
        # Used to elide redundant CMP/CPX/CPY #0 when flags are already valid.
        self.nz_valid_for: str | None = None  # 'A', 'X', 'Y', or None
        # Track banks that have had data/code emitted to avoid .ORG 0 overlap
        self._used_banks: set = set()
        # HiROM flag: when True, .BASE $C0 is emitted before ROM sections
        # and .BASE $00 before RAM sections so #: bank bytes resolve correctly.
        self.is_hirom: bool = False

    # ========================================================================
    # Low-Level Node Emission
    # ========================================================================

    def emit(self, node: AsmNode):
        """Emit a single node."""
        self.nodes.append(node)

    def emit_instr(self, opcode: Opcode, operand: Operand | None = None,
                   comment: str | None = None, source_loc: 'SourceLocation | None' = None):
        """
        Emit an instruction using structured Opcode enum.

        Args:
            opcode: The Opcode enum value (e.g., Opcode.LDA_IMMEDIATE)
            operand: The operand (Immediate, Address, StackOffset, etc.)
            comment: Optional comment
            source_loc: Optional source location for debug info
        """
        self.nodes.append(Instruction(opcode, operand, comment, source_loc))
        self._update_nz_tracking(opcode)

    def emit_raw(self, text: str):
        """Emit raw assembly text."""
        self.nodes.append(RawAsm(text))
        self.nz_valid_for = None  # Unknown effect on flags

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
        """Emit .ACCU directive and update tracked mode."""
        self.nodes.append(Directive(".ACCU", [str(bits)]))
        self._current_accu_mode = bits

    def emit_index_mode(self, bits: int):
        """Emit .INDEX directive and update tracked mode."""
        self.nodes.append(Directive(".INDEX", [str(bits)]))
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
            ; Compiler Version: 0.1.0
            ; ============================================================================
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.emit_section_header("Generated by R65 Compiler")
        self.emit_comment(f"Source: {self.source_file}")
        self.emit_comment(f"Generated: {now}")
        self.emit_comment("Compiler Version: 0.1.0")
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
                        has_wram_7f: bool = False):
        """
        Emit WLA-DX memory map and ROM bank map.

        Args:
            rom_type: "lorom" (32KB banks) or "hirom" (64KB banks)
            banks: Number of ROM banks

        Generated (LoROM):
            .MEMORYMAP
                DEFAULTSLOT 0
                SLOTSIZE $8000
                SLOT 0 $8000
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

        # Store HiROM flag so emit_bank_directive and emit_ramsection
        # can emit the correct .BASE before each section.
        self.is_hirom = rom_type.lower() != "lorom"

    # ========================================================================
    # Bank Directives
    # ========================================================================

    def emit_bank_directive(self, bank_num: int, slot: int = 0):
        """
        Emit bank directive and origin.

        For HiROM, emits .BASE $C0 so #: bank bytes resolve to $C0+
        (where $0000-$FFFF all map to ROM).

        Args:
            bank_num: Bank number
            slot: Slot number (usually 0)
        """
        if self.is_hirom:
            self.emit_directive(".BASE $C0")
        self.emit_directive(f".BANK {bank_num} SLOT {slot}")
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
        # Reset .BASE so subsequent ROM bank labels get correct bank bytes.
        # Without this, .BASE $7E persists and offsets all later :label
        # calculations (e.g., BANK 1 would resolve to $7F instead of $01).
        self.emit_directive(".BASE $00")

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

    def emit_local_label(self, label: str):
        """
        Emit a local label with indentation.

        Args:
            label: Label name

        Generated:
            __L1:
        """
        self.emit_label(label)

    # ========================================================================
    # N/Z Flag Tracking
    # ========================================================================

    # Mnemonics that always set N/Z for A (all addressing modes)
    _NZ_A_MNEMONICS = frozenset({'LDA', 'AND', 'ORA', 'EOR', 'ADC', 'SBC'})
    # Mnemonics that always set N/Z for X
    _NZ_X_MNEMONICS = frozenset({'LDX'})
    # Mnemonics that always set N/Z for Y
    _NZ_Y_MNEMONICS = frozenset({'LDY'})
    # Accumulator-form implied instructions that set N/Z for A
    _NZ_A_OPCODES = frozenset({
        Opcode.TXA, Opcode.TYA, Opcode.PLA, Opcode.XBA, Opcode.TSC,
        Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR,  # Accumulator shifts
        Opcode.INC, Opcode.DEC,  # Accumulator inc/dec
    })
    # Implied instructions that set N/Z for X
    _NZ_X_OPCODES = frozenset({
        Opcode.TAX, Opcode.PLX, Opcode.INX, Opcode.DEX, Opcode.TSX,
    })
    # Implied instructions that set N/Z for Y
    _NZ_Y_OPCODES = frozenset({
        Opcode.TAY, Opcode.PLY, Opcode.INY, Opcode.DEY,
    })
    # Instructions that do NOT affect N/Z flags (tracking preserved)
    _NZ_PRESERVE_MNEMONICS = frozenset({
        'STA', 'STX', 'STY', 'STZ',
        'PHA', 'PHX', 'PHY', 'PHB', 'PHD', 'PHK',
        'TXS', 'TCS', 'TCD',  # Transfers to S/SP/D don't affect N/Z
        'NOP', 'WDM',
        'BRA', 'BRL', 'BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS',
        'JMP',
        'CLC', 'SEC', 'CLI', 'SEI', 'CLD', 'SED', 'CLV',
        'REP', 'SEP',  # Only affect M/X bits (#$20/#$10) in compiler usage
    })

    def _update_nz_tracking(self, opcode: Opcode):
        """
        Update N/Z flag tracking after emitting an instruction.

        Tracks which register (A, X, or Y) the processor N/Z flags currently
        reflect. Used to elide redundant CMP/CPX/CPY #0 instructions when
        the flags are already valid from a prior load, transfer, or ALU op.
        """
        mn = mnemonic(opcode)

        # Check mnemonic-based sets first (covers all addressing modes)
        if mn in self._NZ_A_MNEMONICS:
            self.nz_valid_for = 'A'
        elif mn in self._NZ_X_MNEMONICS:
            self.nz_valid_for = 'X'
        elif mn in self._NZ_Y_MNEMONICS:
            self.nz_valid_for = 'Y'
        # Check specific opcodes (implied/accumulator forms)
        elif opcode in self._NZ_A_OPCODES:
            self.nz_valid_for = 'A'
        elif opcode in self._NZ_X_OPCODES:
            self.nz_valid_for = 'X'
        elif opcode in self._NZ_Y_OPCODES:
            self.nz_valid_for = 'Y'
        # Instructions that don't touch N/Z — preserve tracking
        elif mn in self._NZ_PRESERVE_MNEMONICS:
            pass
        else:
            # Everything else (CMP, CPX, CPY, BIT, PLP, JSR, JSL, RTI,
            # memory-form INC/DEC/ASL/LSR/ROL/ROR, etc.) — clear tracking
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
            .DEFINE TEMP $20        ; main.r65:5
        """
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
        """
        line = f".EQU {name} {value}"

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

    def emit_incbin(self, filepath: str, label: Optional[str] = None):
        """
        Emit binary file inclusion (.INCBIN directive).

        Args:
            filepath: Path to binary file to include
            label: Optional label for the data

        Generated:
            LABEL:
            .INCBIN "path/to/file.bin"
        """
        if label:
            self.emit_label(label)
        self.emit_directive(f'    .INCBIN "{filepath}"')

    # ========================================================================
    # Interrupt Vectors
    # ========================================================================

    def emit_snes_header(self, snesrom_config=None, romsize_value: int = 0x08):
        """
        Emit SNES ROM header using .SNESHEADER directive.

        Args:
            snesrom_config: SnesRomConfig from #[snesrom(...)] directive, or None for defaults
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

        Note: Vectors are placed separately using .ORGA directives.
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

    def emit_hirom_vector_trampolines(self, reset=None, nmi=None, irq=None):
        """
        Emit HiROM interrupt vector trampolines at addresses >= $8000.

        In HiROM, the CPU reads interrupt vectors from bank $00 where only
        $8000-$FFFF maps to ROM. Code at .ORG 0 gets addresses $0000-$7FFF
        which are WRAM/IO in bank $00. These trampolines at $FF00+ sit in
        ROM and jump to the real handlers via bank $C0.

        Generated at $FF00:
            __hirom_reset: SEI / CLC / XCE / JML main
            __hirom_nmi:   JML nmi_handler
            __hirom_irq:   JML irq_handler
        """
        self.emit_section_header("HiROM Vector Trampolines")
        self.emit_directive(".BANK 0 SLOT 0")
        self.emit_directive(".ORG $FF00")

        if reset:
            self.emit_label("__hirom_reset")
            self.emit_instr(Opcode.SEI)
            self.emit_instr(Opcode.CLC)
            self.emit_instr(Opcode.XCE)
            self.emit_raw(f"    JML {reset}")

        if nmi:
            self.emit_label("__hirom_nmi")
            self.emit_raw(f"    JML {nmi}")

        if irq:
            self.emit_label("__hirom_irq")
            self.emit_raw(f"    JML {irq}")

        self.emit_blank_line()

    def emit_interrupt_vectors(self, nmi=None, irq=None, reset=None):
        """
        Emit interrupt vector table using .SNESNATIVEVECTOR and .SNESEMUVECTOR.

        Args:
            nmi: NMI handler label (or None for __empty_handler)
            irq: IRQ handler label (or None for __empty_handler)
            reset: RESET handler label (or None for __empty_handler)

        Generated:
            .SNESNATIVEVECTOR
              COP __empty_handler
              BRK __empty_handler
              ABORT __empty_handler
              NMI nmi_handler
              IRQ __empty_handler
            .ENDNATIVEVECTOR

            .SNESEMUVECTOR
              COP __empty_handler
              ABORT __empty_handler
              NMI __empty_handler
              RESET main
              IRQBRK __empty_handler
            .ENDEMUVECTOR
        """
        # Default to empty handler
        nmi = nmi or "__empty_handler"
        irq = irq or "__empty_handler"
        reset = reset or "__empty_handler"

        # Native mode vectors (5 vectors, no UNUSED)
        self.emit_section_header("Interrupt Vectors (Native Mode)")
        self.emit_directive(".SNESNATIVEVECTOR")
        self.emit_directive(f"COP {nmi}")
        self.emit_directive(f"BRK __empty_handler")
        self.emit_directive(f"ABORT __empty_handler")
        self.emit_directive(f"NMI {nmi}")
        self.emit_directive(f"IRQ {irq}")
        self.emit_directive(".ENDNATIVEVECTOR")
        self.emit_blank_line()

        # Emulation mode vectors (5 vectors, no UNUSED)
        self.emit_section_header("Interrupt Vectors (Emulation Mode)")
        self.emit_directive(".SNESEMUVECTOR")
        self.emit_directive(f"COP __empty_handler")
        self.emit_directive(f"ABORT __empty_handler")
        self.emit_directive(f"NMI __empty_handler")
        self.emit_directive(f"RESET {reset}")
        self.emit_directive(f"IRQBRK {irq}")
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
