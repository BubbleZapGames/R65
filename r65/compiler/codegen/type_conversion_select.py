"""
Type conversion selector: TypeConvert instruction handling.

Handles type conversion instruction generation including widening,
narrowing, and reinterpret casts.
"""

from typing import TYPE_CHECKING
from r65.compiler.mir.nodes import TypeConvert, Immediate as MIRImmediate
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector


class TypeConversionSelector:
    """
    Handles type conversion instruction selection.

    Manages generation of type conversion instructions including
    widening (zero/sign extend), narrowing (truncate), and reinterpret casts.
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize type conversion selector.

        Args:
            parent: Parent instruction selector (for helper method access)
        """
        self.parent = parent

    @property
    def emitter(self):
        return self.parent.emitter

    # ========================================================================
    # Emission Helpers
    # ========================================================================

    def _emit_instr(self, opcode: Opcode, operand=None, comment: str = None):
        """Emit an instruction using the node emitter."""
        self.emitter.emit_instr(opcode, operand, comment)

    def _emit_load_store(self, mnemonic: str, location, comment: str = None):
        """Emit a load/store instruction using parent's opcode selection."""
        opcode, operand = self.parent._get_opcode_for_location(mnemonic, location)
        self._emit_instr(opcode, operand, comment)

    def _emit_label(self, name: str):
        """Emit a label."""
        self.emitter.emit_label(name)

    # ========================================================================
    # Type Conversion Selection
    # ========================================================================

    def select_type_convert(self, instr: TypeConvert):
        """
        Generate code for TypeConvert instruction.

        Handles:
        - Widening: u8->u16 (zero-extend), i8->i16 (sign-extend)
        - Narrowing: u16->u8 (truncate to low byte)
        - Reinterpret: u8<->i8 (no operation, same bits)

        Args:
            instr: TypeConvert instruction
        """
        src_operand = instr.source
        dest_loc = self.parent._get_operand_location(instr.dest)

        # Get type information
        source_type = str(instr.source_type)
        target_type = str(instr.target_type)
        source_size = 1 if source_type in ['u8', 'i8', 'bool'] else 2
        target_size = 1 if target_type in ['u8', 'i8', 'bool'] else 2
        source_signed = source_type.startswith('i')

        # Case 1: Widening (8-bit -> 16-bit)
        if source_size == 1 and target_size == 2:
            self._emit_widening_conversion(src_operand, dest_loc, source_signed)

        # Case 2: Narrowing (16-bit -> 8-bit)
        elif source_size == 2 and target_size == 1:
            self._emit_narrowing_conversion(src_operand, dest_loc)

        # Case 3: Same size - should not happen (handled as Move in MIR builder)
        else:
            raise InstructionSelectionError(f"Unexpected type conversion: {source_type} to {target_type}")

    # ========================================================================
    # Widening Conversion
    # ========================================================================

    def _emit_widening_conversion(self, src_operand, dest_loc, source_signed: bool):
        """
        Emit widening conversion (8-bit to 16-bit).

        Args:
            src_operand: Source operand
            dest_loc: Destination location
            source_signed: True if source is signed (requires sign extension)
        """
        # Load source into A
        if isinstance(src_operand, MIRImmediate):
            value = src_operand.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            self._emit_load_store('LDA', src_loc)

        # Store low byte
        self._emit_load_store('STA', dest_loc)

        if source_signed:
            # Sign extension for i8 -> i16
            self._emit_sign_extension(dest_loc)
        else:
            # Zero extension for u8 -> u16
            self._emit_zero_extension(dest_loc)

    def _emit_sign_extension(self, dest_loc):
        """
        Emit sign extension for i8 -> i16.

        If high bit is set (negative), extend with 0xFF, else 0x00.
        """
        self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x80), "Check sign bit")
        self._emit_instr(Opcode.BEQ, Address("+"), "Branch if positive")
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0xFF), "Negative: extend with $FF")
        self._emit_instr(Opcode.BRA, Address("++"))
        self._emit_label("+")
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x00), "Positive: extend with $00")
        self._emit_label("++")

        # Store high byte
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_load_store('STA', dest_high)

    def _emit_zero_extension(self, dest_loc):
        """Emit zero extension for u8 -> u16."""
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x00), "Zero-extend high byte")

        # Store high byte
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_load_store('STA', dest_high)

    # ========================================================================
    # Narrowing Conversion
    # ========================================================================

    def _emit_narrowing_conversion(self, src_operand, dest_loc):
        """
        Emit narrowing conversion (16-bit to 8-bit).

        Truncates to low byte only.

        Args:
            src_operand: Source operand
            dest_loc: Destination location
        """
        if isinstance(src_operand, MIRImmediate):
            value = src_operand.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            self._emit_load_store('LDA', src_loc, "Load low byte")

        self._emit_load_store('STA', dest_loc)
