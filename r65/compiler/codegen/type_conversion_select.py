"""
Type conversion selector: TypeConvert instruction handling.

Handles type conversion instruction generation including widening,
narrowing, and reinterpret casts.
"""

from r65.compiler.mir.nodes import TypeConvert, Immediate as MIRImmediate, HardwareRegister
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address
from r65.compiler.codegen.base_selector import BaseSelector
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.hir.types import PointerTypeInfo


class TypeConversionSelector(BaseSelector):
    """
    Handles type conversion instruction selection.

    Manages generation of type conversion instructions including
    widening (zero/sign extend), narrowing (truncate), and reinterpret casts.
    """

    # ========================================================================
    # Emission Helpers
    # ========================================================================

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
        - Pointer conversions: near<->far, pointer-to-integer, integer-to-pointer

        Args:
            instr: TypeConvert instruction
        """
        src_operand = instr.source
        dest_loc = self.parent._get_operand_location(instr.dest)

        # Handle pointer type conversions
        if isinstance(instr.source_type, PointerTypeInfo) or isinstance(instr.target_type, PointerTypeInfo):
            self._emit_pointer_conversion(instr, src_operand, dest_loc)
            return

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
        elif isinstance(src_operand, HardwareRegister):
            # Transfer from hardware register to A
            if src_operand.name == 'X':
                self._emit_instr(Opcode.TXA, comment="Widen X to 16-bit")
            elif src_operand.name == 'Y':
                self._emit_instr(Opcode.TYA, comment="Widen Y to 16-bit")
            elif src_operand.name == 'A':
                pass  # Already in A
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                if src_loc.hw_register == 'X':
                    self._emit_instr(Opcode.TXA, comment="Widen X to 16-bit")
                elif src_loc.hw_register == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Widen Y to 16-bit")
                elif src_loc.hw_register == 'A':
                    pass  # Already in A
            else:
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

        Truncates to low byte only. For hardware register destinations (X, Y),
        uses AND #$00FF pattern to ensure clean 16-bit value with zeroed high byte.

        Args:
            src_operand: Source operand
            dest_loc: Destination location
        """
        # Check if destination is a hardware register that needs special handling
        if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register in ('X', 'Y'):
            # X/Y = value as u8: zero-extend to ensure clean 16-bit value
            # Pattern: REP #$20, load value, AND #$00FF, TAX/TAY, SEP #$20
            self.parent._ensure_m16_mode()

            if isinstance(src_operand, MIRImmediate):
                # Immediate value - just mask it
                value = src_operand.value & 0xFF
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value), "Load masked immediate")
            elif isinstance(src_operand, HardwareRegister):
                if src_operand.name == 'X':
                    self._emit_instr(Opcode.TXA, comment="Load X for narrowing")
                elif src_operand.name == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Load Y for narrowing")
                elif src_operand.name == 'A':
                    pass  # Already in A
                elif src_operand.name == 'B':
                    self.parent._access_b_value_in_a()
                    self.parent._ensure_m16_mode()  # _access_b_value_in_a switches to m8
                # Mask to low byte
                self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Zero-extend to 16-bit")
            else:
                src_loc = self.parent._get_operand_location(src_operand)
                if src_loc.kind == LocationKind.HARDWARE:
                    if src_loc.hw_register == 'X':
                        self._emit_instr(Opcode.TXA, comment="Load X for narrowing")
                    elif src_loc.hw_register == 'Y':
                        self._emit_instr(Opcode.TYA, comment="Load Y for narrowing")
                    elif src_loc.hw_register == 'A':
                        pass
                    elif src_loc.hw_register == 'B':
                        self.parent._access_b_value_in_a()
                        self.parent._ensure_m16_mode()
                    self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Zero-extend to 16-bit")
                else:
                    self._emit_load_store('LDA', src_loc)
                    self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Zero-extend to 16-bit")

            # Transfer to destination register
            if dest_loc.hw_register == 'X':
                self._emit_instr(Opcode.TAX, comment="Transfer to X")
            else:
                self._emit_instr(Opcode.TAY, comment="Transfer to Y")

            self.parent._ensure_m8_mode()
            return

        # Check if destination is A register
        if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
            # A = value as u8: just load the low byte, switch to 8-bit mode
            if isinstance(src_operand, MIRImmediate):
                value = src_operand.value & 0xFF
                self.parent._ensure_m8_mode()
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            elif isinstance(src_operand, HardwareRegister):
                if src_operand.name == 'X':
                    # In 16-bit mode, TXA would copy both bytes including B
                    # Use AND #$00FF pattern to ensure only low byte
                    self.parent._ensure_m16_mode()
                    self._emit_instr(Opcode.TXA, comment="Load X for narrowing")
                    self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Mask to low byte")
                    self.parent._ensure_m8_mode()
                elif src_operand.name == 'Y':
                    self.parent._ensure_m16_mode()
                    self._emit_instr(Opcode.TYA, comment="Load Y for narrowing")
                    self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Mask to low byte")
                    self.parent._ensure_m8_mode()
                elif src_operand.name == 'A':
                    # A = A as u8: just switch to 8-bit mode (truncate)
                    self.parent._ensure_m8_mode()
                elif src_operand.name == 'B':
                    # A = B as u8: B is already 8-bit
                    self.parent._access_b_value_in_a()
            else:
                src_loc = self.parent._get_operand_location(src_operand)
                self.parent._ensure_m8_mode()
                if src_loc.kind == LocationKind.HARDWARE:
                    if src_loc.hw_register == 'X':
                        self._emit_instr(Opcode.TXA, comment="Narrow X to u8")
                    elif src_loc.hw_register == 'Y':
                        self._emit_instr(Opcode.TYA, comment="Narrow Y to u8")
                    elif src_loc.hw_register == 'B':
                        self.parent._access_b_value_in_a()
                else:
                    self._emit_load_store('LDA', src_loc, "Load low byte")
            return

        # Default case: memory destination
        if isinstance(src_operand, MIRImmediate):
            value = src_operand.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
        elif isinstance(src_operand, HardwareRegister):
            # Transfer from hardware register to A for narrowing
            if src_operand.name == 'X':
                self._emit_instr(Opcode.TXA, comment="Narrow X to u8")
            elif src_operand.name == 'Y':
                self._emit_instr(Opcode.TYA, comment="Narrow Y to u8")
            elif src_operand.name == 'A':
                pass  # Already in A
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                # Hardware register location
                if src_loc.hw_register == 'X':
                    self._emit_instr(Opcode.TXA, comment="Narrow X to u8")
                elif src_loc.hw_register == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Narrow Y to u8")
                elif src_loc.hw_register == 'A':
                    pass  # Already in A
            else:
                self._emit_load_store('LDA', src_loc, "Load low byte")

        # CRITICAL: Switch to 8-bit mode before storing narrowed value.
        # If we're in 16-bit mode (from previous 16-bit operations), STA would
        # write 2 bytes instead of 1, potentially corrupting adjacent memory
        # (including return addresses on the stack).
        self.parent._ensure_m8_mode()
        self._emit_load_store('STA', dest_loc)

    # ========================================================================
    # Pointer Conversion
    # ========================================================================

    def _emit_pointer_conversion(self, instr: TypeConvert, src_operand, dest_loc):
        """
        Emit pointer type conversion.

        Handles:
        - Near pointer to far pointer (2 bytes → 3 bytes, add bank 0)
        - Far pointer to near pointer (3 bytes → 2 bytes, truncate)
        - Pointer to u16 (extract 16-bit address)
        - Pointer to u8 (extract low byte)
        - u16 to pointer (direct copy)

        Args:
            instr: TypeConvert instruction
            src_operand: Source operand
            dest_loc: Destination location
        """
        source_type = instr.source_type
        target_type = instr.target_type

        # Get sizes
        if isinstance(source_type, PointerTypeInfo):
            source_size = 3 if source_type.is_far else 2
        else:
            source_size = 1 if str(source_type) in ['u8', 'i8', 'bool'] else 2

        if isinstance(target_type, PointerTypeInfo):
            target_size = 3 if target_type.is_far else 2
        else:
            target_size = 1 if str(target_type) in ['u8', 'i8', 'bool'] else 2

        # Near to far pointer: copy 2 bytes, add bank 0
        if source_size == 2 and target_size == 3:
            self._emit_near_to_far_pointer(src_operand, dest_loc)

        # Far to near pointer: truncate to 2 bytes
        elif source_size == 3 and target_size == 2:
            self._emit_far_to_near_pointer(src_operand, dest_loc)

        # Far pointer to u8: extract low byte
        elif source_size == 3 and target_size == 1:
            self._emit_pointer_to_u8(src_operand, dest_loc)

        # Same size: direct copy
        elif source_size == target_size:
            self._emit_pointer_copy(src_operand, dest_loc, source_size)

        else:
            raise InstructionSelectionError(
                f"Unsupported pointer conversion: {source_type} to {target_type}"
            )

    def _emit_near_to_far_pointer(self, src_operand, dest_loc):
        """Convert near pointer (2 bytes) to far pointer (3 bytes)."""
        if isinstance(src_operand, MIRImmediate):
            # Immediate: load and store each byte
            value = src_operand.value & 0xFFFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value & 0xFF))
            self._emit_load_store('STA', dest_loc)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate((value >> 8) & 0xFF))
            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_load_store('STA', dest_high)
        elif isinstance(src_operand, HardwareRegister):
            # Hardware register source: X or Y (16-bit)
            if src_operand.name == 'X':
                self._emit_instr(Opcode.TXA, comment="Near ptr from X")
            elif src_operand.name == 'Y':
                self._emit_instr(Opcode.TYA, comment="Near ptr from Y")
            # Store low byte
            self._emit_load_store('STA', dest_loc)
            # High byte is in B for 16-bit, need to use XBA
            self._emit_instr(Opcode.XBA, comment="Get high byte")
            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_load_store('STA', dest_high)
            self._emit_instr(Opcode.XBA, comment="Restore A")
        else:
            # Variable: copy 2 bytes
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                # Hardware register location
                if src_loc.hw_register == 'X':
                    self._emit_instr(Opcode.TXA, comment="Near ptr from X")
                elif src_loc.hw_register == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Near ptr from Y")
                self._emit_load_store('STA', dest_loc)
                self._emit_instr(Opcode.XBA, comment="Get high byte")
                dest_high = self.parent._offset_location(dest_loc, 1)
                self._emit_load_store('STA', dest_high)
                self._emit_instr(Opcode.XBA, comment="Restore A")
            else:
                self._emit_load_store('LDA', src_loc)
                self._emit_load_store('STA', dest_loc)
                src_high = self.parent._offset_location(src_loc, 1)
                dest_high = self.parent._offset_location(dest_loc, 1)
                self._emit_load_store('LDA', src_high)
                self._emit_load_store('STA', dest_high)

        # Set bank byte - use symbol's ROM bank if available, else 0
        dest_bank = self.parent._offset_location(dest_loc, 2)
        bank_ref = 0x00
        symbol = None
        # Check if source has a symbol with ROM label (for address-of ROM data)
        # Symbol can be on Immediate or VirtualRegister (propagated from address-of)
        if hasattr(src_operand, 'symbol') and src_operand.symbol:
            symbol = src_operand.symbol
            if hasattr(symbol, 'rom_label') and symbol.rom_label:
                bank_ref = f":{symbol.rom_label}"
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(bank_ref),
                        f"Bank byte{' of ' + symbol.rom_label if isinstance(bank_ref, str) else ' = 0'}")
        self._emit_load_store('STA', dest_bank)

    def _emit_far_to_near_pointer(self, src_operand, dest_loc):
        """Convert far pointer (3 bytes) to near pointer (2 bytes)."""
        if isinstance(src_operand, MIRImmediate):
            value = src_operand.value & 0xFFFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value & 0xFF))
            self._emit_load_store('STA', dest_loc)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate((value >> 8) & 0xFF))
            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_load_store('STA', dest_high)
        elif isinstance(src_operand, HardwareRegister):
            # Truncate from hardware register (unlikely but handle it)
            if src_operand.name == 'X':
                self._emit_instr(Opcode.TXA, comment="Far ptr from X (truncate)")
            elif src_operand.name == 'Y':
                self._emit_instr(Opcode.TYA, comment="Far ptr from Y (truncate)")
            self._emit_load_store('STA', dest_loc)
            self._emit_instr(Opcode.XBA, comment="Get high byte")
            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_load_store('STA', dest_high)
            self._emit_instr(Opcode.XBA, comment="Restore A")
        else:
            # Copy low 2 bytes only
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                if src_loc.hw_register == 'X':
                    self._emit_instr(Opcode.TXA, comment="Far ptr from X (truncate)")
                elif src_loc.hw_register == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Far ptr from Y (truncate)")
                self._emit_load_store('STA', dest_loc)
                self._emit_instr(Opcode.XBA, comment="Get high byte")
                dest_high = self.parent._offset_location(dest_loc, 1)
                self._emit_load_store('STA', dest_high)
                self._emit_instr(Opcode.XBA, comment="Restore A")
            else:
                self._emit_load_store('LDA', src_loc)
                self._emit_load_store('STA', dest_loc)
                src_high = self.parent._offset_location(src_loc, 1)
                dest_high = self.parent._offset_location(dest_loc, 1)
                self._emit_load_store('LDA', src_high)
                self._emit_load_store('STA', dest_high)

    def _emit_pointer_to_u8(self, src_operand, dest_loc):
        """Convert pointer to u8 (extract low byte)."""
        if isinstance(src_operand, MIRImmediate):
            value = src_operand.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
        elif isinstance(src_operand, HardwareRegister):
            if src_operand.name == 'X':
                self._emit_instr(Opcode.TXA, comment="Ptr to u8 from X")
            elif src_operand.name == 'Y':
                self._emit_instr(Opcode.TYA, comment="Ptr to u8 from Y")
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                if src_loc.hw_register == 'X':
                    self._emit_instr(Opcode.TXA, comment="Ptr to u8 from X")
                elif src_loc.hw_register == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Ptr to u8 from Y")
            else:
                self._emit_load_store('LDA', src_loc)

        self._emit_load_store('STA', dest_loc)

    def _emit_pointer_copy(self, src_operand, dest_loc, size: int):
        """Copy pointer of given size."""
        if isinstance(src_operand, MIRImmediate):
            value = src_operand.value
            # Check if source has a symbol with ROM label (for address-of ROM data)
            symbol = None
            if hasattr(src_operand, 'symbol') and src_operand.symbol:
                symbol = src_operand.symbol
                if hasattr(symbol, 'rom_label') and symbol.rom_label:
                    pass  # Will use symbol.rom_label for bank byte
                else:
                    symbol = None  # No ROM label, use numeric value

            for i in range(size):
                if i == 2 and symbol and hasattr(symbol, 'rom_label') and symbol.rom_label:
                    # Use symbol's ROM bank for byte 2 (bank byte)
                    bank_ref = f":{symbol.rom_label}"
                    self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(bank_ref),
                                    f"Bank byte of {symbol.rom_label}")
                else:
                    self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate((value >> (i * 8)) & 0xFF))
                if i == 0:
                    self._emit_load_store('STA', dest_loc)
                else:
                    dest_byte = self.parent._offset_location(dest_loc, i)
                    self._emit_load_store('STA', dest_byte)
        elif isinstance(src_operand, HardwareRegister):
            # Copy from hardware register (X or Y)
            if src_operand.name == 'X':
                self._emit_instr(Opcode.TXA, comment="Ptr copy from X")
            elif src_operand.name == 'Y':
                self._emit_instr(Opcode.TYA, comment="Ptr copy from Y")
            self._emit_load_store('STA', dest_loc)
            if size > 1:
                self._emit_instr(Opcode.XBA, comment="Get high byte")
                dest_high = self.parent._offset_location(dest_loc, 1)
                self._emit_load_store('STA', dest_high)
                self._emit_instr(Opcode.XBA, comment="Restore A")
            if size > 2:
                # Bank byte - X/Y registers are only 16-bit so we can't get bank from them
                # Default to 0 (caller should use near-to-far conversion for ROM addresses)
                dest_bank = self.parent._offset_location(dest_loc, 2)
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x00), "Bank byte = 0 (register has no bank)")
                self._emit_load_store('STA', dest_bank)
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                # Copy from hardware register location
                if src_loc.hw_register == 'X':
                    self._emit_instr(Opcode.TXA, comment="Ptr copy from X")
                elif src_loc.hw_register == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Ptr copy from Y")
                self._emit_load_store('STA', dest_loc)
                if size > 1:
                    self._emit_instr(Opcode.XBA, comment="Get high byte")
                    dest_high = self.parent._offset_location(dest_loc, 1)
                    self._emit_load_store('STA', dest_high)
                    self._emit_instr(Opcode.XBA, comment="Restore A")
                if size > 2:
                    # Bank byte - X/Y registers are only 16-bit so we can't get bank from them
                    # Default to 0 (caller should use near-to-far conversion for ROM addresses)
                    dest_bank = self.parent._offset_location(dest_loc, 2)
                    self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x00), "Bank byte = 0 (register has no bank)")
                    self._emit_load_store('STA', dest_bank)
            else:
                for i in range(size):
                    if i == 0:
                        self._emit_load_store('LDA', src_loc)
                        self._emit_load_store('STA', dest_loc)
                    else:
                        src_byte = self.parent._offset_location(src_loc, i)
                        dest_byte = self.parent._offset_location(dest_loc, i)
                        self._emit_load_store('LDA', src_byte)
                        self._emit_load_store('STA', dest_byte)
