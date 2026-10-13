# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
from r65.compiler.hir.types import PointerTypeInfo
from r65.compiler.codegen.type_utils import convert_operand_widths, scalar_width


class TypeConversionSelector(BaseSelector):
    """
    Handles type conversion instruction selection.

    Manages generation of type conversion instructions including
    widening (zero/sign extend), narrowing (truncate), and reinterpret casts.
    """

    # ========================================================================
    # Emission Helpers
    # ========================================================================

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
        from r65.compiler.hir.types import strip_newtype

        src_operand = instr.source
        dest_loc = self.parent._get_operand_location(instr.dest)

        # A conversion is about machine widths, so newtypes answer for their
        # payload. Without this the size falls out of a *name* comparison below
        # and any newtype would be taken for 2 bytes.
        instr_source_type = strip_newtype(instr.source_type)
        instr_target_type = strip_newtype(instr.target_type)

        # Handle pointer type conversions
        if isinstance(instr_source_type, PointerTypeInfo) or isinstance(instr_target_type, PointerTypeInfo):
            self._emit_pointer_conversion(instr, src_operand, dest_loc)
            return

        # Get type information. The widths come from the shared helper so the
        # register allocator, which has to predict which branch below runs
        # before it can let a result live in A, asks exactly this question.
        source_type = str(instr_source_type)
        target_type = str(instr_target_type)
        source_size, target_size = convert_operand_widths(instr)
        source_signed = source_type.startswith('i')

        # Case 1: Widening (8-bit -> 16-bit)
        if source_size == 1 and target_size == 2:
            self._emit_widening_conversion(src_operand, dest_loc, source_signed)

        # Case 2: Narrowing (16-bit -> 8-bit)
        elif source_size == 2 and target_size == 1:
            self._emit_narrowing_conversion(src_operand, dest_loc)

        # Case 3: Same size - should not happen (handled as Move in MIR builder)
        else:
            raise InstructionSelectionError(f"Unexpected type conversion: {source_type} to {target_type}", source_loc=self.parent._current_source_loc)

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
        if not source_signed and self._emit_zero_extension_through_a(src_operand, dest_loc):
            return

        # For unsigned zero-extension from memory/stack sources to memory/stack
        # destinations, use m16 AND #$00FF for a compact single-store approach.
        # This avoids mode-switching overhead (SEP/REP) of the byte-by-byte path.
        if (not source_signed and
                not isinstance(src_operand, (MIRImmediate, HardwareRegister)) and
                not dest_loc.is_hw()):
            src_loc = self.parent._get_operand_location(src_operand)
            if not src_loc.is_hw():
                self.parent._ensure_m16_mode()
                self._emit_load_store('LDA', src_loc)
                self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF),
                                 "Zero-extend u8 to u16")
                self._emit_load_store('STA', dest_loc)
                return

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
            elif src_operand.name == 'B':
                self.parent._access_b_value_in_a()
            elif src_operand.name == 'A':
                pass  # Already in A
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.is_hw():
                if src_loc.hw_register == 'X':
                    self._emit_instr(Opcode.TXA, comment="Widen X to 16-bit")
                elif src_loc.hw_register == 'Y':
                    self._emit_instr(Opcode.TYA, comment="Widen Y to 16-bit")
                elif src_loc.hw_register == 'B':
                    self.parent._access_b_value_in_a()
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

    def _emit_zero_extension_through_a(self, src_operand, dest_loc) -> bool:
        """Zero-extend in m16 when the source or the result lives in A.

        A result in A (`is_a_resident_convert` let it coalesce) is left there
        in m16 with no store. A source already in A is masked in place and
        stored with one 16-bit STA. Returns False for every other shape, which
        the byte-wise path handles.
        """
        if isinstance(src_operand, MIRImmediate):
            src_reg, src_loc = None, None
        elif isinstance(src_operand, HardwareRegister):
            src_reg, src_loc = src_operand.name, None
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            src_reg = src_loc.hw_register if src_loc.is_hw() else None

        dest_in_a = dest_loc.is_hw('A')
        if not dest_in_a and (src_reg != 'A' or dest_loc.is_hw()):
            return False

        if isinstance(src_operand, MIRImmediate):
            self.parent._ensure_m16_mode()
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(src_operand.value & 0xFF))
        else:
            if src_reg == 'B':
                self.parent._access_b_value_in_a()
            elif src_reg is None and src_loc.storage_type == 'hw':
                # One byte only: a 16-bit read also reads the next I/O port,
                # and some of those clear a flag when read.
                self.parent._ensure_m8_mode()
                self._emit_load_store('LDA', src_loc)
                src_reg = 'A'
            self.parent._ensure_m16_mode()
            if src_reg == 'X':
                self._emit_instr(Opcode.TXA, comment="Widen X to 16-bit")
            elif src_reg == 'Y':
                self._emit_instr(Opcode.TYA, comment="Widen Y to 16-bit")
            elif src_reg is None:
                self._emit_load_store('LDA', src_loc)
            self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF),
                             "Zero-extend u8 to u16")
        if not dest_in_a:
            self._emit_load_store('STA', dest_loc)
        return True

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
        # CRITICAL: Switch to 8-bit mode before storing high byte.
        # If we're in 16-bit mode, STA would write 2 bytes instead of 1,
        # corrupting adjacent memory (including other stack slots).
        self.parent._ensure_m8_mode()
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_load_store('STA', dest_high)

    def _emit_zero_extension(self, dest_loc):
        """Emit zero extension for u8 -> u16."""
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x00), "Zero-extend high byte")

        # Store high byte
        # CRITICAL: Switch to 8-bit mode before storing high byte.
        # If we're in 16-bit mode, STA would write 2 bytes instead of 1,
        # corrupting adjacent memory (including other stack slots).
        self.parent._ensure_m8_mode()
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
        if dest_loc.is_hw() and dest_loc.hw_register in ('X', 'Y'):
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
                if src_loc.is_hw():
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
        if dest_loc.is_hw('A'):
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
                if src_loc.is_hw():
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
            if src_loc.is_hw():
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
        # Strip here too: select_type_convert stripped into locals, but this
        # method re-reads the instruction, so a newtype would size by name.
        from r65.compiler.hir.types import strip_newtype
        source_type = strip_newtype(instr.source_type)
        target_type = strip_newtype(instr.target_type)

        source_size = scalar_width(source_type) or 2
        target_size = scalar_width(target_type) or 2

        # Near to far: the two bytes, then a bank byte. Narrowing (far to
        # near, a pointer to u8) and a same-size copy keep the low bytes.
        if source_size == 2 and target_size == 3:
            self._emit_pointer_bytes(src_operand, dest_loc, 2, add_bank=True)
        elif target_size <= source_size:
            self._emit_pointer_bytes(src_operand, dest_loc, target_size)
        else:
            raise InstructionSelectionError(
                f"Unsupported pointer conversion: {source_type} to {target_type}",
                source_loc=self.parent._current_source_loc
            )

    def _emit_pointer_bytes(self, src_operand, dest_loc, count: int,
                            add_bank: bool = False):
        """Copy the low *count* bytes of a pointer to dest, then with
        *add_bank* a bank byte: the ROM bank when the source names ROM data
        (`&TABLE`), else 0.

        Byte by byte in m8: in m16 each LDA/STA widens and the per-byte
        stores overlap. A pointer in X or Y is moved with one 16-bit transfer
        and store - TXA in m8 copies only the low byte, and B would supply a
        stale high one.
        """
        symbol = getattr(src_operand, 'symbol', None)
        rom_label = getattr(symbol, 'rom_label', None) if symbol else None

        if isinstance(src_operand, HardwareRegister):
            src_reg, src_loc = src_operand.name, None
        elif isinstance(src_operand, MIRImmediate):
            src_reg, src_loc = None, None
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            src_reg = src_loc.hw_register if src_loc.is_hw() else None

        def dest_at(i):
            return dest_loc if i == 0 else self.parent._offset_location(dest_loc, i)

        if src_reg in ('X', 'Y'):
            transfer = Opcode.TXA if src_reg == 'X' else Opcode.TYA
            if count >= 2:
                self.parent._ensure_m16_mode()
                self._emit_instr(transfer, comment=f"Pointer from {src_reg}")
                self._emit_load_store('STA', dest_loc)
                self.parent._ensure_m8_mode()
            else:
                self.parent._ensure_m8_mode()
                self._emit_instr(transfer, comment=f"Pointer low byte from {src_reg}")
                self._emit_load_store('STA', dest_loc)
            if count > 2:
                # X and Y hold 16 bits; the bank is not in them.
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x00),
                                 "Bank byte = 0 (register has no bank)")
                self._emit_load_store('STA', dest_at(2))
        else:
            self.parent._ensure_m8_mode()
            for i in range(count):
                if isinstance(src_operand, MIRImmediate):
                    if i == 2 and rom_label:
                        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f":{rom_label}"),
                                         f"Bank byte of {rom_label}")
                    else:
                        self._emit_instr(Opcode.LDA_IMMEDIATE,
                                         Immediate((src_operand.value >> (i * 8)) & 0xFF))
                else:
                    src = src_loc if i == 0 else self.parent._offset_location(src_loc, i)
                    self._emit_load_store('LDA', src)
                self._emit_load_store('STA', dest_at(i))

        if add_bank:
            bank_ref = f":{rom_label}" if rom_label else 0x00
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(bank_ref),
                             f"Bank byte of {rom_label}" if rom_label else "Bank byte = 0")
            self._emit_load_store('STA', dest_at(2))
