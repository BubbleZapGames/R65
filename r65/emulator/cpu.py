"""
65816 CPU Emulator with Instruction Implementation.

Inherits from CPU65816Base and implements all 65816 instructions as methods.
"""

from typing import TYPE_CHECKING, Callable
from .cpu_base import CPU65816Base, CPUState, StopExecution, WaitForInterrupt
from . import addressing as addr
from . import operations as ops

if TYPE_CHECKING:
    from .memory import Memory

# Re-export for backwards compatibility
__all__ = ['CPU65816', 'CPUState', 'StopExecution', 'WaitForInterrupt']

# Opcodes that depend on flag_m (accumulator size)
M_DEPENDENT_OPS: set[int] = {
    0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x09, 0x0A, 0x0C, 0x0D, 0x0E, 0x0F,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x19, 0x1A, 0x1C, 0x1D, 0x1E, 0x1F,
    0x21, 0x23, 0x24, 0x25, 0x26, 0x27, 0x29, 0x2A, 0x2C, 0x2D, 0x2E, 0x2F,
    0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x39, 0x3A, 0x3C, 0x3D, 0x3E, 0x3F,
    0x41, 0x43, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4D, 0x4E, 0x4F,
    0x51, 0x52, 0x53, 0x55, 0x56, 0x57, 0x59, 0x5D, 0x5E, 0x5F,
    0x61, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6D, 0x6E, 0x6F,
    0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x79, 0x7D, 0x7E, 0x7F,
    0x81, 0x83, 0x85, 0x87, 0x89, 0x8A, 0x8D, 0x8F,
    0x91, 0x92, 0x93, 0x95, 0x97, 0x98, 0x99, 0x9C, 0x9D, 0x9E, 0x9F,
    0xA1, 0xA3, 0xA5, 0xA7, 0xA9, 0xAD, 0xAF,
    0xB1, 0xB2, 0xB3, 0xB5, 0xB7, 0xB9, 0xBD, 0xBF,
    0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCD, 0xCE, 0xCF,
    0xD1, 0xD2, 0xD3, 0xD5, 0xD6, 0xD7, 0xD9, 0xDD, 0xDE, 0xDF,
    0xE1, 0xE3, 0xE5, 0xE6, 0xE7, 0xE9, 0xED, 0xEE, 0xEF,
    0xF1, 0xF2, 0xF3, 0xF5, 0xF6, 0xF7, 0xF9, 0xFB, 0xFD, 0xFE, 0xFF,
}

# Opcodes that depend on flag_x (index register size)
X_DEPENDENT_OPS: set[int] = {
    0x28, 0x40, 0x44, 0x54, 0x5A, 0x7A,
    0x84, 0x86, 0x88, 0x8C, 0x8E, 0x94, 0x96, 0x9B,
    0xA0, 0xA2, 0xA4, 0xA6, 0xA8, 0xAA, 0xAC, 0xAE,
    0xB4, 0xB6, 0xBA, 0xBB, 0xBC, 0xBE,
    0xC0, 0xC2, 0xC4, 0xC8, 0xCA, 0xCC, 0xDA,
    0xE0, 0xE2, 0xE4, 0xE8, 0xEC, 0xFA, 0xFB, 0xFC,
}


class CPU65816(CPU65816Base):
    """
    65816 CPU emulator with full instruction set implementation.

    All instructions are implemented as methods of this class.
    """

    def __init__(self, memory: 'Memory'):
        super().__init__(memory)
        self._instructions = self._build_instruction_table()

    def _branch(self, condition: bool) -> int:
        """Execute conditional branch."""
        target, extra = addr.relative_8(self)
        if condition:
            self.PC = target
            return 3 + extra
        return 2

    # ============== STEP IMPLEMENTATION ==============

    def step(self) -> int:
        """
        Execute one instruction.

        Uses 10-bit lookup table with key format: MXIIIIIIII
        where M=flag_m, X=flag_x, I=opcode (8 bits).

        Returns:
            Number of cycles used.
        """
        if self.stopped:
            raise StopExecution("CPU stopped")
        if self.waiting:
            raise WaitForInterrupt("CPU waiting for interrupt")

        # Fetch opcode
        opcode = self.fetch_byte()

        # Build 10-bit key: MXIIIIIIII
        key = (int(self.flag_m) << 9) | (int(self.flag_x) << 8) | opcode

        # Look up and execute instruction
        handler = self._instructions[key]
        if handler is not None:
            cycles = handler()
            self.cycles += cycles
            return cycles
        else:
            # Unknown opcode - treat as NOP
            self.cycles += 2
            return 2

    def _txs(self) -> int:
        if self.emulation_mode:
            self.SP = 0x0100 | (self.X & 0xFF)
        else:
            self.SP = self.X & 0xFFFF
        return 2


    def _tcd(self) -> int:
        self.D = self.A & 0xFFFF
        self.set_nz_flags(self.D, True)
        return 2


    def _tdc(self) -> int:
        self.A = self.D
        self.set_nz_flags(self.A, True)
        return 2


    def _tcs(self) -> int:
        if self.emulation_mode:
            self.SP = 0x0100 | (self.A & 0xFF)
        else:
            self.SP = self.A & 0xFFFF
        return 2


    def _tsc(self) -> int:
        self.A = self.SP
        self.set_nz_flags(self.A, True)
        return 2


    # ============== STACK INSTRUCTIONS ==============

    def _php(self) -> int:
        self.push_byte(self.P)
        return 3


    def _phb(self) -> int:
        self.push_byte(self.DBR)
        return 3


    def _phd(self) -> int:
        self.push_word(self.D)
        return 4


    def _phk(self) -> int:
        self.push_byte(self.PBR)
        return 3


    def _plb(self) -> int:
        self.DBR = self.pull_byte()
        self.set_nz_flags(self.DBR, False)
        return 4


    def _pea(self) -> int:
        """Push Effective Absolute Address."""
        address = self.fetch_word()
        self.push_word(address)
        return 5


    def _pei(self) -> int:
        """Push Effective Indirect Address."""
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        self.push_word(value)
        return 6 + extra


    def _per(self) -> int:
        """Push Effective PC Relative Address."""
        offset = self.fetch_word()
        if offset & 0x8000:
            offset = offset - 65536
        address = (self.PC + offset) & 0xFFFF
        self.push_word(address)
        return 6


    # ============== ARITHMETIC INSTRUCTIONS ==============

    def _bra(self) -> int:
        target, extra = addr.relative_8(self)
        self.PC = target
        return 3 + extra


    def _brl(self) -> int:
        target, _ = addr.relative_16(self)
        self.PC = target
        return 4


    def _beq(self) -> int:
        return self._branch(self.flag_z)


    def _bne(self) -> int:
        return self._branch(not self.flag_z)


    def _bcs(self) -> int:
        return self._branch(self.flag_c)


    def _bcc(self) -> int:
        return self._branch(not self.flag_c)


    def _bmi(self) -> int:
        return self._branch(self.flag_n)


    def _bpl(self) -> int:
        return self._branch(not self.flag_n)


    def _bvs(self) -> int:
        return self._branch(self.flag_v)


    def _bvc(self) -> int:
        return self._branch(not self.flag_v)


    # ============== JUMP INSTRUCTIONS ==============

    def _jmp_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.PC = address
        # JMP doesn't change PBR
        return 3


    def _jmp_long(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        self.PBR = bank
        self.PC = address
        return 4


    def _jmp_ind(self) -> int:
        bank, address, _ = addr.absolute_indirect(self)
        self.PC = address
        return 5


    def _jmp_ind_long(self) -> int:
        bank, address, _ = addr.absolute_indirect_long(self)
        self.PBR = bank
        self.PC = address
        return 6


    def _jmp_indexed_ind(self) -> int:
        bank, address, _ = addr.absolute_indexed_indirect(self)
        self.PC = address
        return 6


    def _jsr_abs(self) -> int:
        address = self.fetch_word()
        self.push_word(self.PC - 1)
        self.PC = address
        return 6


    def _jsr_long(self) -> int:
        address = self.fetch_word()
        bank = self.fetch_byte()
        self.push_byte(self.PBR)
        self.push_word(self.PC - 1)
        self.PBR = bank
        self.PC = address
        return 8


    def _rts(self) -> int:
        self.PC = (self.pull_word() + 1) & 0xFFFF
        return 6


    def _rtl(self) -> int:
        self.PC = (self.pull_word() + 1) & 0xFFFF
        self.PBR = self.pull_byte()
        return 6


    def _sec(self) -> int:
        self.flag_c = True
        return 2


    def _clc(self) -> int:
        self.flag_c = False
        return 2


    def _sei(self) -> int:
        self.flag_i = True
        return 2


    def _cli(self) -> int:
        self.flag_i = False
        return 2


    def _sed(self) -> int:
        self.flag_d = True
        return 2


    def _clv(self) -> int:
        self.flag_v = False
        return 2


    def _wdm(self) -> int:
        """Reserved for future use - acts as 2-byte NOP."""
        self.fetch_byte()  # Skip signature byte
        return 2


    def _stp(self) -> int:
        """Stop the processor."""
        self.stopped = True
        return 3


    def _wai(self) -> int:
        """Wait for interrupt."""
        self.waiting = True
        return 3


    def _xba(self) -> int:
        """Exchange B and A (swap high and low bytes of A)."""
        hi = (self.A >> 8) & 0xFF
        lo = self.A & 0xFF
        self.A = (lo << 8) | hi
        self.set_nz_flags(hi, False)  # N/Z based on new low byte
        return 3


    def _brk(self) -> int:
        """Software interrupt."""
        self.fetch_byte()  # Signature byte
        if self.emulation_mode:
            self.push_word(self.PC)
            self.push_byte(self.P | 0x10)  # B flag set
            self.flag_i = True
            self.flag_d = False
            self.PC = self.memory.read16(0, 0xFFFE)
        else:
            self.push_byte(self.PBR)
            self.push_word(self.PC)
            self.push_byte(self.P)
            self.flag_i = True
            self.flag_d = False
            self.PBR = 0
            self.PC = self.memory.read16(0, 0xFFE6)
        return 8


    def _cop(self) -> int:
        """Co-processor interrupt."""
        self.fetch_byte()  # Signature byte
        if self.emulation_mode:
            self.push_word(self.PC)
            self.push_byte(self.P)
            self.flag_i = True
            self.flag_d = False
            self.PC = self.memory.read16(0, 0xFFF4)
        else:
            self.push_byte(self.PBR)
            self.push_word(self.PC)
            self.push_byte(self.P)
            self.flag_i = True
            self.flag_d = False
            self.PBR = 0
            self.PC = self.memory.read16(0, 0xFFE4)
        return 8



    # ============== INLINED OPCODE DISPATCH METHODS ==============
    # Generated methods with flag-specific implementations

    def op00(self) -> int:
        return self._brk()

    def op02(self) -> int:
        return self._cop()

    def op08(self) -> int:
        return self._php()

    def op0b(self) -> int:
        return self._phd()

    def op10(self) -> int:
        return self._bpl()

    def op18(self) -> int:
        return self._clc()

    def op1b(self) -> int:
        return self._tcs()

    def op20(self) -> int:
        return self._jsr_abs()

    def op22(self) -> int:
        return self._jsr_long()

    def op2b(self) -> int:
        # PLD - Pull direct page register
        self.D = self.pull_word()
        self.set_nz_flags(self.D, True)
        return 5

    def op30(self) -> int:
        return self._bmi()

    def op38(self) -> int:
        return self._sec()

    def op3b(self) -> int:
        return self._tsc()

    def op42(self) -> int:
        return self._wdm()

    def op4b(self) -> int:
        return self._phk()

    def op4c(self) -> int:
        return self._jmp_abs()

    def op50(self) -> int:
        return self._bvc()

    def op58(self) -> int:
        return self._cli()

    def op5b(self) -> int:
        return self._tcd()

    def op5c(self) -> int:
        return self._jmp_long()

    def op60(self) -> int:
        return self._rts()

    def op62(self) -> int:
        return self._per()

    def op6b(self) -> int:
        return self._rtl()

    def op6c(self) -> int:
        return self._jmp_ind()

    def op70(self) -> int:
        return self._bvs()

    def op78(self) -> int:
        return self._sei()

    def op7b(self) -> int:
        return self._tdc()

    def op7c(self) -> int:
        return self._jmp_indexed_ind()

    def op80(self) -> int:
        return self._bra()

    def op82(self) -> int:
        return self._brl()

    def op8b(self) -> int:
        return self._phb()

    def op90(self) -> int:
        return self._bcc()

    def op9a(self) -> int:
        return self._txs()

    def opab(self) -> int:
        return self._plb()

    def opb0(self) -> int:
        return self._bcs()

    def opb8(self) -> int:
        return self._clv()

    def opcb(self) -> int:
        return self._wai()

    def opd0(self) -> int:
        return self._bne()

    def opd4(self) -> int:
        return self._pei()

    def opd8(self) -> int:
        # CLD - Clear decimal flag
        self.flag_d = False
        return 2

    def opdb(self) -> int:
        return self._stp()

    def opdc(self) -> int:
        return self._jmp_ind_long()

    def opea(self) -> int:
        # NOP - No operation
        return 2

    def opeb(self) -> int:
        return self._xba()

    def opf0(self) -> int:
        return self._beq()

    def opf4(self) -> int:
        return self._pea()

    def opf8(self) -> int:
        return self._sed()

    def op01M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 8 + extra

    def op01M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 7 + extra

    def op03M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 6

    def op03M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 5

    def op04M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        self.flag_z = (self.A & value) == 0
        result = value | self.A
        self.memory.write16(bank, address, result)
        return 8 + extra

    def op04M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        self.flag_z = ((self.A & 0xFF) & value) == 0
        result = value | (self.A & 0xFF)
        self.memory.write(bank, address, result)
        return 6 + extra

    def op05M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 5 + extra

    def op05M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 4 + extra

    def op06M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        result = ops.asl16(self, value)
        self.memory.write16(bank, address, result)
        return 8 + extra

    def op06M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        result = ops.asl8(self, value)
        self.memory.write(bank, address, result)
        return 6 + extra

    def op07M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 8 + extra

    def op07M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 7 + extra

    def op09M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.ora16(self, value)
        return 4

    def op09M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.ora8(self, value)
        return 3

    def op0aM0(self) -> int:
        self.A = ops.asl16(self, self.A)
        return 2

    def op0aM1(self) -> int:
        result = ops.asl8(self, self.A & 0xFF)
        self.A = (self.A & 0xFF00) | result
        return 2

    def op0cM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        self.flag_z = (self.A & value) == 0
        result = value | self.A
        self.memory.write16(bank, address, result)
        return 10

    def op0cM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        self.flag_z = ((self.A & 0xFF) & value) == 0
        result = value | (self.A & 0xFF)
        self.memory.write(bank, address, result)
        return 8

    def op0dM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 6

    def op0dM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 5

    def op0eM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        result = ops.asl16(self, value)
        self.memory.write16(bank, address, result)
        return 10

    def op0eM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        result = ops.asl8(self, value)
        self.memory.write(bank, address, result)
        return 8

    def op0fM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 7

    def op0fM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 6

    def op11M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 7 + extra

    def op11M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 6 + extra

    def op12M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 7 + extra

    def op12M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 6 + extra

    def op13M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 9

    def op13M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 8

    def op14M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        self.flag_z = (self.A & value) == 0
        result = value & ~self.A
        self.memory.write16(bank, address, result & 0xFFFF)
        return 8 + extra

    def op14M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        self.flag_z = ((self.A & 0xFF) & value) == 0
        result = value & ~(self.A & 0xFF)
        self.memory.write(bank, address, result & 0xFF)
        return 6 + extra

    def op15M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 6 + extra

    def op15M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 5 + extra

    def op16M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        result = ops.asl16(self, value)
        self.memory.write16(bank, address, result)
        return 9 + extra

    def op16M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        result = ops.asl8(self, value)
        self.memory.write(bank, address, result)
        return 7 + extra

    def op17M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 8 + extra

    def op17M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 7 + extra

    def op19M0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 6 + extra

    def op19M1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 5 + extra

    def op1aM0(self) -> int:
        self.A = ops.inc16(self, self.A)
        return 2

    def op1aM1(self) -> int:
        result = ops.inc8(self, self.A & 0xFF)
        self.A = (self.A & 0xFF00) | result
        return 2

    def op1cM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        self.flag_z = (self.A & value) == 0
        result = value & ~self.A
        self.memory.write16(bank, address, result & 0xFFFF)
        return 10

    def op1cM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        self.flag_z = ((self.A & 0xFF) & value) == 0
        result = value & ~(self.A & 0xFF)
        self.memory.write(bank, address, result & 0xFF)
        return 8

    def op1dM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 6 + extra

    def op1dM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 5 + extra

    def op1eM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read16(bank, address)
        result = ops.asl16(self, value)
        self.memory.write16(bank, address, result)
        return 11

    def op1eM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read(bank, address)
        result = ops.asl8(self, value)
        self.memory.write(bank, address, result)
        return 9

    def op1fM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read16(bank, address)
        ops.ora16(self, value)
        return 7

    def op1fM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read(bank, address)
        ops.ora8(self, value)
        return 6

    def op21M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 8 + extra

    def op21M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 7 + extra

    def op23M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 6

    def op23M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 5

    def op24M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.bit16(self, value)
        return 5 + extra

    def op24M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.bit8(self, value)
        return 4 + extra

    def op25M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 5 + extra

    def op25M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 4 + extra

    def op26M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        result = ops.rol16(self, value)
        self.memory.write16(bank, address, result)
        return 8 + extra

    def op26M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        result = ops.rol8(self, value)
        self.memory.write(bank, address, result)
        return 6 + extra

    def op27M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 8 + extra

    def op27M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 7 + extra

    def op29M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.and16(self, value)
        return 4

    def op29M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.and8(self, value)
        return 3

    def op2aM0(self) -> int:
        self.A = ops.rol16(self, self.A)
        return 2

    def op2aM1(self) -> int:
        result = ops.rol8(self, self.A & 0xFF)
        self.A = (self.A & 0xFF00) | result
        return 2

    def op2cM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.bit16(self, value)
        return 6

    def op2cM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.bit8(self, value)
        return 5

    def op2dM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 6

    def op2dM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 5

    def op2eM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        result = ops.rol16(self, value)
        self.memory.write16(bank, address, result)
        return 10

    def op2eM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        result = ops.rol8(self, value)
        self.memory.write(bank, address, result)
        return 8

    def op2fM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 7

    def op2fM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 6

    def op31M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 7 + extra

    def op31M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 6 + extra

    def op32M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 7 + extra

    def op32M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 6 + extra

    def op33M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 9

    def op33M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 8

    def op34M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.bit16(self, value)
        return 6 + extra

    def op34M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.bit8(self, value)
        return 5 + extra

    def op35M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 6 + extra

    def op35M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 5 + extra

    def op36M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        result = ops.rol16(self, value)
        self.memory.write16(bank, address, result)
        return 9 + extra

    def op36M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        result = ops.rol8(self, value)
        self.memory.write(bank, address, result)
        return 7 + extra

    def op37M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 8 + extra

    def op37M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 7 + extra

    def op39M0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 6 + extra

    def op39M1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 5 + extra

    def op3aM0(self) -> int:
        self.A = ops.dec16(self, self.A)
        return 2

    def op3aM1(self) -> int:
        result = ops.dec8(self, self.A & 0xFF)
        self.A = (self.A & 0xFF00) | result
        return 2

    def op3cM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.bit16(self, value)
        return 6 + extra

    def op3cM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.bit8(self, value)
        return 5 + extra

    def op3dM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 6 + extra

    def op3dM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 5 + extra

    def op3eM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read16(bank, address)
        result = ops.rol16(self, value)
        self.memory.write16(bank, address, result)
        return 11

    def op3eM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read(bank, address)
        result = ops.rol8(self, value)
        self.memory.write(bank, address, result)
        return 9

    def op3fM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read16(bank, address)
        ops.and16(self, value)
        return 7

    def op3fM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read(bank, address)
        ops.and8(self, value)
        return 6

    def op41M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 8 + extra

    def op41M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 7 + extra

    def op43M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 6

    def op43M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 5

    def op45M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 5 + extra

    def op45M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 4 + extra

    def op46M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        result = ops.lsr16(self, value)
        self.memory.write16(bank, address, result)
        return 8 + extra

    def op46M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        result = ops.lsr8(self, value)
        self.memory.write(bank, address, result)
        return 6 + extra

    def op47M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 8 + extra

    def op47M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 7 + extra

    def op48M0(self) -> int:
        self.push_word(self.A)
        return 4

    def op48M1(self) -> int:
        self.push_byte(self.A & 0xFF)
        return 3

    def op49M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.eor16(self, value)
        return 4

    def op49M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.eor8(self, value)
        return 3

    def op4aM0(self) -> int:
        self.A = ops.lsr16(self, self.A)
        return 2

    def op4aM1(self) -> int:
        result = ops.lsr8(self, self.A & 0xFF)
        self.A = (self.A & 0xFF00) | result
        return 2

    def op4dM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 6

    def op4dM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 5

    def op4eM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        result = ops.lsr16(self, value)
        self.memory.write16(bank, address, result)
        return 10

    def op4eM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        result = ops.lsr8(self, value)
        self.memory.write(bank, address, result)
        return 8

    def op4fM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 7

    def op4fM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 6

    def op51M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 7 + extra

    def op51M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 6 + extra

    def op52M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 7 + extra

    def op52M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 6 + extra

    def op53M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 9

    def op53M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 8

    def op55M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 6 + extra

    def op55M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 5 + extra

    def op56M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        result = ops.lsr16(self, value)
        self.memory.write16(bank, address, result)
        return 9 + extra

    def op56M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        result = ops.lsr8(self, value)
        self.memory.write(bank, address, result)
        return 7 + extra

    def op57M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 8 + extra

    def op57M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 7 + extra

    def op59M0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 6 + extra

    def op59M1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 5 + extra

    def op5dM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 6 + extra

    def op5dM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 5 + extra

    def op5eM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read16(bank, address)
        result = ops.lsr16(self, value)
        self.memory.write16(bank, address, result)
        return 11

    def op5eM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read(bank, address)
        result = ops.lsr8(self, value)
        self.memory.write(bank, address, result)
        return 9

    def op5fM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read16(bank, address)
        ops.eor16(self, value)
        return 7

    def op5fM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read(bank, address)
        ops.eor8(self, value)
        return 6

    def op61M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 8 + extra

    def op61M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 7 + extra

    def op63M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 6

    def op63M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 5

    def op64M0(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write16(bank, address, 0)
        return 5 + extra

    def op64M1(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write(bank, address, 0)
        return 4 + extra

    def op65M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 5 + extra

    def op65M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 4 + extra

    def op66M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        result = ops.ror16(self, value)
        self.memory.write16(bank, address, result)
        return 8 + extra

    def op66M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        result = ops.ror8(self, value)
        self.memory.write(bank, address, result)
        return 6 + extra

    def op67M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 8 + extra

    def op67M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 7 + extra

    def op68M0(self) -> int:
        self.A = self.pull_word()
        self.set_nz_flags(self.A, True)
        return 5

    def op68M1(self) -> int:
        self.A = (self.A & 0xFF00) | self.pull_byte()
        self.set_nz_flags(self.A & 0xFF, False)
        return 4

    def op69M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.adc16(self, value)
        return 4

    def op69M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.adc8(self, value)
        return 3

    def op6aM0(self) -> int:
        self.A = ops.ror16(self, self.A)
        return 2

    def op6aM1(self) -> int:
        result = ops.ror8(self, self.A & 0xFF)
        self.A = (self.A & 0xFF00) | result
        return 2

    def op6dM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 6

    def op6dM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 5

    def op6eM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        result = ops.ror16(self, value)
        self.memory.write16(bank, address, result)
        return 10

    def op6eM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        result = ops.ror8(self, value)
        self.memory.write(bank, address, result)
        return 8

    def op6fM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 7

    def op6fM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 6

    def op71M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 7 + extra

    def op71M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 6 + extra

    def op72M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 7 + extra

    def op72M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 6 + extra

    def op73M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 9

    def op73M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 8

    def op74M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        self.memory.write16(bank, address, 0)
        return 6 + extra

    def op74M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        self.memory.write(bank, address, 0)
        return 5 + extra

    def op75M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 6 + extra

    def op75M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 5 + extra

    def op76M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        result = ops.ror16(self, value)
        self.memory.write16(bank, address, result)
        return 9 + extra

    def op76M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        result = ops.ror8(self, value)
        self.memory.write(bank, address, result)
        return 7 + extra

    def op77M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 8 + extra

    def op77M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 7 + extra

    def op79M0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 6 + extra

    def op79M1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 5 + extra

    def op7dM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 6 + extra

    def op7dM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 5 + extra

    def op7eM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read16(bank, address)
        result = ops.ror16(self, value)
        self.memory.write16(bank, address, result)
        return 11

    def op7eM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read(bank, address)
        result = ops.ror8(self, value)
        self.memory.write(bank, address, result)
        return 9

    def op7fM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read16(bank, address)
        ops.adc16(self, value)
        return 7

    def op7fM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read(bank, address)
        ops.adc8(self, value)
        return 6

    def op81M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 8 + extra

    def op81M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 7 + extra

    def op83M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 6

    def op83M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 5

    def op85M0(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 5 + extra

    def op85M1(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 4 + extra

    def op87M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 8 + extra

    def op87M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 7 + extra

    def op89M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.bit16(self, value, set_nv=False)
        return 4

    def op89M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.bit8(self, value, set_nv=False)
        return 3

    def op8aM0(self) -> int:
        self.A = self.X & 0xFFFF
        self.set_nz_flags(self.A, True)
        return 2

    def op8aM1(self) -> int:
        self.A = (self.A & 0xFF00) | (self.X & 0xFF)
        self.set_nz_flags(self.A & 0xFF, False)
        return 2

    def op8dM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 6

    def op8dM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 5

    def op8fM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 7

    def op8fM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 6

    def op91M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 8 + extra

    def op91M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 7 + extra

    def op92M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 7 + extra

    def op92M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 6 + extra

    def op93M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 9

    def op93M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 8

    def op95M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 6 + extra

    def op95M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 5 + extra

    def op97M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 8 + extra

    def op97M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 7 + extra

    def op98M0(self) -> int:
        self.A = self.Y & 0xFFFF
        self.set_nz_flags(self.A, True)
        return 2

    def op98M1(self) -> int:
        self.A = (self.A & 0xFF00) | (self.Y & 0xFF)
        self.set_nz_flags(self.A & 0xFF, False)
        return 2

    def op99M0(self) -> int:
        bank, address, _ = addr.absolute_y(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 7

    def op99M1(self) -> int:
        bank, address, _ = addr.absolute_y(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 6

    def op9cM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write16(bank, address, 0)
        return 6

    def op9cM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write(bank, address, 0)
        return 5

    def op9dM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 7

    def op9dM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 6

    def op9eM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        self.memory.write16(bank, address, 0)
        return 7

    def op9eM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        self.memory.write(bank, address, 0)
        return 6

    def op9fM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        self.memory.write16(bank, address, self.A & 0xFFFF)
        return 7

    def op9fM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        self.memory.write(bank, address, self.A & 0xFF)
        return 6

    def opa1M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 8 + extra

    def opa1M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 7 + extra

    def opa3M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 6

    def opa3M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 5

    def opa5M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 5 + extra

    def opa5M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 4 + extra

    def opa7M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 8 + extra

    def opa7M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 7 + extra

    def opa9M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.lda16(self, value)
        return 4

    def opa9M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.lda8(self, value)
        return 3

    def opadM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 6

    def opadM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 5

    def opafM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 7

    def opafM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 6

    def opb1M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 7 + extra

    def opb1M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 6 + extra

    def opb2M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 7 + extra

    def opb2M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 6 + extra

    def opb3M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 9

    def opb3M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 8

    def opb5M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 6 + extra

    def opb5M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 5 + extra

    def opb7M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 8 + extra

    def opb7M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 7 + extra

    def opb9M0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 6 + extra

    def opb9M1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 5 + extra

    def opbdM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 6 + extra

    def opbdM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 5 + extra

    def opbfM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read16(bank, address)
        ops.lda16(self, value)
        return 7

    def opbfM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read(bank, address)
        ops.lda8(self, value)
        return 6

    def opc1M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 8 + extra

    def opc1M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 7 + extra

    def opc3M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 6

    def opc3M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 5

    def opc5M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 5 + extra

    def opc5M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 4 + extra

    def opc6M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        result = ops.dec16(self, value)
        self.memory.write16(bank, address, result)
        return 8 + extra

    def opc6M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        result = ops.dec8(self, value)
        self.memory.write(bank, address, result)
        return 6 + extra

    def opc7M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 8 + extra

    def opc7M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 7 + extra

    def opc9M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.cmp16(self, self.A, value)
        return 4

    def opc9M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.cmp8(self, self.A, value)
        return 3

    def opcdM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 6

    def opcdM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 5

    def opceM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        result = ops.dec16(self, value)
        self.memory.write16(bank, address, result)
        return 10

    def opceM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        result = ops.dec8(self, value)
        self.memory.write(bank, address, result)
        return 8

    def opcfM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 7

    def opcfM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 6

    def opd1M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 7 + extra

    def opd1M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 6 + extra

    def opd2M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 7 + extra

    def opd2M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 6 + extra

    def opd3M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 9

    def opd3M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 8

    def opd5M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 6 + extra

    def opd5M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 5 + extra

    def opd6M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        result = ops.dec16(self, value)
        self.memory.write16(bank, address, result)
        return 9 + extra

    def opd6M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        result = ops.dec8(self, value)
        self.memory.write(bank, address, result)
        return 7 + extra

    def opd7M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 8 + extra

    def opd7M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 7 + extra

    def opd9M0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 6 + extra

    def opd9M1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 5 + extra

    def opddM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 6 + extra

    def opddM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 5 + extra

    def opdeM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read16(bank, address)
        result = ops.dec16(self, value)
        self.memory.write16(bank, address, result)
        return 11

    def opdeM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read(bank, address)
        result = ops.dec8(self, value)
        self.memory.write(bank, address, result)
        return 9

    def opdfM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.A, value)
        return 7

    def opdfM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.A, value)
        return 6

    def ope1M0(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 8 + extra

    def ope1M1(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 7 + extra

    def ope3M0(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 6

    def ope3M1(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 5

    def ope5M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 5 + extra

    def ope5M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 4 + extra

    def ope6M0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        result = ops.inc16(self, value)
        self.memory.write16(bank, address, result)
        return 8 + extra

    def ope6M1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        result = ops.inc8(self, value)
        self.memory.write(bank, address, result)
        return 6 + extra

    def ope7M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 8 + extra

    def ope7M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 7 + extra

    def ope9M0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.sbc16(self, value)
        return 4

    def ope9M1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.sbc8(self, value)
        return 3

    def opedM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 6

    def opedM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 5

    def opeeM0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        result = ops.inc16(self, value)
        self.memory.write16(bank, address, result)
        return 10

    def opeeM1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        result = ops.inc8(self, value)
        self.memory.write(bank, address, result)
        return 8

    def opefM0(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 7

    def opefM1(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 6

    def opf1M0(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 7 + extra

    def opf1M1(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 6 + extra

    def opf2M0(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 7 + extra

    def opf2M1(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 6 + extra

    def opf3M0(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 9

    def opf3M1(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 8

    def opf5M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 6 + extra

    def opf5M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 5 + extra

    def opf6M0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        result = ops.inc16(self, value)
        self.memory.write16(bank, address, result)
        return 9 + extra

    def opf6M1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        result = ops.inc8(self, value)
        self.memory.write(bank, address, result)
        return 7 + extra

    def opf7M0(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 8 + extra

    def opf7M1(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 7 + extra

    def opf9M0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 6 + extra

    def opf9M1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 5 + extra

    def opfdM0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 6 + extra

    def opfdM1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 5 + extra

    def opfeM0(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read16(bank, address)
        result = ops.inc16(self, value)
        self.memory.write16(bank, address, result)
        return 11

    def opfeM1(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        value = self.memory.read(bank, address)
        result = ops.inc8(self, value)
        self.memory.write(bank, address, result)
        return 9

    def opffM0(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read16(bank, address)
        ops.sbc16(self, value)
        return 7

    def opffM1(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        value = self.memory.read(bank, address)
        ops.sbc8(self, value)
        return 6

    def op28X0(self) -> int:
        self.P = self.pull_byte()
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 4

    def op28X1(self) -> int:
        self.P = self.pull_byte()
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 4

    def op40X0(self) -> int:
        self.P = self.pull_byte()
        self.PC = self.pull_word()
        if not self.emulation_mode:
            self.PBR = self.pull_byte()
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 7 if self.emulation_mode else 8

    def op40X1(self) -> int:
        self.P = self.pull_byte()
        self.PC = self.pull_word()
        if not self.emulation_mode:
            self.PBR = self.pull_byte()
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 7 if self.emulation_mode else 8

    def op5aX0(self) -> int:
        self.push_word(self.Y)
        return 4

    def op5aX1(self) -> int:
        self.push_byte(self.Y & 0xFF)
        return 3

    def op7aX0(self) -> int:
        self.Y = self.pull_word()
        self.set_nz_flags(self.Y, True)
        return 5

    def op7aX1(self) -> int:
        self.Y = self.pull_byte()
        self.set_nz_flags(self.Y, False)
        return 4

    def op84X0(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write16(bank, address, self.Y & 0xFFFF)
        return 5 + extra

    def op84X1(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write(bank, address, self.Y & 0xFF)
        return 4 + extra

    def op86X0(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write16(bank, address, self.X & 0xFFFF)
        return 5 + extra

    def op86X1(self) -> int:
        bank, address, extra = addr.direct(self)
        self.memory.write(bank, address, self.X & 0xFF)
        return 4 + extra

    def op88X0(self) -> int:
        self.Y = ops.dec16(self, self.Y)
        return 2

    def op88X1(self) -> int:
        self.Y = ops.dec8(self, self.Y)
        return 2

    def op8cX0(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write16(bank, address, self.Y & 0xFFFF)
        return 6

    def op8cX1(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write(bank, address, self.Y & 0xFF)
        return 5

    def op8eX0(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write16(bank, address, self.X & 0xFFFF)
        return 6

    def op8eX1(self) -> int:
        bank, address, _ = addr.absolute(self)
        self.memory.write(bank, address, self.X & 0xFF)
        return 5

    def op94X0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        self.memory.write16(bank, address, self.Y & 0xFFFF)
        return 6 + extra

    def op94X1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        self.memory.write(bank, address, self.Y & 0xFF)
        return 5 + extra

    def op96X0(self) -> int:
        bank, address, extra = addr.direct_y(self)
        self.memory.write16(bank, address, self.X & 0xFFFF)
        return 6 + extra

    def op96X1(self) -> int:
        bank, address, extra = addr.direct_y(self)
        self.memory.write(bank, address, self.X & 0xFF)
        return 5 + extra

    def op9bX0(self) -> int:
        self.Y = self.X
        self.set_nz_flags(self.Y, True)
        return 2

    def op9bX1(self) -> int:
        self.Y = self.X & 0xFF
        self.set_nz_flags(self.Y, False)
        return 2

    def opa0X0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.ldy16(self, value)
        return 4

    def opa0X1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.ldy8(self, value)
        return 3

    def opa2X0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.ldx16(self, value)
        return 4

    def opa2X1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.ldx8(self, value)
        return 3

    def opa4X0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.ldy16(self, value)
        return 5 + extra

    def opa4X1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.ldy8(self, value)
        return 4 + extra

    def opa6X0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.ldx16(self, value)
        return 5 + extra

    def opa6X1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.ldx8(self, value)
        return 4 + extra

    def opa8X0(self) -> int:
        self.Y = self.A & 0xFFFF
        self.set_nz_flags(self.Y, True)
        return 2

    def opa8X1(self) -> int:
        self.Y = self.A & 0xFF
        self.set_nz_flags(self.Y, False)
        return 2

    def opaaX0(self) -> int:
        self.X = self.A & 0xFFFF
        self.set_nz_flags(self.X, True)
        return 2

    def opaaX1(self) -> int:
        self.X = self.A & 0xFF
        self.set_nz_flags(self.X, False)
        return 2

    def opacX0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.ldy16(self, value)
        return 6

    def opacX1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.ldy8(self, value)
        return 5

    def opaeX0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.ldx16(self, value)
        return 6

    def opaeX1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.ldx8(self, value)
        return 5

    def opb4X0(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read16(bank, address)
        ops.ldy16(self, value)
        return 6 + extra

    def opb4X1(self) -> int:
        bank, address, extra = addr.direct_x(self)
        value = self.memory.read(bank, address)
        ops.ldy8(self, value)
        return 5 + extra

    def opb6X0(self) -> int:
        bank, address, extra = addr.direct_y(self)
        value = self.memory.read16(bank, address)
        ops.ldx16(self, value)
        return 6 + extra

    def opb6X1(self) -> int:
        bank, address, extra = addr.direct_y(self)
        value = self.memory.read(bank, address)
        ops.ldx8(self, value)
        return 5 + extra

    def opbaX0(self) -> int:
        self.X = self.SP
        self.set_nz_flags(self.X, True)
        return 2

    def opbaX1(self) -> int:
        self.X = self.SP & 0xFF
        self.set_nz_flags(self.X, False)
        return 2

    def opbbX0(self) -> int:
        self.X = self.Y
        self.set_nz_flags(self.X, True)
        return 2

    def opbbX1(self) -> int:
        self.X = self.Y & 0xFF
        self.set_nz_flags(self.X, False)
        return 2

    def opbcX0(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read16(bank, address)
        ops.ldy16(self, value)
        return 6 + extra

    def opbcX1(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        value = self.memory.read(bank, address)
        ops.ldy8(self, value)
        return 5 + extra

    def opbeX0(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read16(bank, address)
        ops.ldx16(self, value)
        return 6 + extra

    def opbeX1(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        value = self.memory.read(bank, address)
        ops.ldx8(self, value)
        return 5 + extra

    def opc0X0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.cmp16(self, self.Y, value)
        return 4

    def opc0X1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.cmp8(self, self.Y, value)
        return 3

    def opc4X0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.Y, value)
        return 5 + extra

    def opc4X1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.Y, value)
        return 4 + extra

    def opc8X0(self) -> int:
        self.Y = ops.inc16(self, self.Y)
        return 2

    def opc8X1(self) -> int:
        self.Y = ops.inc8(self, self.Y)
        return 2

    def opcaX0(self) -> int:
        self.X = ops.dec16(self, self.X)
        return 2

    def opcaX1(self) -> int:
        self.X = ops.dec8(self, self.X)
        return 2

    def opccX0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.Y, value)
        return 6

    def opccX1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.Y, value)
        return 5

    def opdaX0(self) -> int:
        self.push_word(self.X)
        return 4

    def opdaX1(self) -> int:
        self.push_byte(self.X & 0xFF)
        return 3

    def ope0X0(self) -> int:
        value, _ = addr.immediate_16(self)
        ops.cmp16(self, self.X, value)
        return 4

    def ope0X1(self) -> int:
        value, _ = addr.immediate_8(self)
        ops.cmp8(self, self.X, value)
        return 3

    def ope4X0(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.X, value)
        return 5 + extra

    def ope4X1(self) -> int:
        bank, address, extra = addr.direct(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.X, value)
        return 4 + extra

    def ope8X0(self) -> int:
        self.X = ops.inc16(self, self.X)
        return 2

    def ope8X1(self) -> int:
        self.X = ops.inc8(self, self.X)
        return 2

    def opecX0(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read16(bank, address)
        ops.cmp16(self, self.X, value)
        return 6

    def opecX1(self) -> int:
        bank, address, _ = addr.absolute(self)
        value = self.memory.read(bank, address)
        ops.cmp8(self, self.X, value)
        return 5

    def opfaX0(self) -> int:
        self.X = self.pull_word()
        self.set_nz_flags(self.X, True)
        return 5

    def opfaX1(self) -> int:
        self.X = self.pull_byte()
        self.set_nz_flags(self.X, False)
        return 4

    def opfcX0(self) -> int:
        base = self.fetch_word()
        self.push_word(self.PC - 1)
        x = self.X & 0xFFFF
        ptr = (base + x) & 0xFFFF
        address = self.memory.read16(self.PBR, ptr)
        self.PC = address
        return 8

    def opfcX1(self) -> int:
        base = self.fetch_word()
        self.push_word(self.PC - 1)
        x = self.X & 0xFF
        ptr = (base + x) & 0xFFFF
        address = self.memory.read16(self.PBR, ptr)
        self.PC = address
        return 8

    def op44M0X0(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X - 1) & 0xFFFF
        self.Y = (self.Y - 1) & 0xFFFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def op44M0X1(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X - 1) & 0xFF
        self.Y = (self.Y - 1) & 0xFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def op44M1X0(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X - 1) & 0xFFFF
        self.Y = (self.Y - 1) & 0xFFFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def op44M1X1(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X - 1) & 0xFF
        self.Y = (self.Y - 1) & 0xFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def op54M0X0(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X + 1) & 0xFFFF
        self.Y = (self.Y + 1) & 0xFFFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def op54M0X1(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X + 1) & 0xFF
        self.Y = (self.Y + 1) & 0xFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def op54M1X0(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X + 1) & 0xFFFF
        self.Y = (self.Y + 1) & 0xFFFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def op54M1X1(self) -> int:
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)
        self.X = (self.X + 1) & 0xFF
        self.Y = (self.Y + 1) & 0xFF
        self.A = (self.A - 1) & 0xFFFF
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF
        return 7

    def opc2M0X0(self) -> int:
        value = self.fetch_byte()
        self.P &= ~value
        return 3

    def opc2M0X1(self) -> int:
        value = self.fetch_byte()
        self.P &= ~value
        return 3

    def opc2M1X0(self) -> int:
        value = self.fetch_byte()
        self.P &= ~value
        return 3

    def opc2M1X1(self) -> int:
        value = self.fetch_byte()
        self.P &= ~value
        return 3

    def ope2M0X0(self) -> int:
        value = self.fetch_byte()
        self.P |= value
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 3

    def ope2M0X1(self) -> int:
        value = self.fetch_byte()
        self.P |= value
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 3

    def ope2M1X0(self) -> int:
        value = self.fetch_byte()
        self.P |= value
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 3

    def ope2M1X1(self) -> int:
        value = self.fetch_byte()
        self.P |= value
        if self.flag_x:
            self.X &= 0xFF
            self.Y &= 0xFF
        return 3

    def opfbM0X0(self) -> int:
        new_emulation = self.flag_c
        self.flag_c = self.emulation_mode
        self.emulation_mode = new_emulation
        if self.emulation_mode:
            self.flag_m = True
            self.flag_x = True
            self.X &= 0xFF
            self.Y &= 0xFF
            self.SP = 0x0100 | (self.SP & 0xFF)
        return 2

    def opfbM0X1(self) -> int:
        new_emulation = self.flag_c
        self.flag_c = self.emulation_mode
        self.emulation_mode = new_emulation
        if self.emulation_mode:
            self.flag_m = True
            self.flag_x = True
            self.X &= 0xFF
            self.Y &= 0xFF
            self.SP = 0x0100 | (self.SP & 0xFF)
        return 2

    def opfbM1X0(self) -> int:
        new_emulation = self.flag_c
        self.flag_c = self.emulation_mode
        self.emulation_mode = new_emulation
        if self.emulation_mode:
            self.flag_m = True
            self.flag_x = True
            self.X &= 0xFF
            self.Y &= 0xFF
            self.SP = 0x0100 | (self.SP & 0xFF)
        return 2

    def opfbM1X1(self) -> int:
        new_emulation = self.flag_c
        self.flag_c = self.emulation_mode
        self.emulation_mode = new_emulation
        if self.emulation_mode:
            self.flag_m = True
            self.flag_x = True
            self.X &= 0xFF
            self.Y &= 0xFF
            self.SP = 0x0100 | (self.SP & 0xFF)
        return 2


    # ============== INSTRUCTION TABLE ==============

    def _build_instruction_table(self) -> list[Callable[[], int] | None]:
        """
        Build 1024-entry instruction lookup table.

        Key format: MXIIIIIIII (10 bits)
        - M: flag_m (bit 9)
        - X: flag_x (bit 8)
        - I: opcode (bits 7-0)

        References dispatch methods named based on flag dependencies:
        - Uses M only: op{op:02x}M{m}
        - Uses X only: op{op:02x}X{x}
        - Uses both M and X: op{op:02x}M{m}X{x}
        - Uses neither: op{op:02x}
        """
        table: list[Callable[[], int] | None] = [None] * 1024

        for opcode in range(256):
            uses_m = opcode in M_DEPENDENT_OPS
            uses_x = opcode in X_DEPENDENT_OPS

            if uses_m and uses_x:
                # Both M and X: 4 dispatch methods
                for m in [0, 1]:
                    for x in [0, 1]:
                        key = (m << 9) | (x << 8) | opcode
                        dispatch_name = f'op{opcode:02x}M{m}X{x}'
                        table[key] = getattr(self, dispatch_name, None)
            elif uses_m:
                # M only: 2 dispatch methods, shared across X values
                for m in [0, 1]:
                    dispatch_method = getattr(self, f'op{opcode:02x}M{m}', None)
                    for x in [0, 1]:
                        key = (m << 9) | (x << 8) | opcode
                        table[key] = dispatch_method
            elif uses_x:
                # X only: 2 dispatch methods, shared across M values
                for x in [0, 1]:
                    dispatch_method = getattr(self, f'op{opcode:02x}X{x}', None)
                    for m in [0, 1]:
                        key = (m << 9) | (x << 8) | opcode
                        table[key] = dispatch_method
            else:
                # Neither: single dispatch method for all M/X combinations
                dispatch_method = getattr(self, f'op{opcode:02x}', None)
                for m in [0, 1]:
                    for x in [0, 1]:
                        key = (m << 9) | (x << 8) | opcode
                        table[key] = dispatch_method

        return table
