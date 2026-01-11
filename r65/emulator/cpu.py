"""
65816 CPU Emulator with Instruction Implementation.

Inherits from CPU65816Base and implements all 65816 instructions as methods.
"""

from typing import TYPE_CHECKING, Callable, List, Optional
from .cpu_base import CPU65816Base, CPUState, StopExecution, WaitForInterrupt
from . import addressing as addr
from . import operations as ops

if TYPE_CHECKING:
    from .memory import Memory

# Re-export for backwards compatibility
__all__ = ['CPU65816', 'CPUState', 'StopExecution', 'WaitForInterrupt']


class CPU65816(CPU65816Base):
    """
    65816 CPU emulator with full instruction set implementation.

    All instructions are implemented as methods of this class.
    """

    def __init__(self, memory: 'Memory'):
        super().__init__(memory)
        self._instructions: List[Callable[[], int]] = self._build_instruction_table()

    def step(self) -> int:
        """Execute one instruction and return cycles consumed."""
        if self.stopped:
            raise StopExecution("CPU stopped")
        if self.waiting:
            raise WaitForInterrupt("CPU waiting for interrupt")

        opcode = self.fetch_byte()
        handler = self._instructions[opcode]
        cycles = handler()
        self.cycles += cycles
        return cycles

    def _branch(self, condition: bool) -> int:
        """Execute conditional branch."""
        target, extra = addr.relative_8(self)
        if condition:
            self.PC = target
            return 3 + extra
        return 2

    # ============== LOAD INSTRUCTIONS ==============

    def _lda_imm(self) -> int:
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.lda8(self, value)
            return 2
        else:
            ops.lda16(self, value)
            return 3

    def _lda_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _lda_long(self) -> int:
        bank, address, extra = addr.absolute_long(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _lda_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _lda_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _lda_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _lda_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _lda_long_x(self) -> int:
        bank, address, extra = addr.absolute_long_x(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _lda_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _lda_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _lda_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _lda_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _lda_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _lda_sr(self) -> int:
        bank, address, extra = addr.stack_relative(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _lda_sr_ind_y(self) -> int:
        bank, address, extra = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            ops.lda8(self, self.memory.read(bank, address))
            return 7 + extra
        else:
            ops.lda16(self, self.memory.read16(bank, address))
            return 8 + extra

    def _ldx_imm(self) -> int:
        value, _ = addr.immediate_idx(self)
        if self.flag_x:
            ops.ldx8(self, value)
            return 2
        else:
            ops.ldx16(self, value)
            return 3

    def _ldx_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_x:
            ops.ldx8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ldx16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ldx_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_x:
            ops.ldx8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.ldx16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _ldx_dp_y(self) -> int:
        bank, address, extra = addr.direct_y(self)
        if self.flag_x:
            ops.ldx8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ldx16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ldx_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_x:
            ops.ldx8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ldx16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ldy_imm(self) -> int:
        value, _ = addr.immediate_idx(self)
        if self.flag_x:
            ops.ldy8(self, value)
            return 2
        else:
            ops.ldy16(self, value)
            return 3

    def _ldy_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_x:
            ops.ldy8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ldy16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ldy_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_x:
            ops.ldy8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.ldy16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _ldy_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_x:
            ops.ldy8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ldy16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ldy_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_x:
            ops.ldy8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ldy16(self, self.memory.read16(bank, address))
            return 5 + extra

    # ============== STORE INSTRUCTIONS ==============

    def _sta_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 4
        else:
            self.memory.write16(bank, address, self.A)
            return 5

    def _sta_long(self) -> int:
        bank, address, _ = addr.absolute_long(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 5
        else:
            self.memory.write16(bank, address, self.A)
            return 6

    def _sta_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 3 + extra
        else:
            self.memory.write16(bank, address, self.A)
            return 4 + extra

    def _sta_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 4 + extra
        else:
            self.memory.write16(bank, address, self.A)
            return 5 + extra

    def _sta_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 5
        else:
            self.memory.write16(bank, address, self.A)
            return 6

    def _sta_abs_y(self) -> int:
        bank, address, _ = addr.absolute_y(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 5
        else:
            self.memory.write16(bank, address, self.A)
            return 6

    def _sta_long_x(self) -> int:
        bank, address, _ = addr.absolute_long_x(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 5
        else:
            self.memory.write16(bank, address, self.A)
            return 6

    def _sta_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 5 + extra
        else:
            self.memory.write16(bank, address, self.A)
            return 6 + extra

    def _sta_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 6 + extra
        else:
            self.memory.write16(bank, address, self.A)
            return 7 + extra

    def _sta_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 6 + extra
        else:
            self.memory.write16(bank, address, self.A)
            return 7 + extra

    def _sta_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 6 + extra
        else:
            self.memory.write16(bank, address, self.A)
            return 7 + extra

    def _sta_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 6 + extra
        else:
            self.memory.write16(bank, address, self.A)
            return 7 + extra

    def _sta_sr(self) -> int:
        bank, address, _ = addr.stack_relative(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 4
        else:
            self.memory.write16(bank, address, self.A)
            return 5

    def _sta_sr_ind_y(self) -> int:
        bank, address, _ = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            self.memory.write(bank, address, self.A & 0xFF)
            return 7
        else:
            self.memory.write16(bank, address, self.A)
            return 8

    def _stx_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_x:
            self.memory.write(bank, address, self.X & 0xFF)
            return 4
        else:
            self.memory.write16(bank, address, self.X)
            return 5

    def _stx_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_x:
            self.memory.write(bank, address, self.X & 0xFF)
            return 3 + extra
        else:
            self.memory.write16(bank, address, self.X)
            return 4 + extra

    def _stx_dp_y(self) -> int:
        bank, address, extra = addr.direct_y(self)
        if self.flag_x:
            self.memory.write(bank, address, self.X & 0xFF)
            return 4 + extra
        else:
            self.memory.write16(bank, address, self.X)
            return 5 + extra

    def _sty_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_x:
            self.memory.write(bank, address, self.Y & 0xFF)
            return 4
        else:
            self.memory.write16(bank, address, self.Y)
            return 5

    def _sty_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_x:
            self.memory.write(bank, address, self.Y & 0xFF)
            return 3 + extra
        else:
            self.memory.write16(bank, address, self.Y)
            return 4 + extra

    def _sty_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_x:
            self.memory.write(bank, address, self.Y & 0xFF)
            return 4 + extra
        else:
            self.memory.write16(bank, address, self.Y)
            return 5 + extra

    def _stz_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            self.memory.write(bank, address, 0)
            return 4
        else:
            self.memory.write16(bank, address, 0)
            return 5

    def _stz_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            self.memory.write(bank, address, 0)
            return 3 + extra
        else:
            self.memory.write16(bank, address, 0)
            return 4 + extra

    def _stz_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            self.memory.write(bank, address, 0)
            return 4 + extra
        else:
            self.memory.write16(bank, address, 0)
            return 5 + extra

    def _stz_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            self.memory.write(bank, address, 0)
            return 5
        else:
            self.memory.write16(bank, address, 0)
            return 6

    # ============== ARITHMETIC INSTRUCTIONS ==============

    def _adc_imm(self) -> int:
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.adc8(self, value)
            return 2
        else:
            ops.adc16(self, value)
            return 3

    def _adc_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _adc_long(self) -> int:
        bank, address, extra = addr.absolute_long(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _adc_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _adc_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _adc_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _adc_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _adc_long_x(self) -> int:
        bank, address, extra = addr.absolute_long_x(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _adc_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _adc_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _adc_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _adc_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _adc_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _adc_sr(self) -> int:
        bank, address, extra = addr.stack_relative(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _adc_sr_ind_y(self) -> int:
        bank, address, extra = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            ops.adc8(self, self.memory.read(bank, address))
            return 7 + extra
        else:
            ops.adc16(self, self.memory.read16(bank, address))
            return 8 + extra

    def _sbc_imm(self) -> int:
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.sbc8(self, value)
            return 2
        else:
            ops.sbc16(self, value)
            return 3

    def _sbc_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _sbc_long(self) -> int:
        bank, address, extra = addr.absolute_long(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _sbc_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _sbc_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _sbc_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _sbc_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _sbc_long_x(self) -> int:
        bank, address, extra = addr.absolute_long_x(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _sbc_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _sbc_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _sbc_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _sbc_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _sbc_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _sbc_sr(self) -> int:
        bank, address, extra = addr.stack_relative(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _sbc_sr_ind_y(self) -> int:
        bank, address, extra = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            ops.sbc8(self, self.memory.read(bank, address))
            return 7 + extra
        else:
            ops.sbc16(self, self.memory.read16(bank, address))
            return 8 + extra

    # ============== COMPARE INSTRUCTIONS ==============

    def _cmp_imm(self) -> int:
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.cmp8(self, self.A, value)
            return 2
        else:
            ops.cmp16(self, self.A, value)
            return 3

    def _cmp_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 5 + extra

    def _cmp_long(self) -> int:
        bank, address, extra = addr.absolute_long(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 6 + extra

    def _cmp_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 4 + extra

    def _cmp_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 5 + extra

    def _cmp_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 5 + extra

    def _cmp_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 5 + extra

    def _cmp_long_x(self) -> int:
        bank, address, extra = addr.absolute_long_x(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 6 + extra

    def _cmp_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 6 + extra

    def _cmp_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 7 + extra

    def _cmp_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 7 + extra

    def _cmp_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 6 + extra

    def _cmp_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 7 + extra

    def _cmp_sr(self) -> int:
        bank, address, extra = addr.stack_relative(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 5 + extra

    def _cmp_sr_ind_y(self) -> int:
        bank, address, extra = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            ops.cmp8(self, self.A, self.memory.read(bank, address))
            return 7 + extra
        else:
            ops.cmp16(self, self.A, self.memory.read16(bank, address))
            return 8 + extra

    def _cpx_imm(self) -> int:
        value, _ = addr.immediate_idx(self)
        if self.flag_x:
            ops.cmp8(self, self.X, value)
            return 2
        else:
            ops.cmp16(self, self.X, value)
            return 3

    def _cpx_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_x:
            ops.cmp8(self, self.X, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.cmp16(self, self.X, self.memory.read16(bank, address))
            return 5 + extra

    def _cpx_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_x:
            ops.cmp8(self, self.X, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.cmp16(self, self.X, self.memory.read16(bank, address))
            return 4 + extra

    def _cpy_imm(self) -> int:
        value, _ = addr.immediate_idx(self)
        if self.flag_x:
            ops.cmp8(self, self.Y, value)
            return 2
        else:
            ops.cmp16(self, self.Y, value)
            return 3

    def _cpy_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_x:
            ops.cmp8(self, self.Y, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.cmp16(self, self.Y, self.memory.read16(bank, address))
            return 5 + extra

    def _cpy_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_x:
            ops.cmp8(self, self.Y, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.cmp16(self, self.Y, self.memory.read16(bank, address))
            return 4 + extra

    # ============== LOGIC INSTRUCTIONS ==============

    def _and_imm(self) -> int:
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.and8(self, value)
            return 2
        else:
            ops.and16(self, value)
            return 3

    def _and_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _and_long(self) -> int:
        bank, address, extra = addr.absolute_long(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _and_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _and_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _and_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _and_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _and_long_x(self) -> int:
        bank, address, extra = addr.absolute_long_x(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _and_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _and_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _and_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _and_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _and_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _and_sr(self) -> int:
        bank, address, extra = addr.stack_relative(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _and_sr_ind_y(self) -> int:
        bank, address, extra = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            ops.and8(self, self.memory.read(bank, address))
            return 7 + extra
        else:
            ops.and16(self, self.memory.read16(bank, address))
            return 8 + extra

    def _ora_imm(self) -> int:
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.ora8(self, value)
            return 2
        else:
            ops.ora16(self, value)
            return 3

    def _ora_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ora_long(self) -> int:
        bank, address, extra = addr.absolute_long(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _ora_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _ora_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ora_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ora_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ora_long_x(self) -> int:
        bank, address, extra = addr.absolute_long_x(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _ora_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _ora_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _ora_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _ora_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _ora_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _ora_sr(self) -> int:
        bank, address, extra = addr.stack_relative(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _ora_sr_ind_y(self) -> int:
        bank, address, extra = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            ops.ora8(self, self.memory.read(bank, address))
            return 7 + extra
        else:
            ops.ora16(self, self.memory.read16(bank, address))
            return 8 + extra

    def _eor_imm(self) -> int:
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.eor8(self, value)
            return 2
        else:
            ops.eor16(self, value)
            return 3

    def _eor_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _eor_long(self) -> int:
        bank, address, extra = addr.absolute_long(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _eor_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _eor_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _eor_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _eor_abs_y(self) -> int:
        bank, address, extra = addr.absolute_y(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _eor_long_x(self) -> int:
        bank, address, extra = addr.absolute_long_x(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _eor_dp_ind(self) -> int:
        bank, address, extra = addr.direct_indirect(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _eor_dp_ind_long(self) -> int:
        bank, address, extra = addr.direct_indirect_long(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _eor_dp_x_ind(self) -> int:
        bank, address, extra = addr.direct_x_indirect(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _eor_dp_ind_y(self) -> int:
        bank, address, extra = addr.direct_indirect_y(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 5 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 6 + extra

    def _eor_dp_ind_long_y(self) -> int:
        bank, address, extra = addr.direct_indirect_long_y(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 6 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 7 + extra

    def _eor_sr(self) -> int:
        bank, address, extra = addr.stack_relative(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _eor_sr_ind_y(self) -> int:
        bank, address, extra = addr.stack_relative_indirect_y(self)
        if self.flag_m:
            ops.eor8(self, self.memory.read(bank, address))
            return 7 + extra
        else:
            ops.eor16(self, self.memory.read16(bank, address))
            return 8 + extra

    def _bit_imm(self) -> int:
        """BIT immediate - only sets Z flag, not N/V."""
        value, _ = addr.immediate_acc(self)
        if self.flag_m:
            ops.bit8(self, value, set_nv=False)
            return 2
        else:
            ops.bit16(self, value, set_nv=False)
            return 3

    def _bit_abs(self) -> int:
        bank, address, extra = addr.absolute(self)
        if self.flag_m:
            ops.bit8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.bit16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _bit_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            ops.bit8(self, self.memory.read(bank, address))
            return 3 + extra
        else:
            ops.bit16(self, self.memory.read16(bank, address))
            return 4 + extra

    def _bit_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            ops.bit8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.bit16(self, self.memory.read16(bank, address))
            return 5 + extra

    def _bit_abs_x(self) -> int:
        bank, address, extra = addr.absolute_x(self)
        if self.flag_m:
            ops.bit8(self, self.memory.read(bank, address))
            return 4 + extra
        else:
            ops.bit16(self, self.memory.read16(bank, address))
            return 5 + extra

    # ============== SHIFT/ROTATE INSTRUCTIONS ==============

    def _asl_acc(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | ops.asl8(self, self.A & 0xFF)
        else:
            self.A = ops.asl16(self, self.A)
        return 2

    def _asl_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.asl8(self, value))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.asl16(self, value))
            return 8

    def _asl_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.asl8(self, value))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.asl16(self, value))
            return 7 + extra

    def _asl_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.asl8(self, value))
            return 6 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.asl16(self, value))
            return 8 + extra

    def _asl_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.asl8(self, value))
            return 7
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.asl16(self, value))
            return 9

    def _lsr_acc(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | ops.lsr8(self, self.A & 0xFF)
        else:
            self.A = ops.lsr16(self, self.A)
        return 2

    def _lsr_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.lsr8(self, value))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.lsr16(self, value))
            return 8

    def _lsr_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.lsr8(self, value))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.lsr16(self, value))
            return 7 + extra

    def _lsr_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.lsr8(self, value))
            return 6 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.lsr16(self, value))
            return 8 + extra

    def _lsr_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.lsr8(self, value))
            return 7
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.lsr16(self, value))
            return 9

    def _rol_acc(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | ops.rol8(self, self.A & 0xFF)
        else:
            self.A = ops.rol16(self, self.A)
        return 2

    def _rol_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.rol8(self, value))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.rol16(self, value))
            return 8

    def _rol_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.rol8(self, value))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.rol16(self, value))
            return 7 + extra

    def _rol_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.rol8(self, value))
            return 6 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.rol16(self, value))
            return 8 + extra

    def _rol_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.rol8(self, value))
            return 7
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.rol16(self, value))
            return 9

    def _ror_acc(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | ops.ror8(self, self.A & 0xFF)
        else:
            self.A = ops.ror16(self, self.A)
        return 2

    def _ror_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.ror8(self, value))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.ror16(self, value))
            return 8

    def _ror_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.ror8(self, value))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.ror16(self, value))
            return 7 + extra

    def _ror_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.ror8(self, value))
            return 6 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.ror16(self, value))
            return 8 + extra

    def _ror_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.ror8(self, value))
            return 7
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.ror16(self, value))
            return 9

    # ============== INCREMENT/DECREMENT INSTRUCTIONS ==============

    def _inc_acc(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | ops.inc8(self, self.A & 0xFF)
        else:
            self.A = ops.inc16(self, self.A)
        return 2

    def _inc_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.inc8(self, value))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.inc16(self, value))
            return 8

    def _inc_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.inc8(self, value))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.inc16(self, value))
            return 7 + extra

    def _inc_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.inc8(self, value))
            return 6 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.inc16(self, value))
            return 8 + extra

    def _inc_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.inc8(self, value))
            return 7
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.inc16(self, value))
            return 9

    def _dec_acc(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | ops.dec8(self, self.A & 0xFF)
        else:
            self.A = ops.dec16(self, self.A)
        return 2

    def _dec_abs(self) -> int:
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.dec8(self, value))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.dec16(self, value))
            return 8

    def _dec_dp(self) -> int:
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.dec8(self, value))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.dec16(self, value))
            return 7 + extra

    def _dec_dp_x(self) -> int:
        bank, address, extra = addr.direct_x(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.dec8(self, value))
            return 6 + extra
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.dec16(self, value))
            return 8 + extra

    def _dec_abs_x(self) -> int:
        bank, address, _ = addr.absolute_x_no_penalty(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.memory.write(bank, address, ops.dec8(self, value))
            return 7
        else:
            value = self.memory.read16(bank, address)
            self.memory.write16(bank, address, ops.dec16(self, value))
            return 9

    def _inx(self) -> int:
        if self.flag_x:
            self.X = ops.inc8(self, self.X)
        else:
            self.X = ops.inc16(self, self.X)
        return 2

    def _iny(self) -> int:
        if self.flag_x:
            self.Y = ops.inc8(self, self.Y)
        else:
            self.Y = ops.inc16(self, self.Y)
        return 2

    def _dex(self) -> int:
        if self.flag_x:
            self.X = ops.dec8(self, self.X)
        else:
            self.X = ops.dec16(self, self.X)
        return 2

    def _dey(self) -> int:
        if self.flag_x:
            self.Y = ops.dec8(self, self.Y)
        else:
            self.Y = ops.dec16(self, self.Y)
        return 2

    # ============== TRANSFER INSTRUCTIONS ==============

    def _tax(self) -> int:
        if self.flag_x:
            self.X = self.A & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = self.A
            self.set_nz_flags(self.X, True)
        return 2

    def _tay(self) -> int:
        if self.flag_x:
            self.Y = self.A & 0xFF
            self.set_nz_flags(self.Y, False)
        else:
            self.Y = self.A
            self.set_nz_flags(self.Y, True)
        return 2

    def _txa(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | (self.X & 0xFF)
            self.set_nz_flags(self.A & 0xFF, False)
        else:
            self.A = self.X
            self.set_nz_flags(self.A, True)
        return 2

    def _tya(self) -> int:
        if self.flag_m:
            self.A = (self.A & 0xFF00) | (self.Y & 0xFF)
            self.set_nz_flags(self.A & 0xFF, False)
        else:
            self.A = self.Y
            self.set_nz_flags(self.A, True)
        return 2

    def _tsx(self) -> int:
        if self.flag_x:
            self.X = self.SP & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = self.SP
            self.set_nz_flags(self.X, True)
        return 2

    def _txy(self) -> int:
        if self.flag_x:
            self.Y = self.X & 0xFF
            self.set_nz_flags(self.Y, False)
        else:
            self.Y = self.X
            self.set_nz_flags(self.Y, True)
        return 2

    def _tyx(self) -> int:
        if self.flag_x:
            self.X = self.Y & 0xFF
            self.set_nz_flags(self.X, False)
        else:
            self.X = self.Y
            self.set_nz_flags(self.X, True)
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

    # ============== ADDITIONAL STACK INSTRUCTIONS ==============

    def _pha(self) -> int:
        if self.flag_m:
            self.push_byte(self.A & 0xFF)
            return 3
        else:
            self.push_word(self.A)
            return 4

    def _phx(self) -> int:
        if self.flag_x:
            self.push_byte(self.X & 0xFF)
            return 3
        else:
            self.push_word(self.X)
            return 4

    def _phy(self) -> int:
        if self.flag_x:
            self.push_byte(self.Y & 0xFF)
            return 3
        else:
            self.push_word(self.Y)
            return 4

    def _pla(self) -> int:
        if self.flag_m:
            value = self.pull_byte()
            self.A = (self.A & 0xFF00) | value
            self.set_nz_flags(value, False)
            return 4
        else:
            self.A = self.pull_word()
            self.set_nz_flags(self.A, True)
            return 5

    def _plx(self) -> int:
        if self.flag_x:
            self.X = self.pull_byte()
            self.set_nz_flags(self.X, False)
            return 4
        else:
            self.X = self.pull_word()
            self.set_nz_flags(self.X, True)
            return 5

    def _ply(self) -> int:
        if self.flag_x:
            self.Y = self.pull_byte()
            self.set_nz_flags(self.Y, False)
            return 4
        else:
            self.Y = self.pull_word()
            self.set_nz_flags(self.Y, True)
            return 5

    def _plp(self) -> int:
        self.P = self.pull_byte()
        if self.emulation_mode:
            self.P |= 0x30  # M and X always set in emulation mode
        return 4

    def _pld(self) -> int:
        self.D = self.pull_word()
        self.set_nz_flags(self.D, True)
        return 5

    # ============== MODE CONTROL INSTRUCTIONS ==============

    def _sep(self) -> int:
        """Set processor status bits."""
        bits = self.fetch_byte()
        self.P |= bits
        if self.emulation_mode:
            self.P |= 0x30  # M and X always set in emulation mode
        # Truncate X/Y if switching to 8-bit index mode
        if bits & 0x10:  # X flag being set
            self.X &= 0xFF
            self.Y &= 0xFF
        return 3

    def _rep(self) -> int:
        """Reset processor status bits."""
        bits = self.fetch_byte()
        self.P &= ~bits
        if self.emulation_mode:
            self.P |= 0x30  # M and X always set in emulation mode
        return 3

    def _xce(self) -> int:
        """Exchange carry and emulation flags."""
        old_c = self.flag_c
        self.flag_c = self.emulation_mode
        self.emulation_mode = old_c
        if self.emulation_mode:
            self.P |= 0x30  # M and X always set in emulation mode
            self.X &= 0xFF
            self.Y &= 0xFF
            self.SP = 0x0100 | (self.SP & 0xFF)
        return 2

    def _cld(self) -> int:
        """Clear decimal mode."""
        self.flag_d = False
        return 2

    def _nop(self) -> int:
        """No operation."""
        return 2

    def _rti(self) -> int:
        """Return from interrupt."""
        if self.emulation_mode:
            self.P = self.pull_byte()
            self.P |= 0x30  # M and X always set
            self.PC = self.pull_word()
            return 6
        else:
            self.P = self.pull_byte()
            self.PC = self.pull_word()
            self.PBR = self.pull_byte()
            return 7

    # ============== BLOCK MOVE INSTRUCTIONS ==============

    def _mvn(self) -> int:
        """Move block negative (increment addresses)."""
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank

        # Move one byte
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)

        # Increment addresses
        if self.flag_x:
            self.X = (self.X + 1) & 0xFF
            self.Y = (self.Y + 1) & 0xFF
        else:
            self.X = (self.X + 1) & 0xFFFF
            self.Y = (self.Y + 1) & 0xFFFF

        # Decrement count
        self.A = (self.A - 1) & 0xFFFF

        # If not done, repeat instruction
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF

        return 7

    def _mvp(self) -> int:
        """Move block positive (decrement addresses)."""
        dest_bank = self.fetch_byte()
        src_bank = self.fetch_byte()
        self.DBR = dest_bank

        # Move one byte
        value = self.memory.read(src_bank, self.X)
        self.memory.write(dest_bank, self.Y, value)

        # Decrement addresses
        if self.flag_x:
            self.X = (self.X - 1) & 0xFF
            self.Y = (self.Y - 1) & 0xFF
        else:
            self.X = (self.X - 1) & 0xFFFF
            self.Y = (self.Y - 1) & 0xFFFF

        # Decrement count
        self.A = (self.A - 1) & 0xFFFF

        # If not done, repeat instruction
        if self.A != 0xFFFF:
            self.PC = (self.PC - 3) & 0xFFFF

        return 7

    def _jsr_indexed_ind(self) -> int:
        """JSR (addr,X) - Jump to Subroutine Indexed Indirect."""
        base = self.fetch_word()
        self.push_word(self.PC - 1)
        x = self.X & self.idx_mask
        ptr = (base + x) & 0xFFFF
        self.PC = self.memory.read16(self.PBR, ptr)
        return 8

    def _trb_abs(self) -> int:
        """Test and Reset Bits - Absolute."""
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.flag_z = (value & (self.A & 0xFF)) == 0
            self.memory.write(bank, address, value & ~(self.A & 0xFF))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.flag_z = (value & self.A) == 0
            self.memory.write16(bank, address, value & ~self.A)
            return 8

    def _trb_dp(self) -> int:
        """Test and Reset Bits - Direct Page."""
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.flag_z = (value & (self.A & 0xFF)) == 0
            self.memory.write(bank, address, value & ~(self.A & 0xFF))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.flag_z = (value & self.A) == 0
            self.memory.write16(bank, address, value & ~self.A)
            return 7 + extra

    def _tsb_abs(self) -> int:
        """Test and Set Bits - Absolute."""
        bank, address, _ = addr.absolute(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.flag_z = (value & (self.A & 0xFF)) == 0
            self.memory.write(bank, address, value | (self.A & 0xFF))
            return 6
        else:
            value = self.memory.read16(bank, address)
            self.flag_z = (value & self.A) == 0
            self.memory.write16(bank, address, value | self.A)
            return 8

    def _tsb_dp(self) -> int:
        """Test and Set Bits - Direct Page."""
        bank, address, extra = addr.direct(self)
        if self.flag_m:
            value = self.memory.read(bank, address)
            self.flag_z = (value & (self.A & 0xFF)) == 0
            self.memory.write(bank, address, value | (self.A & 0xFF))
            return 5 + extra
        else:
            value = self.memory.read16(bank, address)
            self.flag_z = (value & self.A) == 0
            self.memory.write16(bank, address, value | self.A)
            return 7 + extra

    def _illegal(self) -> int:
        """Illegal/undefined opcode - acts as 1-byte NOP."""
        return 2

    # ============== INSTRUCTION TABLE ==============

    def _build_instruction_table(self) -> List[Callable[[], int]]:
        """Build the opcode dispatch table mapping 0x00-0xFF to handler methods."""
        # Initialize with illegal opcode handler
        table: List[Callable[[], int]] = [self._illegal] * 256

        # 0x00-0x0F
        table[0x00] = self._brk
        table[0x01] = self._ora_dp_x_ind
        table[0x02] = self._cop
        table[0x03] = self._ora_sr
        table[0x04] = self._tsb_dp
        table[0x05] = self._ora_dp
        table[0x06] = self._asl_dp
        table[0x07] = self._ora_dp_ind_long
        table[0x08] = self._php
        table[0x09] = self._ora_imm
        table[0x0A] = self._asl_acc
        table[0x0B] = self._phd
        table[0x0C] = self._tsb_abs
        table[0x0D] = self._ora_abs
        table[0x0E] = self._asl_abs
        table[0x0F] = self._ora_long

        # 0x10-0x1F
        table[0x10] = self._bpl
        table[0x11] = self._ora_dp_ind_y
        table[0x12] = self._ora_dp_ind
        table[0x13] = self._ora_sr_ind_y
        table[0x14] = self._trb_dp
        table[0x15] = self._ora_dp_x
        table[0x16] = self._asl_dp_x
        table[0x17] = self._ora_dp_ind_long_y
        table[0x18] = self._clc
        table[0x19] = self._ora_abs_y
        table[0x1A] = self._inc_acc
        table[0x1B] = self._tcs
        table[0x1C] = self._trb_abs
        table[0x1D] = self._ora_abs_x
        table[0x1E] = self._asl_abs_x
        table[0x1F] = self._ora_long_x

        # 0x20-0x2F
        table[0x20] = self._jsr_abs
        table[0x21] = self._and_dp_x_ind
        table[0x22] = self._jsr_long
        table[0x23] = self._and_sr
        table[0x24] = self._bit_dp
        table[0x25] = self._and_dp
        table[0x26] = self._rol_dp
        table[0x27] = self._and_dp_ind_long
        table[0x28] = self._plp
        table[0x29] = self._and_imm
        table[0x2A] = self._rol_acc
        table[0x2B] = self._pld
        table[0x2C] = self._bit_abs
        table[0x2D] = self._and_abs
        table[0x2E] = self._rol_abs
        table[0x2F] = self._and_long

        # 0x30-0x3F
        table[0x30] = self._bmi
        table[0x31] = self._and_dp_ind_y
        table[0x32] = self._and_dp_ind
        table[0x33] = self._and_sr_ind_y
        table[0x34] = self._bit_dp_x
        table[0x35] = self._and_dp_x
        table[0x36] = self._rol_dp_x
        table[0x37] = self._and_dp_ind_long_y
        table[0x38] = self._sec
        table[0x39] = self._and_abs_y
        table[0x3A] = self._dec_acc
        table[0x3B] = self._tsc
        table[0x3C] = self._bit_abs_x
        table[0x3D] = self._and_abs_x
        table[0x3E] = self._rol_abs_x
        table[0x3F] = self._and_long_x

        # 0x40-0x4F
        table[0x40] = self._rti
        table[0x41] = self._eor_dp_x_ind
        table[0x42] = self._wdm
        table[0x43] = self._eor_sr
        table[0x44] = self._mvp
        table[0x45] = self._eor_dp
        table[0x46] = self._lsr_dp
        table[0x47] = self._eor_dp_ind_long
        table[0x48] = self._pha
        table[0x49] = self._eor_imm
        table[0x4A] = self._lsr_acc
        table[0x4B] = self._phk
        table[0x4C] = self._jmp_abs
        table[0x4D] = self._eor_abs
        table[0x4E] = self._lsr_abs
        table[0x4F] = self._eor_long

        # 0x50-0x5F
        table[0x50] = self._bvc
        table[0x51] = self._eor_dp_ind_y
        table[0x52] = self._eor_dp_ind
        table[0x53] = self._eor_sr_ind_y
        table[0x54] = self._mvn
        table[0x55] = self._eor_dp_x
        table[0x56] = self._lsr_dp_x
        table[0x57] = self._eor_dp_ind_long_y
        table[0x58] = self._cli
        table[0x59] = self._eor_abs_y
        table[0x5A] = self._phy
        table[0x5B] = self._tcd
        table[0x5C] = self._jmp_long
        table[0x5D] = self._eor_abs_x
        table[0x5E] = self._lsr_abs_x
        table[0x5F] = self._eor_long_x

        # 0x60-0x6F
        table[0x60] = self._rts
        table[0x61] = self._adc_dp_x_ind
        table[0x62] = self._per
        table[0x63] = self._adc_sr
        table[0x64] = self._stz_dp
        table[0x65] = self._adc_dp
        table[0x66] = self._ror_dp
        table[0x67] = self._adc_dp_ind_long
        table[0x68] = self._pla
        table[0x69] = self._adc_imm
        table[0x6A] = self._ror_acc
        table[0x6B] = self._rtl
        table[0x6C] = self._jmp_ind
        table[0x6D] = self._adc_abs
        table[0x6E] = self._ror_abs
        table[0x6F] = self._adc_long

        # 0x70-0x7F
        table[0x70] = self._bvs
        table[0x71] = self._adc_dp_ind_y
        table[0x72] = self._adc_dp_ind
        table[0x73] = self._adc_sr_ind_y
        table[0x74] = self._stz_dp_x
        table[0x75] = self._adc_dp_x
        table[0x76] = self._ror_dp_x
        table[0x77] = self._adc_dp_ind_long_y
        table[0x78] = self._sei
        table[0x79] = self._adc_abs_y
        table[0x7A] = self._ply
        table[0x7B] = self._tdc
        table[0x7C] = self._jmp_indexed_ind
        table[0x7D] = self._adc_abs_x
        table[0x7E] = self._ror_abs_x
        table[0x7F] = self._adc_long_x

        # 0x80-0x8F
        table[0x80] = self._bra
        table[0x81] = self._sta_dp_x_ind
        table[0x82] = self._brl
        table[0x83] = self._sta_sr
        table[0x84] = self._sty_dp
        table[0x85] = self._sta_dp
        table[0x86] = self._stx_dp
        table[0x87] = self._sta_dp_ind_long
        table[0x88] = self._dey
        table[0x89] = self._bit_imm
        table[0x8A] = self._txa
        table[0x8B] = self._phb
        table[0x8C] = self._sty_abs
        table[0x8D] = self._sta_abs
        table[0x8E] = self._stx_abs
        table[0x8F] = self._sta_long

        # 0x90-0x9F
        table[0x90] = self._bcc
        table[0x91] = self._sta_dp_ind_y
        table[0x92] = self._sta_dp_ind
        table[0x93] = self._sta_sr_ind_y
        table[0x94] = self._sty_dp_x
        table[0x95] = self._sta_dp_x
        table[0x96] = self._stx_dp_y
        table[0x97] = self._sta_dp_ind_long_y
        table[0x98] = self._tya
        table[0x99] = self._sta_abs_y
        table[0x9A] = self._txs
        table[0x9B] = self._txy
        table[0x9C] = self._stz_abs
        table[0x9D] = self._sta_abs_x
        table[0x9E] = self._stz_abs_x
        table[0x9F] = self._sta_long_x

        # 0xA0-0xAF
        table[0xA0] = self._ldy_imm
        table[0xA1] = self._lda_dp_x_ind
        table[0xA2] = self._ldx_imm
        table[0xA3] = self._lda_sr
        table[0xA4] = self._ldy_dp
        table[0xA5] = self._lda_dp
        table[0xA6] = self._ldx_dp
        table[0xA7] = self._lda_dp_ind_long
        table[0xA8] = self._tay
        table[0xA9] = self._lda_imm
        table[0xAA] = self._tax
        table[0xAB] = self._plb
        table[0xAC] = self._ldy_abs
        table[0xAD] = self._lda_abs
        table[0xAE] = self._ldx_abs
        table[0xAF] = self._lda_long

        # 0xB0-0xBF
        table[0xB0] = self._bcs
        table[0xB1] = self._lda_dp_ind_y
        table[0xB2] = self._lda_dp_ind
        table[0xB3] = self._lda_sr_ind_y
        table[0xB4] = self._ldy_dp_x
        table[0xB5] = self._lda_dp_x
        table[0xB6] = self._ldx_dp_y
        table[0xB7] = self._lda_dp_ind_long_y
        table[0xB8] = self._clv
        table[0xB9] = self._lda_abs_y
        table[0xBA] = self._tsx
        table[0xBB] = self._tyx
        table[0xBC] = self._ldy_abs_x
        table[0xBD] = self._lda_abs_x
        table[0xBE] = self._ldx_abs_y
        table[0xBF] = self._lda_long_x

        # 0xC0-0xCF
        table[0xC0] = self._cpy_imm
        table[0xC1] = self._cmp_dp_x_ind
        table[0xC2] = self._rep
        table[0xC3] = self._cmp_sr
        table[0xC4] = self._cpy_dp
        table[0xC5] = self._cmp_dp
        table[0xC6] = self._dec_dp
        table[0xC7] = self._cmp_dp_ind_long
        table[0xC8] = self._iny
        table[0xC9] = self._cmp_imm
        table[0xCA] = self._dex
        table[0xCB] = self._wai
        table[0xCC] = self._cpy_abs
        table[0xCD] = self._cmp_abs
        table[0xCE] = self._dec_abs
        table[0xCF] = self._cmp_long

        # 0xD0-0xDF
        table[0xD0] = self._bne
        table[0xD1] = self._cmp_dp_ind_y
        table[0xD2] = self._cmp_dp_ind
        table[0xD3] = self._cmp_sr_ind_y
        table[0xD4] = self._pei
        table[0xD5] = self._cmp_dp_x
        table[0xD6] = self._dec_dp_x
        table[0xD7] = self._cmp_dp_ind_long_y
        table[0xD8] = self._cld
        table[0xD9] = self._cmp_abs_y
        table[0xDA] = self._phx
        table[0xDB] = self._stp
        table[0xDC] = self._jmp_ind_long
        table[0xDD] = self._cmp_abs_x
        table[0xDE] = self._dec_abs_x
        table[0xDF] = self._cmp_long_x

        # 0xE0-0xEF
        table[0xE0] = self._cpx_imm
        table[0xE1] = self._sbc_dp_x_ind
        table[0xE2] = self._sep
        table[0xE3] = self._sbc_sr
        table[0xE4] = self._cpx_dp
        table[0xE5] = self._sbc_dp
        table[0xE6] = self._inc_dp
        table[0xE7] = self._sbc_dp_ind_long
        table[0xE8] = self._inx
        table[0xE9] = self._sbc_imm
        table[0xEA] = self._nop
        table[0xEB] = self._xba
        table[0xEC] = self._cpx_abs
        table[0xED] = self._sbc_abs
        table[0xEE] = self._inc_abs
        table[0xEF] = self._sbc_long

        # 0xF0-0xFF
        table[0xF0] = self._beq
        table[0xF1] = self._sbc_dp_ind_y
        table[0xF2] = self._sbc_dp_ind
        table[0xF3] = self._sbc_sr_ind_y
        table[0xF4] = self._pea
        table[0xF5] = self._sbc_dp_x
        table[0xF6] = self._inc_dp_x
        table[0xF7] = self._sbc_dp_ind_long_y
        table[0xF8] = self._sed
        table[0xF9] = self._sbc_abs_y
        table[0xFA] = self._plx
        table[0xFB] = self._xce
        table[0xFC] = self._jsr_indexed_ind
        table[0xFD] = self._sbc_abs_x
        table[0xFE] = self._inc_abs_x
        table[0xFF] = self._sbc_long_x

        return table
