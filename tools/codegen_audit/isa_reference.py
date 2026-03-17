# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Compact 65816 ISA reference for Agent 1 context.

Provides a focused instruction set summary organized by category,
including mnemonic, addressing modes, cycle costs, byte sizes, and flag effects.
Designed to fit in an AI prompt without overwhelming context.
"""


def get_isa_reference() -> str:
    """Return a compact 65816 ISA reference string for use in AI prompts."""
    return _ISA_REFERENCE


_ISA_REFERENCE = """\
# 65816 Instruction Set Reference (WLA-DX Syntax)

## Notation
- dp = direct page ($00-$FF), abs = 16-bit absolute, long = 24-bit
- #imm = immediate (size depends on M/X flags)
- Cycles: base cost; +1 if M=0 (16-bit A) for A ops; +1 if page crossed for indexed
- Flags: N=negative, V=overflow, Z=zero, C=carry, M=8/16-bit A, X=8/16-bit XY

## Load/Store
| Mnemonic | Modes | Bytes | Cycles | Flags |
|----------|-------|-------|--------|-------|
| LDA | #imm,dp,dp+X,abs,abs+X,abs+Y,long,long+X,(dp),(dp)+Y,[dp],[dp]+Y,d+S,(d+S)+Y | 2-4 | 2-7 | NZ |
| LDX | #imm,dp,dp+Y,abs,abs+Y | 2-3 | 2-5 | NZ |
| LDY | #imm,dp,dp+X,abs,abs+X | 2-3 | 2-5 | NZ |
| STA | dp,dp+X,abs,abs+X,abs+Y,long,long+X,(dp),(dp)+Y,[dp],[dp]+Y,d+S,(d+S)+Y | 2-4 | 3-7 | - |
| STX | dp,dp+Y,abs | 2-3 | 3-4 | - |
| STY | dp,dp+X,abs | 2-3 | 3-4 | - |
| STZ | dp,dp+X,abs,abs+X | 2-3 | 3-5 | - |

## Arithmetic
| Mnemonic | Description | Modes | Flags |
|----------|-------------|-------|-------|
| ADC | A = A + operand + C | same as LDA | NVZC |
| SBC | A = A - operand - !C | same as LDA | NVZC |
| INC | Increment A or memory | A,dp,dp+X,abs,abs+X | NZ |
| DEC | Decrement A or memory | A,dp,dp+X,abs,abs+X | NZ |
| INX | X++ | implied (1B, 2cy) | NZ |
| INY | Y++ | implied (1B, 2cy) | NZ |
| DEX | X-- | implied (1B, 2cy) | NZ |
| DEY | Y-- | implied (1B, 2cy) | NZ |

## Bitwise
| Mnemonic | Description | Modes | Flags |
|----------|-------------|-------|-------|
| AND | A = A & operand | same as LDA | NZ |
| ORA | A = A | operand | same as LDA | NZ |
| EOR | A = A ^ operand | same as LDA | NZ |
| ASL | Shift left (A or mem) | A,dp,dp+X,abs,abs+X | NZC |
| LSR | Shift right (A or mem) | A,dp,dp+X,abs,abs+X | NZC |
| ROL | Rotate left through C | A,dp,dp+X,abs,abs+X | NZC |
| ROR | Rotate right through C | A,dp,dp+X,abs,abs+X | NZC |
| BIT | Test bits (A & operand) | #imm,dp,dp+X,abs,abs+X | NVZ (imm: Z only) |
| TRB | Test and reset bits | dp,abs | Z |
| TSB | Test and set bits | dp,abs | Z |

## Compare
| Mnemonic | Description | Modes | Cycles | Flags |
|----------|-------------|-------|--------|-------|
| CMP | Compare A | same as LDA | 2-7 | NZC |
| CPX | Compare X | #imm,dp,abs | 2-4 | NZC |
| CPY | Compare Y | #imm,dp,abs | 2-4 | NZC |

## Branch (all 2 bytes, 2-3 cycles)
| Mnemonic | Condition |
|----------|-----------|
| BEQ | Z=1 (equal) |
| BNE | Z=0 (not equal) |
| BCC/BLT | C=0 (less than, unsigned) |
| BCS/BGE | C=1 (greater/equal, unsigned) |
| BMI | N=1 (negative) |
| BPL | N=0 (positive) |
| BVC | V=0 (no overflow) |
| BVS | V=1 (overflow) |
| BRA | Always (2B, 3cy) |
| BRL | Always long (3B, 4cy) |

## Jump/Call
| Mnemonic | Description | Bytes | Cycles |
|----------|-------------|-------|--------|
| JMP abs | Jump absolute | 3 | 3 |
| JMP long | Jump long (24-bit) | 4 | 4 |
| JMP (abs) | Jump indirect | 3 | 5 |
| JMP (abs,X) | Jump indirect indexed | 3 | 6 |
| JSR abs | Call subroutine | 3 | 6 |
| JSL long | Call long (24-bit) | 4 | 8 |
| RTS | Return from subroutine | 1 | 6 |
| RTL | Return long | 1 | 6 |
| RTI | Return from interrupt | 1 | 7 |

## Stack
| Mnemonic | Description | Bytes | Cycles |
|----------|-------------|-------|--------|
| PHA | Push A | 1 | 3 (+1 if M=0) |
| PHX | Push X | 1 | 3 (+1 if X=0) |
| PHY | Push Y | 1 | 3 (+1 if X=0) |
| PHP | Push status | 1 | 3 |
| PHB | Push data bank | 1 | 3 |
| PHD | Push direct page | 1 | 4 |
| PHK | Push program bank | 1 | 3 |
| PLA | Pull A | 1 | 4 (+1 if M=0) |
| PLX | Pull X | 1 | 4 (+1 if X=0) |
| PLY | Pull Y | 1 | 4 (+1 if X=0) |
| PLP | Pull status | 1 | 4 |
| PLB | Pull data bank | 1 | 4 |
| PLD | Pull direct page | 1 | 5 |
| PEA | Push effective abs | 3 | 5 |

**WARNING**: PLA/PLX/PLY/PLB/PLD all set N and Z flags! Use PLP to restore flags.

## Transfer
| Mnemonic | Description | Bytes | Cycles | Flags |
|----------|-------------|-------|--------|-------|
| TAX | A -> X | 1 | 2 | NZ |
| TAY | A -> Y | 1 | 2 | NZ |
| TXA | X -> A | 1 | 2 | NZ |
| TYA | Y -> A | 1 | 2 | NZ |
| TXY | X -> Y | 1 | 2 | NZ |
| TYX | Y -> X | 1 | 2 | NZ |
| TCD | 16-bit A -> D | 1 | 2 | NZ |
| TCS | 16-bit A -> S | 1 | 2 | - |
| TDC | D -> 16-bit A | 1 | 2 | NZ |
| TSC | S -> 16-bit A | 1 | 2 | NZ |
| TSX | S -> X | 1 | 2 | NZ |
| TXS | X -> S | 1 | 2 | - |
| XBA | Swap A high/low | 1 | 3 | NZ |

## Processor Mode (65816-specific)
| Mnemonic | Description | Bytes | Cycles |
|----------|-------------|-------|--------|
| REP #$xx | Clear status bits (enables 16-bit) | 2 | 3 |
| SEP #$xx | Set status bits (enables 8-bit) | 2 | 3 |
| REP #$20 | M=0: 16-bit accumulator | 2 | 3 |
| SEP #$20 | M=1: 8-bit accumulator | 2 | 3 |
| REP #$10 | X=0: 16-bit index registers | 2 | 3 |
| SEP #$10 | X=1: 8-bit index registers | 2 | 3 |

## Flag Control
| CLC | Clear carry (before ADC) | 1 | 2 |
| SEC | Set carry (before SBC) | 1 | 2 |
| CLI | Clear interrupt disable | 1 | 2 |
| SEI | Set interrupt disable | 1 | 2 |

## Block Move (65816)
| MVN src,dst | Move block (forward, C+1 bytes) | 3 | 7/byte |
| MVP src,dst | Move block (backward, C+1 bytes) | 3 | 7/byte |

## Key Optimization Notes
- Direct page ops are 1 cycle faster than absolute (3 vs 4 for loads)
- Stack-relative addressing: LDA d,S is 4 cycles; STA d,S is 4 cycles
- Indirect long indexed [dp],Y: 6-7 cycles (expensive but needed for far pointers)
- INX/DEX/INY/DEY are 2 cycles — prefer over LDA/ADC/STA for loop counters
- TAX/TXA etc. are 2 cycles — cheaper than push/pull (3+4 = 7 cycles)
- BRA (branch always) is cheaper than JMP for short forward/backward jumps
- REP/SEP cost 3 cycles each — minimize mode switches in loops
- 16-bit immediate loads (REP #$20; LDA #$xxxx) add 1 cycle and 1 byte vs 8-bit
"""
