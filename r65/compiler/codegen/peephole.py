"""
Peephole optimization for R65 assembly code.

Applies local optimizations to instruction sequences to eliminate
redundant operations and improve code quality.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Instruction:
    """Represents a single assembly instruction."""
    opcode: str
    operand: Optional[str] = None
    comment: Optional[str] = None

    def __str__(self):
        # Check if this is a directive or label (no indentation needed)
        if (self.opcode.startswith('.') or
            self.opcode.endswith(':') or
            self.opcode.startswith(';') or
            not self.opcode or
            # Special SNES header keywords (no indentation)
            self.opcode in ('ID', 'NAME', 'LOROM', 'HIROM', 'FASTROM',
                           'CARTRIDGETYPE', 'ROMSIZE', 'SRAMSIZE', 'COUNTRY',
                           'LICENSEECODE', 'VERSION',
                           # Vector keywords (no indentation)
                           'COP', 'BRK', 'ABORT', 'NMI', 'IRQ', 'RESET', 'IRQBRK')):
            # No indentation for directives, labels, and special keywords
            result = self.opcode
        else:
            # Regular instruction - add indentation
            result = f"    {self.opcode}"

        if self.operand:
            result += f" {self.operand}"
        if self.comment:
            result += f"  ; {self.comment}"
        return result


class PeepholeOptimizer:
    """
    Peephole optimizer for 65816 assembly code.

    Performs local optimizations on instruction sequences:
    - Eliminates redundant load after store (STA $XX; LDA $XX)
    - Removes dead stores
    - Tracks accumulator contents to eliminate unnecessary loads
    """

    def __init__(self):
        self.optimizations_applied = 0

    def optimize(self, instructions: List[str]) -> List[str]:
        """
        Apply peephole optimizations to instruction list.

        Args:
            instructions: List of assembly instruction strings

        Returns:
            Optimized instruction list
        """
        # Parse instructions
        parsed = self._parse_instructions(instructions)

        # Apply optimization passes
        parsed = self._eliminate_redundant_load_after_store(parsed)
        parsed = self._eliminate_dead_stores(parsed)

        # Convert back to strings
        return self._unparse_instructions(parsed)

    def _parse_instructions(self, instructions: List[str]) -> List[Instruction]:
        """Parse instruction strings into Instruction objects."""
        parsed = []
        for line in instructions:
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('.') or line.endswith(':'):
                # Keep labels, directives, and comments as-is
                parsed.append(Instruction(opcode=line))
                continue

            # Split instruction into opcode, operand, and comment
            parts = line.split(';', 1)
            instr_part = parts[0].strip()
            comment = parts[1].strip() if len(parts) > 1 else None

            # Split opcode and operand
            instr_tokens = instr_part.split(None, 1)
            opcode = instr_tokens[0] if instr_tokens else ""
            operand = instr_tokens[1] if len(instr_tokens) > 1 else None

            parsed.append(Instruction(opcode=opcode, operand=operand, comment=comment))

        return parsed

    def _unparse_instructions(self, instructions: List[Instruction]) -> List[str]:
        """Convert Instruction objects back to strings."""
        result = []
        for instr in instructions:
            # Handle labels, directives, comments specially
            if not instr.operand and not instr.comment and (
                instr.opcode.startswith(';') or
                instr.opcode.startswith('.') or
                instr.opcode.endswith(':') or
                not instr.opcode
            ):
                result.append(instr.opcode)
            else:
                result.append(str(instr))
        return result

    def _eliminate_redundant_load_after_store(self, instructions: List[Instruction]) -> List[Instruction]:
        """
        Eliminate redundant LDA immediately after STA to same location.

        Pattern: STA $XX; LDA $XX -> STA $XX

        After storing A to memory, the value is still in A, so loading it back is redundant.
        """
        optimized = []
        i = 0

        while i < len(instructions):
            instr = instructions[i]

            # Check for STA followed by LDA to same address
            if (instr.opcode == "STA" and i + 1 < len(instructions)):
                next_instr = instructions[i + 1]

                if (next_instr.opcode == "LDA" and
                    instr.operand == next_instr.operand):
                    # Redundant load after store - skip the LDA
                    optimized.append(instr)
                    i += 2  # Skip both instructions (we keep STA, skip LDA)
                    self.optimizations_applied += 1
                    continue

            optimized.append(instr)
            i += 1

        return optimized

    def _eliminate_dead_stores(self, instructions: List[Instruction]) -> List[Instruction]:
        """
        Eliminate dead stores that are immediately overwritten.

        Pattern: STA $XX; ... (no read of $XX); STA $XX -> ... ; STA $XX

        If a value is stored but then overwritten before being read, the first store is dead.
        """
        optimized = []
        i = 0

        while i < len(instructions):
            instr = instructions[i]

            # Check for STA followed by another STA to same location (without reads between)
            if instr.opcode == "STA" and i + 1 < len(instructions):
                store_addr = instr.operand

                # Skip indexed addressing - if the address uses ,X or ,Y, we can't reliably
                # determine if two stores are to the same location since the index may change
                if store_addr and (',' in store_addr):
                    optimized.append(instr)
                    i += 1
                    continue

                # Look ahead to see if there's another store to same address
                j = i + 1
                is_dead = False

                while j < len(instructions):
                    next_instr = instructions[j]

                    # If we hit a label or directive, stop looking
                    if (next_instr.opcode.endswith(':') or
                        next_instr.opcode.startswith('.') or
                        next_instr.opcode in ['RTS', 'RTI', 'RTL', 'JMP', 'BRA']):
                        break

                    # If we find another store to same address, first store is dead
                    if next_instr.opcode == "STA" and next_instr.operand == store_addr:
                        is_dead = True
                        break

                    # If we find a read of the address, store is not dead
                    if (next_instr.opcode in ['LDA', 'ADC', 'SBC', 'AND', 'ORA', 'EOR', 'CMP'] and
                        next_instr.operand == store_addr):
                        break

                    # If we find a branch, stop looking (conservative)
                    if next_instr.opcode in ['BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS']:
                        break

                    j += 1

                if is_dead:
                    # Skip this dead store
                    i += 1
                    self.optimizations_applied += 1
                    continue

            optimized.append(instr)
            i += 1

        return optimized


def optimize_assembly(assembly_lines: List[str]) -> Tuple[List[str], int]:
    """
    Apply peephole optimizations to assembly code.

    Args:
        assembly_lines: List of assembly instruction strings

    Returns:
        Tuple of (optimized lines, number of optimizations applied)
    """
    optimizer = PeepholeOptimizer()
    optimized = optimizer.optimize(assembly_lines)
    return optimized, optimizer.optimizations_applied
