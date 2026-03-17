# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Command-line interface for the 65816 emulator.
"""

import argparse
import sys
from pathlib import Path

from .cpu import CPU65816, StopExecution, WaitForInterrupt
from .memory import SNESMemory, detect_mapping
from .trace import TraceLogger, CompactTraceLogger, NullTraceLogger
from .disasm import disassemble


def parse_address(addr_str: str) -> tuple:
    """
    Parse an address string like '8000' or '00:8000' or '0x8000'.

    Returns:
        (bank, address) tuple
    """
    addr_str = addr_str.strip().lower()

    # Handle bank:addr format
    if ":" in addr_str:
        parts = addr_str.split(":")
        bank = int(parts[0], 16)
        addr = int(parts[1], 16)
        return (bank, addr)

    # Handle hex prefix
    if addr_str.startswith("0x"):
        addr_str = addr_str[2:]

    # Assume bank 0 for simple addresses
    return (0, int(addr_str, 16))


def main():
    parser = argparse.ArgumentParser(
        description="R65 65816 CPU Emulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  r65-emu game.sfc --trace
  r65-emu game.sfc --max-cycles 1000000
  r65-emu game.sfc --mapping hirom --trace
  r65-emu game.sfc --start 0x8000 --trace
        """
    )

    parser.add_argument(
        "rom",
        type=str,
        help="Path to ROM file (.sfc, .smc, .bin)"
    )

    parser.add_argument(
        "--mapping", "-m",
        choices=["lorom", "hirom", "auto"],
        default="auto",
        help="ROM memory mapping (default: auto-detect)"
    )

    parser.add_argument(
        "--trace", "-t",
        action="store_true",
        help="Enable execution trace logging"
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact trace format"
    )

    parser.add_argument(
        "--max-cycles", "-c",
        type=int,
        default=None,
        help="Maximum cycles to execute"
    )

    parser.add_argument(
        "--max-instructions", "-i",
        type=int,
        default=None,
        help="Maximum instructions to execute"
    )

    parser.add_argument(
        "--start", "-s",
        type=str,
        default=None,
        help="Start address (hex, e.g., 0x8000 or 00:8000)"
    )

    parser.add_argument(
        "--native",
        action="store_true",
        help="Start in native mode (16-bit) instead of emulation mode"
    )

    parser.add_argument(
        "--header",
        action="store_true",
        help="ROM has 512-byte copier header (auto-detected for .smc)"
    )

    parser.add_argument(
        "--no-header",
        action="store_true",
        help="ROM has no copier header"
    )

    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Show CPU state at start and end"
    )

    parser.add_argument(
        "--profile", "-p",
        choices=["snes"],
        default="snes",
        help="Hardware profile (default: snes)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-trace output"
    )

    parser.add_argument(
        "--breakpoint", "-b",
        action="append",
        dest="breakpoints",
        metavar="ADDR",
        help="Set breakpoint at address (hex, can use multiple times)"
    )

    parser.add_argument(
        "--disasm", "-d",
        type=str,
        metavar="ADDR",
        help="Disassemble starting at address instead of running"
    )

    parser.add_argument(
        "--disasm-count",
        type=int,
        default=20,
        metavar="N",
        help="Number of instructions to disassemble (default: 20)"
    )

    args = parser.parse_args()

    # Load ROM
    rom_path = Path(args.rom)
    if not rom_path.exists():
        print(f"Error: ROM file not found: {args.rom}", file=sys.stderr)
        sys.exit(1)

    try:
        rom_data = rom_path.read_bytes()
    except IOError as e:
        print(f"Error reading ROM: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle copier header
    has_header = args.header
    if not args.no_header and not args.header:
        # Auto-detect based on file size only (512-byte header makes size % 1024 == 512)
        # Don't rely on extension - .smc files may or may not have headers
        if len(rom_data) % 1024 == 512:
            has_header = True

    if has_header:
        if not args.quiet:
            print(f"Skipping 512-byte copier header", file=sys.stderr)
        rom_data = rom_data[512:]

    # Detect or set mapping
    if args.mapping == "auto":
        mapping = detect_mapping(rom_data)
        if not args.quiet:
            print(f"Auto-detected mapping: {mapping}", file=sys.stderr)
    else:
        mapping = args.mapping

    # Create memory and CPU
    memory = SNESMemory(rom_data, mapping)
    cpu = CPU65816(memory)

    # Disassembly mode
    if args.disasm:
        bank, addr = parse_address(args.disasm)
        print(f"Disassembly at ${bank:02X}:{addr:04X}:", file=sys.stderr)
        print()

        # Assume 8-bit mode for disassembly unless --native specified
        m_flag = not args.native
        x_flag = not args.native

        for _ in range(args.disasm_count):
            text, size = disassemble(memory, bank, addr, m_flag, x_flag)
            # Show hex bytes
            hex_bytes = " ".join(f"{memory.read((bank << 16) | ((addr + i) & 0xFFFF)):02X}"
                                for i in range(size))
            print(f"${bank:02X}:{addr:04X}  {hex_bytes:<12s}  {text}")
            addr = (addr + size) & 0xFFFF
        sys.exit(0)

    # Reset CPU
    cpu.reset()

    # Enable automatic vblank timing (for games that poll RDNMI $4210)
    cpu.enable_auto_nmi(True)

    # Override start address if specified
    if args.start:
        start = args.start
        if ":" in start:
            # bank:addr format
            parts = start.split(":")
            cpu.PBR = int(parts[0], 16)
            cpu.PC = int(parts[1], 16)
        else:
            cpu.PC = int(start, 16)

    # Switch to native mode if requested
    if args.native:
        cpu.emulation_mode = False
        cpu.P &= ~0x30  # Clear M and X flags for 16-bit mode

    # Create trace logger
    if args.trace:
        if args.compact:
            logger = CompactTraceLogger()
        else:
            logger = TraceLogger()
    else:
        logger = NullTraceLogger()

    # Create a separate logger for state output (always goes to stderr)
    state_logger = TraceLogger(output=sys.stderr)

    # Show initial state
    if args.show_state and not args.quiet:
        state_logger.log_state(cpu, "Initial State")
        state_logger.log_separator()

    if args.trace and not args.compact:
        logger.log_header()

    # Parse breakpoints
    breakpoints = set()
    if args.breakpoints:
        for bp in args.breakpoints:
            try:
                bank, addr = parse_address(bp)
                breakpoints.add((bank, addr))
                if not args.quiet:
                    print(f"Breakpoint set at ${bank:02X}:{addr:04X}", file=sys.stderr)
            except ValueError:
                print(f"Error: Invalid breakpoint address: {bp}", file=sys.stderr)
                sys.exit(1)

    # Run
    instruction_count = 0
    try:
        while True:
            # Check instruction limit
            if args.max_instructions and instruction_count >= args.max_instructions:
                break

            # Check cycle limit
            if args.max_cycles and cpu.cycles >= args.max_cycles:
                break

            # Check if CPU is stopped or waiting
            if cpu.stopped or cpu.waiting:
                break

            # Check breakpoints
            if (cpu.PBR, cpu.PC) in breakpoints:
                if not args.quiet:
                    print(f"\nBreakpoint hit at ${cpu.PBR:02X}:{cpu.PC:04X}", file=sys.stderr)
                break

            # Log and execute
            logger.log(cpu)
            cpu.step()
            instruction_count += 1

    except StopExecution:
        if not args.quiet:
            print(f"\nCPU stopped (STP instruction)", file=sys.stderr)
    except WaitForInterrupt:
        if not args.quiet:
            print(f"\nCPU waiting for interrupt (WAI)", file=sys.stderr)
    except KeyboardInterrupt:
        if not args.quiet:
            print(f"\nInterrupted by user", file=sys.stderr)

    # Show final state
    if args.show_state and not args.quiet:
        state_logger.log_separator()
        state_logger.log_state(cpu, "Final State")

    # Summary
    if not args.quiet:
        print(f"\nExecuted {instruction_count} instructions, {cpu.cycles} cycles",
              file=sys.stderr)


def run_rom(rom_data: bytes, mapping: str = "lorom",
            max_cycles: int = None, trace: bool = False) -> CPU65816:
    """
    Convenience function to run a ROM programmatically.

    Args:
        rom_data: Raw ROM bytes
        mapping: "lorom" or "hirom"
        max_cycles: Maximum cycles to execute
        trace: Enable trace output

    Returns:
        CPU65816 instance after execution
    """
    memory = SNESMemory(rom_data, mapping)
    cpu = CPU65816(memory)
    cpu.reset()
    cpu.enable_auto_nmi(True)  # Enable vblank timing

    if trace:
        logger = TraceLogger()
        cpu.run(max_cycles=max_cycles, trace_callback=logger.log)
    else:
        cpu.run(max_cycles=max_cycles)

    return cpu


if __name__ == "__main__":
    main()
