"""
Command-line interface for the 65816 emulator.
"""

import argparse
import sys
from pathlib import Path

from .cpu import CPU65816, StopExecution, WaitForInterrupt
from .memory import Memory, detect_mapping
from .trace import TraceLogger, CompactTraceLogger, NullTraceLogger


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
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-trace output"
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
    memory = Memory(rom_data, mapping)
    cpu = CPU65816(memory)

    # Reset CPU
    cpu.reset()

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

    # Show initial state
    if args.show_state and not args.quiet:
        logger.log_state(cpu, "Initial State")
        logger.log_separator()

    if args.trace and not args.compact:
        logger.log_header()

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
        logger.log_separator()
        logger.log_state(cpu, "Final State")

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
    memory = Memory(rom_data, mapping)
    cpu = CPU65816(memory)
    cpu.reset()

    if trace:
        logger = TraceLogger()
        cpu.run(max_cycles=max_cycles, trace_callback=logger.log)
    else:
        cpu.run(max_cycles=max_cycles)

    return cpu


if __name__ == "__main__":
    main()
