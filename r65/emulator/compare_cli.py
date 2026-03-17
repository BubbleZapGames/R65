# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Command-line interface for ROM execution comparison.

Usage:
    python -m r65.emulator.compare_cli original.smc port.sfc
    python -m r65.emulator.compare_cli original.smc port.sfc --max-instructions 10000
    python -m r65.emulator.compare_cli original.smc port.sfc --verbose
"""

import argparse
import sys
from pathlib import Path

from .compare import RomComparator, load_rom_with_header_detection


def main():
    parser = argparse.ArgumentParser(
        description="Compare execution of two SNES ROMs instruction-by-instruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python -m r65.emulator.compare_cli original.smc port.sfc

  # With more instructions
  python -m r65.emulator.compare_cli original.smc port.sfc --max-instructions 10000

  # Verbose parallel trace
  python -m r65.emulator.compare_cli original.smc port.sfc --verbose

  # Enable vblank NMI
  python -m r65.emulator.compare_cli original.smc port.sfc --enable-nmi

  # Find multiple divergences
  python -m r65.emulator.compare_cli original.smc port.sfc --continue-on-diverge
        """
    )

    parser.add_argument(
        "rom1",
        type=str,
        help="Path to first ROM (original)"
    )

    parser.add_argument(
        "rom2",
        type=str,
        help="Path to second ROM (port)"
    )

    parser.add_argument(
        "--max-instructions", "-n",
        type=int,
        default=1000,
        help="Maximum instructions to execute (default: 1000)"
    )

    parser.add_argument(
        "--continue-on-diverge", "-c",
        action="store_true",
        help="Continue running after finding divergence"
    )

    parser.add_argument(
        "--enable-nmi",
        action="store_true",
        help="Enable vblank NMI timing"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show parallel trace output"
    )

    parser.add_argument(
        "--mapping", "-m",
        choices=["lorom", "hirom", "auto"],
        default="auto",
        help="ROM memory mapping (default: auto)"
    )

    parser.add_argument(
        "--context", "-C",
        type=int,
        default=10,
        help="Number of instructions to show before divergence (default: 10)"
    )

    parser.add_argument(
        "--header1",
        action="store_true",
        help="Force ROM1 to have 512-byte header"
    )

    parser.add_argument(
        "--no-header1",
        action="store_true",
        help="Force ROM1 to have no header"
    )

    parser.add_argument(
        "--header2",
        action="store_true",
        help="Force ROM2 to have 512-byte header"
    )

    parser.add_argument(
        "--no-header2",
        action="store_true",
        help="Force ROM2 to have no header"
    )

    parser.add_argument(
        "--name1",
        type=str,
        default=None,
        help="Display name for ROM1 (default: filename)"
    )

    parser.add_argument(
        "--name2",
        type=str,
        default=None,
        help="Display name for ROM2 (default: filename)"
    )

    args = parser.parse_args()

    # Load ROMs
    rom1_path = Path(args.rom1)
    rom2_path = Path(args.rom2)

    if not rom1_path.exists():
        print(f"Error: ROM file not found: {args.rom1}", file=sys.stderr)
        sys.exit(1)

    if not rom2_path.exists():
        print(f"Error: ROM file not found: {args.rom2}", file=sys.stderr)
        sys.exit(1)

    # Load ROM data with header handling
    try:
        if args.no_header1:
            rom1_data = rom1_path.read_bytes()
        elif args.header1:
            rom1_data = rom1_path.read_bytes()[512:]
        else:
            rom1_data = load_rom_with_header_detection(str(rom1_path))

        if args.no_header2:
            rom2_data = rom2_path.read_bytes()
        elif args.header2:
            rom2_data = rom2_path.read_bytes()[512:]
        else:
            rom2_data = load_rom_with_header_detection(str(rom2_path))

    except IOError as e:
        print(f"Error reading ROM: {e}", file=sys.stderr)
        sys.exit(1)

    # Set display names
    rom1_name = args.name1 or rom1_path.name
    rom2_name = args.name2 or rom2_path.name

    # Print header
    print("=" * 70)
    print("ROM Execution Comparison")
    print("=" * 70)
    print(f"  ROM 1: {rom1_name}")
    print(f"         {len(rom1_data):,} bytes")
    print(f"  ROM 2: {rom2_name}")
    print(f"         {len(rom2_data):,} bytes")
    print(f"  Max instructions: {args.max_instructions:,}")
    print(f"  NMI timing: {'enabled' if args.enable_nmi else 'disabled'}")
    print("=" * 70)
    print()

    # Create comparator
    comparator = RomComparator(
        rom1_data, rom2_data,
        rom1_name=rom1_name,
        rom2_name=rom2_name,
        mapping=args.mapping
    )

    # Reset and configure
    comparator.reset()
    if args.enable_nmi:
        comparator.enable_nmi(True)

    print(f"Reset vectors:")
    print(f"  {rom1_name}: ${comparator.cpu1.PC:04X}")
    print(f"  {rom2_name}: ${comparator.cpu2.PC:04X}")
    print()

    if comparator.cpu1.PC != comparator.cpu2.PC:
        print("Note: Reset vectors differ - execution paths may diverge from start")
        print()

    # Run comparison
    print("Running comparison...")
    if args.verbose:
        print("-" * 70)

    divergence = comparator.run(
        max_instructions=args.max_instructions,
        continue_on_diverge=args.continue_on_diverge,
        verbose=args.verbose
    )

    # Report results
    print()
    if divergence:
        comparator.format_divergence(divergence, context_before=args.context)

        if args.continue_on_diverge and len(comparator.divergences) > 1:
            print(f"\nTotal divergences found: {len(comparator.divergences)}")
            for i, div in enumerate(comparator.divergences[:10]):  # Show first 10
                print(f"  #{div.instruction_number}: {div.differences[0]}")
            if len(comparator.divergences) > 10:
                print(f"  ... and {len(comparator.divergences) - 10} more")
    else:
        print("=" * 70)
        print("NO DIVERGENCE FOUND")
        print("=" * 70)
        print(f"ROMs executed identically for {args.max_instructions:,} instructions")

    # Summary
    print()
    print(f"Comparison complete.")
    print(f"  Instructions compared: {comparator.logger1.instruction_count:,}")
    print(f"  {rom1_name} cycles: {comparator.cpu1.cycles:,}")
    print(f"  {rom2_name} cycles: {comparator.cpu2.cycles:,}")


if __name__ == "__main__":
    main()
