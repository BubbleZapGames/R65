# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Command-line interface for behavioral ROM comparison.

Compares what ROMs DO (HW register writes, memory writes) rather than
instruction-by-instruction execution.

Usage:
    python -m r65.emulator.compare_behavior_cli original.smc port.sfc
"""

import argparse
import sys
from pathlib import Path

from .compare_behavior import (
    BehaviorComparator, load_rom_with_header_detection, load_symbols,
    EventType
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ROM behavior (what they DO, not instruction-by-instruction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python -m r65.emulator.compare_behavior_cli original.smc port.sfc

  # With more events
  python -m r65.emulator.compare_behavior_cli original.smc port.sfc --max-events 500

  # Compare HW writes during initialization
  python -m r65.emulator.compare_behavior_cli original.smc port.sfc --hw-only

  # Load R65 symbols for ROM2
  python -m r65.emulator.compare_behavior_cli original.smc port.sfc --sym2 port.sym
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
        "--max-events", "-e",
        type=int,
        default=200,
        help="Maximum behavioral events to capture (default: 200)"
    )

    parser.add_argument(
        "--max-instructions", "-n",
        type=int,
        default=100000,
        help="Maximum instructions per ROM (default: 100000)"
    )

    parser.add_argument(
        "--hw-only",
        action="store_true",
        help="Only compare hardware register writes"
    )

    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only compare memory writes"
    )

    parser.add_argument(
        "--mapping", "-m",
        choices=["lorom", "hirom", "auto"],
        default="auto",
        help="ROM memory mapping (default: auto)"
    )

    parser.add_argument(
        "--name1",
        type=str,
        default=None,
        help="Display name for ROM1"
    )

    parser.add_argument(
        "--name2",
        type=str,
        default=None,
        help="Display name for ROM2"
    )

    parser.add_argument(
        "--sym1",
        type=str,
        default=None,
        help="Symbol file for ROM1"
    )

    parser.add_argument(
        "--sym2",
        type=str,
        default=None,
        help="Symbol file for ROM2"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all events, not just differences"
    )

    parser.add_argument(
        "--memory-range",
        type=str,
        action="append",
        help="Memory range to monitor (e.g., '00:0000-00:00FF')"
    )

    parser.add_argument(
        "--compare-memory",
        action="store_true",
        help="Compare final memory state"
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

    try:
        rom1_data = load_rom_with_header_detection(str(rom1_path))
        rom2_data = load_rom_with_header_detection(str(rom2_path))
    except IOError as e:
        print(f"Error reading ROM: {e}", file=sys.stderr)
        sys.exit(1)

    # Set display names
    rom1_name = args.name1 or rom1_path.stem
    rom2_name = args.name2 or rom2_path.stem

    # Load symbols if provided
    symbols1 = load_symbols(args.sym1) if args.sym1 else {}
    symbols2 = load_symbols(args.sym2) if args.sym2 else {}

    # Print header
    print("=" * 70)
    print("Behavioral ROM Comparison")
    print("=" * 70)
    print(f"  ROM 1: {rom1_name}")
    print(f"         {len(rom1_data):,} bytes")
    if symbols1:
        print(f"         {len(symbols1)} symbols loaded")
    print(f"  ROM 2: {rom2_name}")
    print(f"         {len(rom2_data):,} bytes")
    if symbols2:
        print(f"         {len(symbols2)} symbols loaded")
    print(f"  Max events: {args.max_events}")
    print(f"  Max instructions: {args.max_instructions:,}")
    print("=" * 70)
    print()

    # Create comparator
    comparator = BehaviorComparator(
        rom1_data, rom2_data,
        rom1_name=rom1_name,
        rom2_name=rom2_name,
        mapping=args.mapping
    )

    # Parse memory ranges
    if args.memory_range:
        ranges = []
        for r in args.memory_range:
            try:
                start, end = r.split('-')
                bs, as_ = start.split(':')
                be, ae = end.split(':')
                ranges.append((int(bs, 16), int(be, 16), int(as_, 16), int(ae, 16)))
            except ValueError:
                print(f"Invalid memory range: {r}", file=sys.stderr)
                sys.exit(1)
        comparator.set_monitored_ranges(ranges)

    # Reset CPUs
    comparator.reset()

    print(f"Reset vectors:")
    print(f"  {rom1_name}: ${comparator.cpu1.PC:04X}")
    print(f"  {rom2_name}: ${comparator.cpu2.PC:04X}")
    print()

    # Run both ROMs
    print("Running ROMs to capture behavioral events...")
    events1, events2 = comparator.run_until_event_count(
        max_events=args.max_events,
        max_instructions=args.max_instructions
    )

    print(f"  {rom1_name}: {len(events1)} events, {comparator.mem1.instruction_num:,} instructions")
    print(f"  {rom2_name}: {len(events2)} events, {comparator.mem2.instruction_num:,} instructions")
    print()

    # Filter events based on flags
    if args.hw_only:
        events1 = [e for e in events1 if e.event_type == EventType.HW_WRITE]
        events2 = [e for e in events2 if e.event_type == EventType.HW_WRITE]
    elif args.memory_only:
        events1 = [e for e in events1 if e.event_type == EventType.MEMORY_WRITE]
        events2 = [e for e in events2 if e.event_type == EventType.MEMORY_WRITE]

    # Compare and display results
    if args.hw_only or not args.memory_only:
        comparator.format_hw_comparison(events1, events2)

    # Show differences
    hw_diffs = comparator.compare_hw_writes(events1, events2)
    if hw_diffs:
        print("\nHW Write Differences:")
        print("-" * 50)
        for diff in hw_diffs[:20]:
            print(f"  {diff}")
        if len(hw_diffs) > 20:
            print(f"  ... and {len(hw_diffs) - 20} more differences")
    else:
        print("\nHW Writes: MATCH")

    # Compare memory state if requested
    if args.compare_memory:
        print("\n")
        print("=" * 70)
        print("Memory State Comparison (Low WRAM $0000-$1FFF)")
        print("=" * 70)

        mem_diffs = comparator.compare_memory_state()
        if mem_diffs:
            print(f"\n{len(mem_diffs)} addresses differ:\n")
            # Show first 50
            count = 0
            for addr, (v1, v2) in sorted(mem_diffs.items()):
                if count >= 50:
                    print(f"  ... and {len(mem_diffs) - 50} more")
                    break
                print(f"  ${addr:04X}: {rom1_name}=${v1:02X}, {rom2_name}=${v2:02X}")
                count += 1
        else:
            print("\nMemory state: MATCH")

    # Verbose output - show all events
    if args.verbose:
        print("\n")
        print("=" * 70)
        print(f"All Events - {rom1_name}")
        print("=" * 70)
        for i, e in enumerate(events1[:100]):
            print(f"  {i:4d}: {e}")
        if len(events1) > 100:
            print(f"  ... and {len(events1) - 100} more")

        print("\n")
        print("=" * 70)
        print(f"All Events - {rom2_name}")
        print("=" * 70)
        for i, e in enumerate(events2[:100]):
            print(f"  {i:4d}: {e}")
        if len(events2) > 100:
            print(f"  ... and {len(events2) - 100} more")

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
