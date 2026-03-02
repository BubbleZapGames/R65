#!/usr/bin/env python3
"""Thin wrapper — use `r65x fontgen` instead."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from r65.tools.fontgen import fontgen_command
import argparse
from r65.tools.fontgen import DEFAULT_FONT

parser = argparse.ArgumentParser(description="Generate SNES font tiles from TrueType font")
parser.add_argument("--font", default=DEFAULT_FONT, help="Path to .ttf font file")
parser.add_argument("--size", type=int, default=0, help="Font point size (0 = auto-detect)")
parser.add_argument("--low-thresh", type=int, default=20, help="Grayscale threshold for AA edges")
parser.add_argument("--high-thresh", type=int, default=80, help="Grayscale threshold for solid body")
parser.add_argument("--color", type=int, choices=[2, 4, 8], default=2, help="Bits per pixel (2, 4, or 8)")
parser.add_argument("--bold", action="count", default=0, help="Make font bolder")
parser.add_argument("--preview", action="store_true", help="Print ASCII art preview to stderr")
fontgen_command(parser.parse_args())
