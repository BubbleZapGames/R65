# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Packer - Encode and decode files with SNES compression algorithms.

Supports multiple compression formats commonly used in SNES ROMs:
aplib, byte_rle, rle1, rle2, lz1-5, lz19, lz77, hal.

Usage:
  r65x packer pack input.bin -o output.bin -x lz5
  r65x packer unpack input.bin -o output.bin -x lz5
"""

import os
import sys

from r65.tools import compression


def register_parser(subparsers):
    """Register the packer subcommand with the r65x CLI."""
    parser = subparsers.add_parser(
        'packer',
        help='Encode and decode files with compression',
        description='Encode and decode files with SNES compression algorithms.',
    )
    parser.add_argument('action', metavar='pack|unpack',
                        help="Action type")
    parser.add_argument('input', metavar='input.bin',
                        help="Input file")
    parser.add_argument('-o', '--output', required=True, metavar='outfile',
                        default=None, help="File path to output")
    parser.add_argument('-x', '--encoding', metavar='|'.join(compression.get_names()),
                        required=True, type=str, help='Encoding algorithm')
    parser.add_argument('-f', '--fullsize', action='store_true', default=False,
                        help="Ignore destination file size and write full data")


def packer_command(args):
    """Execute the packer command."""
    if not args.action or args.action not in ['pack', 'unpack']:
        print("Error: action must be 'pack' or 'unpack'", file=sys.stderr)
        sys.exit(1)

    if args.input:
        try:
            in_fp = open(args.input, "rb")
            data = bytearray(in_fp.read())
            in_fp.close()
        except Exception as e:
            print("Error: %s" % str(e), file=sys.stderr)
            sys.exit(1)

        try:
            module = getattr(compression, args.encoding)
        except AttributeError:
            print("Unsupported encoding type: %s. Use following types %s." % (
                args.encoding, ",".join(compression.get_names())), file=sys.stderr)
            sys.exit(1)

        if args.action == 'pack':
            output = module.compress(data)
        else:
            output = module.decompress(data)

        # If target file already exists then regulate output size
        if os.path.isfile(args.output):
            size = os.path.getsize(args.output)
            if not args.fullsize and size > 0:
                if size > len(output):
                    output = output + bytes(size - len(output))
                elif size < len(output):
                    print("Warning: Truncating output of compression for file %s" % args.output,
                          file=sys.stderr)
                    output = output[0:size]
        try:
            out_fp = open(args.output, "wb")
            out_fp.write(output)
            out_fp.close()
        except Exception as e:
            print("Error: %s" % str(e), file=sys.stderr)
            sys.exit(1)
