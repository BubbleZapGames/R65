"""
bmp2chr - Convert an indexed bitmap to SNES CHR tile data.

Supports planar (2bpp, 3bpp, 4bpp, 8bpp) and linear (2bpp, 4bpp, 8bpp)
output formats. Optionally outputs a SNES-format palette file.

Usage:
  r65x bmp2chr input.bmp -o output.chr -b4
  r65x bmp2chr input.bmp -o output.chr -l4 --fullsize
  r65x bmp2chr input.bmp -o output.chr -b4 -p
"""

import os
import struct
import sys

from r65.tools.bitmap import BitmapIndex
from r65.tools.tile import (
    Encode2bppTile, Encode3bppTile, Encode4bppTile, Encode8bppTile,
    EncodeLinear2Tile, EncodeLinear4Tile, EncodeLinear8Tile,
)


def register_parser(subparsers):
    """Register the bmp2chr subcommand with the r65x CLI."""
    parser = subparsers.add_parser(
        'bmp2chr',
        help='Convert an indexed bitmap to SNES CHR data',
        description='Convert an indexed bitmap to SNES CHR tile data.',
    )
    parser.add_argument('input', metavar='input.bmp',
                        help="input bitmap file")
    parser.add_argument('-o', '--output', required=True, default=None,
                        help="File path to output *.chr")
    parser.add_argument('-b2', '--b2pp', action='store_true', default=False,
                        help="4 colors planar graphic output")
    parser.add_argument('-b3', '--b3pp', action='store_true', default=False,
                        help="8 colors planar graphic output")
    parser.add_argument('-b4', '--b4pp', action='store_true', default=True,
                        help="16 colors planar graphic output")
    parser.add_argument('-b8', '--b8pp', action='store_true', default=False,
                        help="256 colors planar graphic output")
    parser.add_argument('-l2', '--linear2', action='store_true', default=False,
                        help="4 colors linear graphic output")
    parser.add_argument('-l4', '--linear4', action='store_true', default=False,
                        help="16 colors linear graphic output")
    parser.add_argument('-l8', '--linear8', action='store_true', default=False,
                        help="256 colors linear graphic output")
    parser.add_argument('-p', '--palette', action='store_true', default=False,
                        help="Output color *.pal file")
    parser.add_argument('-f', '--fullsize', action='store_true', default=False,
                        help="Ignore destination CHR file size and write whole bitmap")


def _write_palette(bitmap, output_path):
    """Write SNES-format palette file (.pal) from bitmap palette.

    SNES palette format: 15-bit BGR555, 2 bytes per color (little-endian).
    Each color: 0bbbbbgg gggrrrrr
    """
    pal_path = os.path.splitext(output_path)[0] + '.pal'
    pal_data = bytearray()

    for color in bitmap._palette:
        # BMP palette is stored as BGRA (32-bit little-endian)
        b = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        r = color & 0xFF

        # Convert 8-bit RGB to 5-bit BGR555
        r5 = (r >> 3) & 0x1F
        g5 = (g >> 3) & 0x1F
        b5 = (b >> 3) & 0x1F
        snes_color = r5 | (g5 << 5) | (b5 << 10)

        pal_data += struct.pack('<H', snes_color)

    try:
        with open(pal_path, 'wb') as f:
            f.write(pal_data)
        print("Wrote palette: %s (%d colors)" % (pal_path, len(bitmap._palette)),
              file=sys.stderr)
    except Exception as e:
        print("Error writing palette: %s" % str(e), file=sys.stderr)


def bmp2chr_command(args):
    """Execute the bmp2chr command."""
    if args.input:
        try:
            b = BitmapIndex.read(args.input)
        except Exception as e:
            print("Error: %s" % str(e), file=sys.stderr)
            sys.exit(1)

        if args.b2pp:
            encode = Encode2bppTile
            depth = 2
        elif args.b8pp:
            encode = Encode8bppTile
            depth = 8
        elif args.linear8:
            encode = EncodeLinear8Tile
            depth = 8
        elif args.linear4:
            encode = EncodeLinear4Tile
            depth = 4
        elif args.linear2:
            encode = EncodeLinear2Tile
            depth = 2
        elif args.b3pp:
            encode = Encode3bppTile
            depth = 3
        else:
            encode = Encode4bppTile
            depth = 4

        if depth != b._bcBitCount:
            print("Error: Bitmap file %s does not have a bit depth of %d" % (args.input, depth),
                  file=sys.stderr)
            sys.exit(1)

        if b._bcWidth % 8 != 0 or b._bcHeight % 8 != 0:
            print("Error: Bitmap file %s does not have multiple tile dimensions of 8x8" % args.input,
                  file=sys.stderr)
            sys.exit(1)

        # For odd shaped bitmaps match the number of tiles in the destination chr file by limiting the size
        if os.path.isfile(args.output) and not args.fullsize:
            max_size = os.path.getsize(args.output)
        else:
            # Calculate size based on tile count and bytes per tile
            tiles_wide = b._bcWidth // 8
            tiles_high = b._bcHeight // 8
            bytes_per_tile = {2: 16, 3: 24, 4: 32, 8: 64}.get(depth, 32)
            max_size = tiles_wide * tiles_high * bytes_per_tile

        try:
            chr_fp = open(args.output, "wb")
        except Exception as e:
            print("Error: %s" % str(e), file=sys.stderr)
            sys.exit(1)

        # Write tile data
        running = True
        for ty in range(0, b._bcHeight, 8):
            for tx in range(0, b._bcWidth, 8):
                tile = bytearray()
                for y in range(ty, ty+8):
                    for x in range(tx, tx+8):
                        tile.append(b.getPixel(x, y))
                encoded = encode(tile)
                chr_fp.write(encoded)
                if chr_fp.tell() >= max_size:
                    running = False
                    break

            if not running:
                break
        chr_fp.close()

        # Output palette if requested
        if args.palette:
            _write_palette(b, args.output)
