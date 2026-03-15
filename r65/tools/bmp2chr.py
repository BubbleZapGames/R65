"""
bmp2chr - Convert an indexed bitmap to SNES CHR tile data.

Supports planar (2bpp, 3bpp, 4bpp, 8bpp), linear (2bpp, 4bpp, 8bpp),
and Mode 7 (8bpp, 1 byte per pixel) output formats. Optionally outputs
a SNES-format palette file.

With -t/--tilemap, de-duplicates 8x8 tiles (matching flipped variants and
palette assignments) and outputs a packed .chr plus a .tilemap file.

Usage:
  r65x bmp2chr input.bmp -o output.chr -b4
  r65x bmp2chr input.bmp -o output.chr -l4 --fullsize
  r65x bmp2chr input.bmp -o output.chr -b4 -p
  r65x bmp2chr input.bmp -o output.chr -b4 -t
"""

import os
import struct
import sys

from r65.tools.bitmap import BitmapIndex
from r65.tools.tile import (
    Encode2bppTile, Encode3bppTile, Encode4bppTile, Encode8bppTile,
    EncodeLinear2Tile, EncodeLinear4Tile, EncodeLinear8Tile,
    EncodeMode7Tile,
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
    parser.add_argument('-m7', '--mode7', action='store_true', default=False,
                        help="Mode 7 graphic output (8bpp, 1 byte per pixel)")
    parser.add_argument('-p', '--palette', action='store_true', default=False,
                        help="Output color *.pal file")
    parser.add_argument('-t', '--tilemap', action='store_true', default=False,
                        help="De-duplicate tiles and output .tilemap file")
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
        # Bits 0-7: Blue, Bits 8-15: Green, Bits 16-23: Red
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF

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


def _extract_tile(bitmap, tx, ty):
    """Extract an 8x8 tile as a flat list of pixel indices."""
    tile = bytearray()
    for y in range(ty, ty + 8):
        for x in range(tx, tx + 8):
            tile.append(bitmap.getPixel(x, y))
    return tile


def _flip_h(tile):
    """Horizontally flip an 8x8 tile (reverse each row)."""
    out = bytearray(64)
    for row in range(8):
        off = row * 8
        for col in range(8):
            out[off + col] = tile[off + 7 - col]
    return out


def _flip_v(tile):
    """Vertically flip an 8x8 tile (reverse row order)."""
    out = bytearray(64)
    for row in range(8):
        src = (7 - row) * 8
        dst = row * 8
        out[dst:dst + 8] = tile[src:src + 8]
    return out


def _tile_palette(tile, colors_per_pal):
    """Determine palette index from pixel values. Returns (palette, normalized_tile).

    For tiles using palette N, pixel values are in [N*colors_per_pal, (N+1)*colors_per_pal).
    Normalized tile has pixel values in [0, colors_per_pal).
    Returns None if tile uses colors from multiple palettes (invalid).
    """
    pal = None
    for px in tile:
        if px == 0:
            continue  # Color 0 is transparent, shared across palettes
        p = px // colors_per_pal
        if pal is None:
            pal = p
        elif p != pal:
            return None, None
    if pal is None:
        pal = 0
    normalized = bytearray(len(tile))
    base = pal * colors_per_pal
    for i, px in enumerate(tile):
        if px < base:
            normalized[i] = px  # color 0 / shared colors stay as-is
        else:
            normalized[i] = px - base
    return pal, normalized


def _build_tilemap(bitmap, depth, encode):
    """Build de-duplicated tile set and tilemap.

    Returns (unique_tiles_encoded, tilemap_entries) where tilemap_entries
    is a list of 16-bit SNES tilemap words.
    """
    colors_per_pal = 1 << depth
    num_palette_colors = len(bitmap._palette)
    use_palette_matching = num_palette_colors > colors_per_pal and depth in (2, 4)

    # Map from normalized tile bytes → (tile_index)
    tile_dict = {}
    unique_tiles = []
    tilemap = []

    tiles_wide = bitmap._bcWidth // 8
    tiles_high = bitmap._bcHeight // 8

    for ty_idx in range(tiles_high):
        for tx_idx in range(tiles_wide):
            raw_tile = _extract_tile(bitmap, tx_idx * 8, ty_idx * 8)

            if use_palette_matching:
                pal, norm_tile = _tile_palette(raw_tile, colors_per_pal)
                if pal is None:
                    print("Warning: tile at (%d,%d) uses colors from multiple palettes"
                          % (tx_idx * 8, ty_idx * 8), file=sys.stderr)
                    pal = 0
                    norm_tile = raw_tile
            else:
                pal = 0
                norm_tile = raw_tile

            # Try all flip variants: (h_flip, v_flip)
            variants = [
                (norm_tile, 0, 0),
                (_flip_h(norm_tile), 1, 0),
                (_flip_v(norm_tile), 0, 1),
                (_flip_h(_flip_v(norm_tile)), 1, 1),
            ]

            matched = False
            for variant, h, v in variants:
                key = bytes(variant)
                if key in tile_dict:
                    tile_idx = tile_dict[key]
                    entry = tile_idx | (pal << 10) | (h << 14) | (v << 15)
                    tilemap.append(entry)
                    matched = True
                    break

            if not matched:
                tile_idx = len(unique_tiles)
                if tile_idx > 1023:
                    print("Warning: tile count exceeds 1024 at (%d,%d)"
                          % (tx_idx * 8, ty_idx * 8), file=sys.stderr)
                tile_dict[bytes(norm_tile)] = tile_idx
                unique_tiles.append(encode(bytearray(norm_tile)))
                entry = tile_idx | (pal << 10)
                tilemap.append(entry)

    return unique_tiles, tilemap


def bmp2chr_command(args):
    """Execute the bmp2chr command."""
    if args.input:
        try:
            b = BitmapIndex.read(args.input)
        except Exception as e:
            print("Error: %s" % str(e), file=sys.stderr)
            sys.exit(1)

        if args.mode7 and (args.b2pp or args.b3pp or args.b8pp):
            print("Error: --mode7 cannot be combined with --b2pp, --b3pp, or --b8pp",
                  file=sys.stderr)
            sys.exit(1)

        if args.mode7:
            encode = EncodeMode7Tile
            depth = 8
        elif args.b2pp:
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

        if args.tilemap and depth in (2, 4) and b._bcBitCount > depth:
            # Tilemap mode: allow higher-depth BMPs for multi-palette matching
            # e.g., 8-bit BMP with 4bpp tiles (colors 0-15 = pal 0, 16-31 = pal 1, ...)
            pass
        elif depth != b._bcBitCount:
            print("Error: Bitmap file %s does not have a bit depth of %d" % (args.input, depth),
                  file=sys.stderr)
            sys.exit(1)

        if b._bcWidth % 8 != 0 or b._bcHeight % 8 != 0:
            print("Error: Bitmap file %s does not have multiple tile dimensions of 8x8" % args.input,
                  file=sys.stderr)
            sys.exit(1)

        if args.tilemap:
            # Tilemap mode: de-duplicate tiles and output .tilemap
            unique_tiles, tilemap_entries = _build_tilemap(b, depth, encode)

            try:
                with open(args.output, "wb") as chr_fp:
                    for encoded_tile in unique_tiles:
                        chr_fp.write(encoded_tile)
            except Exception as e:
                print("Error: %s" % str(e), file=sys.stderr)
                sys.exit(1)

            tilemap_path = os.path.splitext(args.output)[0] + '.tilemap'
            try:
                with open(tilemap_path, "wb") as tm_fp:
                    for entry in tilemap_entries:
                        tm_fp.write(struct.pack('<H', entry))
            except Exception as e:
                print("Error writing tilemap: %s" % str(e), file=sys.stderr)
                sys.exit(1)

            tiles_wide = b._bcWidth // 8
            tiles_high = b._bcHeight // 8
            total_tiles = tiles_wide * tiles_high
            print("Wrote %d unique tiles (%d total, %.0f%% reduction): %s"
                  % (len(unique_tiles), total_tiles,
                     (1 - len(unique_tiles) / total_tiles) * 100 if total_tiles else 0,
                     args.output),
                  file=sys.stderr)
            print("Wrote tilemap (%dx%d = %d entries): %s"
                  % (tiles_wide, tiles_high, len(tilemap_entries), tilemap_path),
                  file=sys.stderr)
        else:
            # Standard mode: write all tiles sequentially
            # For odd shaped bitmaps match the number of tiles in the destination chr file
            if os.path.isfile(args.output) and not args.fullsize:
                max_size = os.path.getsize(args.output)
            else:
                tiles_wide = b._bcWidth // 8
                tiles_high = b._bcHeight // 8
                bytes_per_tile = {2: 16, 3: 24, 4: 32, 8: 64}.get(depth, 32)
                max_size = tiles_wide * tiles_high * bytes_per_tile

            try:
                chr_fp = open(args.output, "wb")
            except Exception as e:
                print("Error: %s" % str(e), file=sys.stderr)
                sys.exit(1)

            running = True
            for ty in range(0, b._bcHeight, 8):
                for tx in range(0, b._bcWidth, 8):
                    tile = _extract_tile(b, tx, ty)
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
