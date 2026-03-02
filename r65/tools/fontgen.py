"""
Font Generator for SNES Console

Renders a TrueType monospace font as SNES tile data with 3-color anti-aliased output.

Color mapping:
  Color 0 (bp0=0, bp1=0): background/transparent
  Color 2 (bp0=0, bp1=1): anti-alias edges (light gray)
  Color 3 (bp0=1, bp1=1): character body (white)

Usage:
  r65x fontgen [--font PATH] [--size N] [--color {2,4,8}]
               [--low-thresh N] [--high-thresh N] [--bold] [--preview]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as _Image

TILE_W = 8
TILE_H = 8
FIRST_CHAR = 32   # space
LAST_CHAR = 126    # tilde
NUM_TILES = LAST_CHAR - FIRST_CHAR + 1  # 95

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def find_best_size(font_path: str) -> int:
    """Find the largest point size that fits well in 8x8.

    Allows up to 2 pixels of overflow (clipping descenders on tall chars
    like | or g) to maximize readability for the common case.
    """
    best = 6
    for size in range(6, 20):
        font = ImageFont.truetype(font_path, size)
        ascent, _ = font.getmetrics()
        max_above = 0
        max_below = 0
        max_w = 0
        for code in range(FIRST_CHAR, LAST_CHAR + 1):
            bbox = font.getbbox(chr(code))
            above = ascent - bbox[1]
            below = bbox[3] - ascent
            w = bbox[2] - bbox[0]
            max_above = max(max_above, above)
            max_below = max(max_below, below)
            max_w = max(max_w, w)
        # Allow 2px overflow — rare tall chars (|, g) may clip slightly
        # but the common case (A-Z, a-z, 0-9) reads much better at larger sizes
        if max_above + max_below <= TILE_H + 2 and max_w <= TILE_W:
            best = size
    return best


def render_font_baseline(font: ImageFont.FreeTypeFont) -> list[Image.Image]:
    """Render all glyphs with consistent baseline alignment.

    Computes optimal baseline row to maximize visible content in 8x8:
    - Measures actual max ascent/descent across all printable glyphs
    - Places baseline to fit as much as possible, favoring ascenders over descenders
    """
    ascent, descent = font.getmetrics()

    # Compute actual glyph extents relative to baseline
    max_above = 0  # max pixels above baseline
    max_below = 0  # max pixels below baseline
    for code in range(FIRST_CHAR + 1, LAST_CHAR + 1):  # skip space
        bbox = font.getbbox(chr(code))
        above = ascent - bbox[1]
        below = bbox[3] - ascent
        max_above = max(max_above, above)
        max_below = max(max_below, below)

    # Compute baseline row in tile coordinates.
    # Most chars use the shared baseline. Chars whose full extent doesn't fit
    # get an individually adjusted baseline to maximize visible content.
    # Priority: full capitals and ascenders > partial descender clipping.
    shared_baseline = min(max_above, TILE_H - 1)

    tiles = []
    for code in range(FIRST_CHAR, LAST_CHAR + 1):
        ch = chr(code)

        if ch == " ":
            tiles.append(Image.new("L", (TILE_W, TILE_H), 0))
            continue

        # Render to a canvas large enough for any glyph
        canvas_h = ascent + descent + 4
        canvas_w = TILE_W + 8
        img = Image.new("L", (canvas_w, canvas_h), 0)
        draw = ImageDraw.Draw(img)

        # Draw at x=4, y=0 (Pillow places baseline at y + ascent)
        draw.text((4, 0), ch, font=font, fill=255)

        # Find ink bounds
        content_bbox = img.getbbox()
        if content_bbox is None:
            tiles.append(Image.new("L", (TILE_W, TILE_H), 0))
            continue

        left = content_bbox[0]
        right = content_bbox[2]
        glyph_w = right - left

        # Per-glyph baseline: use shared baseline, but shift up if the glyph
        # is entirely below baseline (e.g., underscore) or if the glyph's
        # descender would be completely invisible.
        bbox = font.getbbox(ch)
        char_above = ascent - bbox[1]
        char_below = bbox[3] - ascent
        baseline_row = shared_baseline

        # If this char has descenders that won't fit, shift baseline up
        # but only as far as needed, and only if ascenders still fit.
        if char_below > 0:
            needed_below = char_below
            avail_below = TILE_H - 1 - baseline_row
            if avail_below < needed_below:
                shift = needed_below - avail_below
                # Only shift if ascenders still fit after shifting
                if baseline_row - shift >= char_above - 1:
                    baseline_row -= shift

        # Center horizontally
        x_off = (TILE_W - glyph_w) // 2
        x_off = max(0, x_off)

        # Map canvas y to tile y:
        # Canvas baseline is at y=ascent. Tile baseline is at y=baseline_row.
        tile = Image.new("L", (TILE_W, TILE_H), 0)
        for ty in range(TILE_H):
            src_y = ty - baseline_row + ascent
            if src_y < 0 or src_y >= canvas_h:
                continue
            for tx in range(TILE_W):
                src_x = tx - x_off + left
                if src_x < 0 or src_x >= canvas_w:
                    continue
                pixel = img.getpixel((src_x, src_y))
                tile.putpixel((tx, ty), pixel)

        tiles.append(tile)

    return tiles


def quantize_tile(tile: Image.Image, low_thresh: int, high_thresh: int) -> list[list[int]]:
    """Quantize an 8x8 grayscale tile to 3 colors (0, 2, 3)."""
    result = []
    for y in range(TILE_H):
        row = []
        for x in range(TILE_W):
            pixel = tile.getpixel((x, y))
            if pixel < low_thresh:
                row.append(0)  # background
            elif pixel < high_thresh:
                row.append(2)  # anti-alias edge
            else:
                row.append(3)  # body
        result.append(row)
    return result


def tile_to_snes(colors: list[list[int]], bpp: int = 2) -> list[int]:
    """Convert quantized 8x8 tile to SNES interleaved bitplane format.

    SNES tiles store bitplanes in pairs. Each pair is 16 bytes (8 rows x 2 planes).
    2bpp = 1 pair (16 bytes), 4bpp = 2 pairs (32 bytes), 8bpp = 4 pairs (64 bytes).

    Output layout: [pair0-1] [pair2-3] [pair4-5] [pair6-7]
    Each pair: [bpN_row0, bpN+1_row0, bpN_row1, bpN+1_row1, ...]
    """
    data = []
    num_pairs = bpp // 2
    for pair in range(num_pairs):
        bp_lo = pair * 2
        bp_hi = pair * 2 + 1
        for y in range(TILE_H):
            lo_byte = 0
            hi_byte = 0
            for x in range(TILE_W):
                c = colors[y][x]
                bit = 7 - x
                if (c >> bp_lo) & 1:
                    lo_byte |= (1 << bit)
                if (c >> bp_hi) & 1:
                    hi_byte |= (1 << bit)
            data.append(lo_byte)
            data.append(hi_byte)
    return data


def preview_tiles(all_colors: list[list[list[int]]]) -> None:
    """Print ASCII art preview of all tiles to stderr."""
    SHADE = {0: " ", 2: ".", 3: "#"}

    for i, colors in enumerate(all_colors):
        code = FIRST_CHAR + i
        ch = chr(code)
        if ch == " ":
            label = "SPC"
        elif ch == "\\":
            label = "\\\\"
        else:
            label = ch
        sys.stderr.write(f"  Tile {i:2d}: '{label}' (0x{code:02X})  ")

        sys.stderr.write("\n")

        for y in range(TILE_H):
            sys.stderr.write("    ")
            for x in range(TILE_W):
                sys.stderr.write(SHADE[colors[y][x]])
            sys.stderr.write("\n")
        sys.stderr.write("\n")


def format_font_data(all_tile_data: list[list[int]]) -> str:
    """Format tile data as R65 array literal content."""
    lines = []
    for i, tile_data in enumerate(all_tile_data):
        code = FIRST_CHAR + i
        ch = chr(code)
        if ch == "'":
            label = "\\'"
        elif ch == "\\":
            label = "\\\\"
        else:
            label = ch
        lines.append(f"    // Tile {i:2d}: '{label}' (0x{code:02X})")

        # Format in lines of 8 bytes each
        num_rows = (len(tile_data) + 7) // 8
        for r in range(num_rows):
            chunk = tile_data[r * 8:(r + 1) * 8]
            row_str = ", ".join(f"0x{b:02X}" for b in chunk)
            is_last_row = (r == num_rows - 1)
            is_last_tile = (i == len(all_tile_data) - 1)
            trailing = "" if (is_last_row and is_last_tile) else ","
            lines.append(f"    {row_str}{trailing}")

    return "\n".join(lines)


def bolden_colors(all_colors: list[list[list[int]]]) -> list[list[list[int]]]:
    """Make font bolder by expanding ink outward.

    For each glyph:
    - Original body (3) stays body
    - Original AA (2) promotes to body (3)
    - Background (0) adjacent to any original ink becomes AA (2)
    """
    result = []
    for colors in all_colors:
        new = [[0] * TILE_W for _ in range(TILE_H)]
        for y in range(TILE_H):
            for x in range(TILE_W):
                orig = colors[y][x]
                if orig == 3 or orig == 2:
                    # Promote all ink to body
                    new[y][x] = 3
                else:
                    # Check 4-connected neighbors for any ink
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < TILE_H and 0 <= nx < TILE_W:
                            if colors[ny][nx] != 0:
                                new[y][x] = 2
                                break
        result.append(new)
    return result


def register_parser(subparsers) -> None:
    """Register the fontgen subcommand with the r65x CLI."""
    parser = subparsers.add_parser(
        'fontgen',
        help='Generate SNES font tiles from a TrueType font',
        description='Render a TrueType monospace font as SNES tile data '
                    'with 3-color anti-aliased output. Prints R65 array '
                    'literal to stdout.',
    )
    parser.add_argument("--font", default=DEFAULT_FONT,
                        help="Path to .ttf font file (default: DejaVu Sans Mono Bold)")
    parser.add_argument("--size", type=int, default=0,
                        help="Font point size (0 = auto-detect best fit for 8x8)")
    parser.add_argument("--low-thresh", type=int, default=20,
                        help="Grayscale threshold for AA edges (default: 20)")
    parser.add_argument("--high-thresh", type=int, default=80,
                        help="Grayscale threshold for solid body (default: 80)")
    parser.add_argument("--color", type=int, choices=[2, 4, 8], default=2,
                        help="Bits per pixel: 2 (4 colors), 4 (16 colors), 8 (256 colors)")
    parser.add_argument("--bold", action="count", default=0,
                        help="Make font bolder (can be repeated: --bold --bold)")
    parser.add_argument("--preview", action="store_true",
                        help="Print ASCII art preview of all tiles to stderr")


def fontgen_command(args) -> None:
    """Execute the fontgen command."""
    from PIL import Image, ImageDraw, ImageFont

    # Make PIL available to module-level functions
    import r65.tools.fontgen as _self
    _self.Image = Image
    _self.ImageDraw = ImageDraw
    _self.ImageFont = ImageFont

    font_path = args.font
    if not Path(font_path).exists():
        print(f"Error: Font file not found: {font_path}", file=sys.stderr)
        sys.exit(1)

    # Determine font size
    if args.size > 0:
        size = args.size
    else:
        size = find_best_size(font_path)
        print(f"Auto-detected font size: {size}pt", file=sys.stderr)

    font = ImageFont.truetype(font_path, size)
    print(f"Rendering {NUM_TILES} tiles from {Path(font_path).name} at {size}pt...",
          file=sys.stderr)

    # Render all glyphs with baseline alignment
    grayscale_tiles = render_font_baseline(font)

    # Quantize to 3 colors
    all_colors = []
    for tile in grayscale_tiles:
        colors = quantize_tile(tile, args.low_thresh, args.high_thresh)
        all_colors.append(colors)

    for _ in range(args.bold):
        all_colors = bolden_colors(all_colors)

    if args.preview:
        preview_tiles(all_colors)

    # Convert to tile data
    bpp = args.color
    bytes_per_tile = bpp * TILE_H

    all_tile_data = []
    color_counts = {0: 0, 2: 0, 3: 0}
    for colors in all_colors:
        tile_data = tile_to_snes(colors, bpp)
        all_tile_data.append(tile_data)
        for row in colors:
            for c in row:
                color_counts[c] += 1

    total_bytes = NUM_TILES * bytes_per_tile
    total_pixels = NUM_TILES * TILE_W * TILE_H

    print(f"Color distribution ({total_pixels} total pixels):", file=sys.stderr)
    print(f"  Color 0 (background): {color_counts[0]:5d} ({100*color_counts[0]/total_pixels:.1f}%)",
          file=sys.stderr)
    print(f"  Color 2 (AA edges):   {color_counts[2]:5d} ({100*color_counts[2]/total_pixels:.1f}%)",
          file=sys.stderr)
    print(f"  Color 3 (body):       {color_counts[3]:5d} ({100*color_counts[3]/total_pixels:.1f}%)",
          file=sys.stderr)
    print(f"Total: {total_bytes} bytes ({NUM_TILES} tiles x {bytes_per_tile} bytes, {bpp}bpp)",
          file=sys.stderr)

    # Output font data to stdout
    font_data_str = format_font_data(all_tile_data)
    print(f"static CONSOLE_FONT: [u8; {total_bytes}] = [")
    print(font_data_str)
    print("];")
