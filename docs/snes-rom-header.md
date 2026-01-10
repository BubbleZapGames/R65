# SNES ROM Header

Configure the SNES ROM header using `#[snesrom(...)]`:

```rust
#[snesrom(name="MY GAME", version=0x01, hirom, fastrom)]
```

## Required Parameter

- `name`: ROM name (max 21 characters, padded/truncated automatically)

## Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `id` | `"SNES"` | Cartridge ID (4 characters) |
| `cartridge_type` | `0x00` | Cartridge type byte |
| `sram_size` | `0x00` | SRAM size byte |
| `country` | `0x01` | Country code (0x01 = USA) |
| `version` | `0x00` | ROM version number |

## Memory Mapping Flags

These flags are mutually exclusive:

- `lorom` - LoROM mapping (default)
- `hirom` - HiROM mapping
- `exhirom` - ExHiROM mapping

## ROM Speed Flags

These flags are mutually exclusive:

- `slowrom` - SlowROM timing (default)
- `fastrom` - FastROM timing

## Generated WLA-DX Output

Example output for `#[snesrom(name="MY GAME", version=0x01, hirom, fastrom)]`:

```
.SNESHEADER
  ID "SNES"
  NAME "MY GAME              "
  HIROM
  FASTROM
  CARTRIDGETYPE $00
  ROMSIZE $08
  SRAMSIZE $00
  COUNTRY $01
  LICENSEECODE $00
  VERSION $01
.ENDSNES
```

## Country Codes

Common country codes:
| Code | Region |
|------|--------|
| `0x00` | Japan |
| `0x01` | USA |
| `0x02` | Europe |

## Cartridge Types

Common cartridge types:
| Code | Type |
|------|------|
| `0x00` | ROM only |
| `0x01` | ROM + RAM |
| `0x02` | ROM + RAM + Battery |

See [WLA-DX documentation](https://wla-dx.readthedocs.io/) for complete lists.
