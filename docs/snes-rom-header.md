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

### What `fastrom` actually does

MEMSEL (`$420D`) only speeds up banks `$80-$FF`, addresses `$8000-$FFFF`. Setting
it while executing from banks `$00-$3F` does nothing, so `fastrom` has to move
the code as well as set the header bit. For a LoROM build the compiler:

- assembles ROM banks under `.BASE $80`, so every `:label` / `#:label` bank byte,
  `JSL`, `JML`, `LDA.l` and `MVN` operand resolves into the `$80-$BF` mirror;
- emits reset and interrupt trampolines in bank `$00` - the CPU always fetches
  vectors with PBR=`$00`, so entry code would otherwise run at SlowROM speed:

  ```
  __fast_reset:
      SEI
      CLC
      XCE
      SEP #$20
      LDA #$01
      STA $420D        ; MEMSEL - enable FastROM
      JML main         ; -> $80:8xxx
  __fast_nmi:
      JML nmi_handler
  ```

- sets `DBR = PBR` (`PHK` / `PLB`) in the entry prologue, so absolute ROM data
  reads use the fast mirror too, not just opcode fetch. Hardware registers
  (`$80:2100`) and low RAM (`$80:0000-1FFF`) mirror into `$80-$BF` as well, so
  absolute accesses to them are unaffected.

HiROM already assembles at `.BASE $C0`, which is the FastROM-capable region, so a
HiROM build only gains the MEMSEL write.

Measured on a ROM-fetch-heavy loop: 102 scanlines SlowROM vs 87 FastROM, ~15%
faster. The theoretical ceiling is 25% (8 to 6 master cycles per access); real
code mixes in RAM and internal cycles, which do not speed up.

**Do not write MEMSEL yourself.** The compiler owns `$420D` so the register can
never disagree with the header speed bit or the bank placement.

**Limit**: banks `$80-$BF` mirror `$00-$3F` for the first 4MB only. The compiler
rejects `fastrom` on a ROM that would need a bank past the mirror.

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
