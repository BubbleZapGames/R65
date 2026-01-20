"""
Processor mode tests for R65.

Tests the #[mode(databank=...)] attribute, automatic mode inference from
parameter types, interrupt handlers, and mode control built-ins.

CPU mode (m8/m16) is now automatically inferred:
- Default: m8 (8-bit A), x16 (16-bit X/Y)
- u16 @ A parameter -> m16 entry mode
- X/Y registers are always u16
"""
