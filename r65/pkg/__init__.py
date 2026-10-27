# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
R65 packages: the `r65.toml` manifest, resolution to include directories, the
`r65.lock` lockfile, and the `r65pkg` tool. See docs/packaging-system.md.

`r65c` imports only `manifest`, `version`, `lockfile` (to read) and `resolve`,
none of which run git or write a file; `fetch` and `cli` are the tool's.
"""
