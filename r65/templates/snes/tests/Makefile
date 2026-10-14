# {{PROJECT_NAME}} - ROM tests, run by `make tests` from the project root
#
# Each *.script file here is a list of mesen-gdb debugger commands with
# `assert` checks, run headless against the ROM:
#
#   mesen-gdb ../build/{{PROJECT_NAME_UPPER}}.smc --headless --script example.script
#
# A script exits 0 when every assert passes, 1 when one fails, 2 when a
# command fails or times out.
#
# mesen-gdb: https://github.com/BubbleZapGames/Mesen-GDB
# (prebuilt binaries under Releases). If it is not on your PATH, point
# MESEN_GDB at it: make tests MESEN_GDB=/path/to/mesen-gdb

ROM ?= ../build/{{PROJECT_NAME_UPPER}}.smc
MESEN_GDB ?= mesen-gdb
SCRIPTS := $(wildcard *.script)

tests:
	@if ! command -v $(MESEN_GDB) >/dev/null 2>&1; then \
		echo "[TESTS] mesen-gdb not found, skipping."; \
		echo "        Get it from https://github.com/BubbleZapGames/Mesen-GDB"; \
		echo "        or run: make tests MESEN_GDB=/path/to/mesen-gdb"; \
		exit 0; \
	fi; \
	if [ ! -f $(ROM) ]; then \
		echo "[TESTS] $(ROM) not found: run make in the project root first."; \
		exit 1; \
	fi; \
	status=0; \
	for s in $(SCRIPTS); do \
		echo "[TEST] $$s"; \
		$(MESEN_GDB) $(ROM) --headless --script $$s || status=1; \
	done; \
	exit $$status

.PHONY: tests
