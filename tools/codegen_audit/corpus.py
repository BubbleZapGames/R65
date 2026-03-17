# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Built-in sample R65 snippets for validating the audit harness itself.

Each corpus entry is a self-contained R65 program with at least one
auditable function (beyond the entry point).
"""


CORPUS: dict[str, dict] = {
    'simple_add': {
        'description': 'Trivial register function (baseline)',
        'source': '''\
#[snesrom(name="CORPUS ADD")]
#[bank(0)]

fn add_ten(val @ A: u8) -> u8 {
    A = A + 10;
    return A;
}

#[entry]
fn main() {
    A = add_ten(5);
}

#[interrupt(nmi)]
fn nmi_handler() {}
''',
        'functions': ['add_ten'],
    },

    'ascii_to_tile': {
        'description': 'Branch/compare function (tests branch optimization)',
        'source': '''\
#[snesrom(name="CORPUS TILE")]
#[bank(0)]

fn ascii_to_tile(char @ A: u8) -> u8 {
    if A == 0x20 {
        A = 0;
    } else {
        A = A - 0x40;
    }
    return A;
}

#[entry]
fn main() {
    A = ascii_to_tile(0x41);
}

#[interrupt(nmi)]
fn nmi_handler() {}
''',
        'functions': ['ascii_to_tile'],
    },

    'array_fill': {
        'description': 'Indexed store loop (tests addressing modes)',
        'source': '''\
#[snesrom(name="CORPUS FILL")]
#[bank(0)]

#[ram]
static mut BUFFER: [u8; 16] = [0; 16];

fn fill_buffer(val @ A: u8) {
    for i in 0..16 {
        BUFFER[i] = A;
    }
}

#[entry]
fn main() {
    fill_buffer(0x42);
}

#[interrupt(nmi)]
fn nmi_handler() {}
''',
        'functions': ['fill_buffer'],
    },

    'state_machine': {
        'description': 'If/else chain (tests branch optimization)',
        'source': '''\
#[snesrom(name="CORPUS STATE")]
#[bank(0)]

#[zeropage]
static mut STATE: u8;

fn next_state(current @ A: u8) -> u8 {
    if A == 0 {
        A = 1;
    } else if A == 1 {
        A = 2;
    } else if A == 2 {
        A = 3;
    } else {
        A = 0;
    }
    return A;
}

#[entry]
fn main() {
    A = next_state(2);
    STATE = A;
}

#[interrupt(nmi)]
fn nmi_handler() {}
''',
        'functions': ['next_state'],
    },
}


def list_corpus() -> list[tuple[str, str]]:
    """Return list of (name, description) for all corpus entries."""
    return [(name, entry['description']) for name, entry in CORPUS.items()]


def get_corpus_entry(name: str) -> dict | None:
    """Get a corpus entry by name, or None if not found."""
    return CORPUS.get(name)


def get_all_corpus_entries() -> dict[str, dict]:
    """Return all corpus entries."""
    return CORPUS
