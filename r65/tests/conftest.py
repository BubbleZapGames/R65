# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Pytest configuration for R65 compiler tests.

Sets up the Python path so tests can import r65.compiler modules.
"""
import sys
import os
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def pytest_addoption(parser):
    parser.addoption(
        "--abi",
        choices=["Default", "FixedStack", "Pascal"],
        default=None,
        help="ABI model for E2E test compilation (Default, FixedStack, Pascal)",
    )


# What the FixedStack ABI rules out by design: each marker skips its tests
# under --abi FixedStack, with the limit as the reason. The compiler rejects
# these programs there with a compile error.
FIXEDSTACK_LIMITS = {
    "requires_recursion": (
        "FixedStack ABI rejects recursion: a parameter lives in one fixed "
        "scratch or register, not a per-call frame"),
    "requires_stack_params": (
        "FixedStack ABI has no stack parameters: an address-taken function "
        "must bind every parameter to a register"),
}


def pytest_configure(config):
    for name, reason in FIXEDSTACK_LIMITS.items():
        config.addinivalue_line("markers", f"{name}: skipped under --abi FixedStack ({reason})")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--abi") != "FixedStack":
        return
    for item in items:
        for name, reason in FIXEDSTACK_LIMITS.items():
            if item.get_closest_marker(name):
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session", autouse=True)
def _configure_abi(request):
    abi = request.config.getoption("--abi")
    if abi:
        from r65.tests.e2e import framework
        framework._abi_override = abi
