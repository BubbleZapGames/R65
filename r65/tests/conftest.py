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


@pytest.fixture(scope="session", autouse=True)
def _configure_abi(request):
    abi = request.config.getoption("--abi")
    if abi:
        from r65.tests.e2e import framework
        framework._abi_override = abi
