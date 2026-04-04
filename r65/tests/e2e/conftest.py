# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Shared fixtures for e2e tests."""

import pytest
from r65.tests.e2e import E2ETest


@pytest.fixture
def e2e():
    """Create E2ETest instance."""
    return E2ETest()
