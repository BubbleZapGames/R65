# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Versions and version requirements, with Cargo's meanings.

A version is `MAJOR.MINOR.PATCH`. A requirement is a caret range - `"0.4"`
is `>= 0.4.0, < 0.5.0`, `"1.2"` is `>= 1.2.0, < 2.0.0`, `"0.0.3"` is exactly
the patch series `0.0.3` - or `"=X.Y.Z"` for one version. The left-most
non-zero component is the one that may not change.
"""

import re
from dataclasses import dataclass
from typing import Tuple

_VERSION = re.compile(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$')
_PARTIAL = re.compile(r'^(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?(?:\.(0|[1-9]\d*))?$')

Version = Tuple[int, int, int]


class VersionError(ValueError):
    """A malformed version or requirement."""


def parse_version(text: str) -> Version:
    m = _VERSION.match(text.strip()) if isinstance(text, str) else None
    if not m:
        raise VersionError(f"invalid version '{text}' (expected MAJOR.MINOR.PATCH)")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def format_version(version: Version) -> str:
    return '.'.join(str(part) for part in version)


@dataclass(frozen=True)
class Requirement:
    """A half-open range `[low, high)` of versions; `text` is as written."""
    text: str
    low: Version
    high: Version

    @classmethod
    def parse(cls, text: str) -> 'Requirement':
        if not isinstance(text, str):
            raise VersionError(f"invalid version requirement {text!r}")
        spec = text.strip()
        if spec.startswith('='):
            low = parse_version(spec[1:])
            return cls(text, low, (low[0], low[1], low[2] + 1))
        if spec.startswith('^'):
            spec = spec[1:]
        m = _PARTIAL.match(spec)
        if not m:
            raise VersionError(
                f"invalid version requirement '{text}' (expected a caret range "
                f"such as \"0.4\" or \"1.2.3\", or \"=X.Y.Z\")")
        parts = [int(g) if g is not None else None for g in m.groups()]
        major, minor, patch = parts
        low = (major, minor or 0, patch or 0)
        # The left-most non-zero component written may not change; an
        # omitted component is free.
        if major > 0 or minor is None:
            high = (major + 1, 0, 0)
        elif minor > 0 or patch is None:
            high = (0, minor + 1, 0)
        else:
            high = (0, 0, patch + 1)
        return cls(text, low, high)

    def matches(self, version: Version) -> bool:
        return self.low <= version < self.high

    def __str__(self) -> str:
        return self.text


def at_least(version: Version, minimum: str) -> bool:
    """Whether `version` is at or above a partial minimum such as `"0.3"`."""
    m = _PARTIAL.match(minimum.strip()) if isinstance(minimum, str) else None
    if not m:
        raise VersionError(f"invalid minimum version '{minimum}'")
    floor = tuple(int(g) if g is not None else 0 for g in m.groups())
    return version >= floor
