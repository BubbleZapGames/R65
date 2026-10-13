"""Tests for the cc65 debug-info collector."""

from r65.compiler.codegen.debug_info import DebugInfoCollector


def test_find_or_add_span_reuses_an_identical_span():
    info = DebugInfoCollector()
    first = info.find_or_add_span(0, 0x10, 3)
    assert info.find_or_add_span(0, 0x10, 3) is first
    assert info.find_or_add_span(1, 0x10, 3) is not first
    assert len(info.spans) == 2
