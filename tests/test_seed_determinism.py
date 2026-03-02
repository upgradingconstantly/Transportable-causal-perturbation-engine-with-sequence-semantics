"""Determinism tests for Phase 1 runtime helpers."""

from tcpe.runtime.seed import deterministic_probe


def test_deterministic_probe_is_repeatable() -> None:
    sample_a = deterministic_probe(1234)
    sample_b = deterministic_probe(1234)
    assert sample_a == sample_b
