"""Phase 7 tests for genomic reference access and sequence window extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tcpe.sequence_windows import (
    ENSEMBL_GRCH38_CHR22_URL,
    GenomeReference,
    SequenceWindowConfig,
    SequenceWindowExtractor,
    normalize_chrom_name,
    reverse_complement,
)


def _write_mock_fasta(path: Path, chrom22_length: int = 3200) -> Path:
    chrom22_seq = ("ACGT" * ((chrom22_length // 4) + 1))[:chrom22_length]
    chrom1_seq = ("GATTACA" * 400)[:2500]
    path.write_text(
        "\n".join(
            [
                ">chr22",
                chrom22_seq,
                ">chr1",
                chrom1_seq,
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_local_reference_url_points_to_release110_chr22() -> None:
    assert "release-110" in ENSEMBL_GRCH38_CHR22_URL
    assert "chromosome.22" in ENSEMBL_GRCH38_CHR22_URL


def test_reference_abstraction_loads_fasta_and_normalizes_chrom_names(tmp_path: Path) -> None:
    fasta_path = _write_mock_fasta(tmp_path / "mock.fa")
    ref = GenomeReference.from_fasta(fasta_path)

    assert normalize_chrom_name("chr22") == "22"
    assert normalize_chrom_name("22") == "22"
    assert "22" in ref.available_chromosomes()
    assert ref.get_chrom_length("chr22") == 3200
    assert ref.get_chrom_length("22") == 3200


def test_window_bounds_handle_chromosome_edges(tmp_path: Path) -> None:
    fasta_path = _write_mock_fasta(tmp_path / "mock.fa", chrom22_length=3200)
    reference = GenomeReference.from_fasta(fasta_path)
    extractor = SequenceWindowExtractor(
        reference=reference,
        config=SequenceWindowConfig(window_size=1000),
    )

    near_start = extractor.extract_single(
        {"perturbation_id": "p_start", "chrom": "chr22", "start": 5, "end": 5, "strand": "+"}
    )
    assert near_start.status == "ok"
    assert near_start.window_start == 1
    assert near_start.window_end == 1000

    near_end = extractor.extract_single(
        {"perturbation_id": "p_end", "chrom": "chr22", "start": 3195, "end": 3195, "strand": "+"}
    )
    assert near_end.status == "ok"
    assert near_end.window_start == 2201
    assert near_end.window_end == 3200


def test_minus_strand_windows_are_reverse_complements(tmp_path: Path) -> None:
    fasta_path = _write_mock_fasta(tmp_path / "mock.fa", chrom22_length=3500)
    reference = GenomeReference.from_fasta(fasta_path)
    extractor = SequenceWindowExtractor(
        reference=reference,
        config=SequenceWindowConfig(window_size=1200, min_window_size=1000, max_window_size=2000),
    )

    plus_record = extractor.extract_single(
        {"perturbation_id": "p_plus", "chrom": "chr22", "start": 1800, "end": 1810, "strand": "+"}
    )
    minus_record = extractor.extract_single(
        {"perturbation_id": "p_minus", "chrom": "chr22", "start": 1800, "end": 1810, "strand": "-"}
    )
    assert minus_record.status == "ok"
    assert plus_record.status == "ok"
    assert minus_record.sequence == reverse_complement(plus_record.sequence)


def test_missing_coordinates_or_chromosome_produce_fallback_reason(tmp_path: Path) -> None:
    fasta_path = _write_mock_fasta(tmp_path / "mock.fa")
    reference = GenomeReference.from_fasta(fasta_path)
    extractor = SequenceWindowExtractor(
        reference=reference,
        config=SequenceWindowConfig(window_size=1000),
    )

    missing_coord = extractor.extract_single(
        {"perturbation_id": "p1", "chrom": "chr22", "start": -1, "end": -1, "strand": "+"}
    )
    unknown_chrom = extractor.extract_single(
        {"perturbation_id": "p2", "chrom": "chr99", "start": 100, "end": 120, "strand": "+"}
    )

    assert missing_coord.status == "fallback"
    assert "coordinates" in missing_coord.reason
    assert unknown_chrom.status == "fallback"
    assert "chromosome" in unknown_chrom.reason


def test_cache_is_deterministic_and_hit_count_increases(tmp_path: Path) -> None:
    fasta_path = _write_mock_fasta(tmp_path / "mock.fa")
    reference = GenomeReference.from_fasta(fasta_path)
    extractor = SequenceWindowExtractor(
        reference=reference,
        config=SequenceWindowConfig(window_size=1000),
    )

    first = extractor.extract_single(
        {"perturbation_id": "pA", "chrom": "chr22", "start": 1500, "end": 1500, "strand": "+"}
    )
    second = extractor.extract_single(
        {
            "perturbation_id": "pA_repeated",
            "chrom": "chr22",
            "start": 1500,
            "end": 1500,
            "strand": "+",
        }
    )

    assert first.status == "ok"
    assert second.status == "ok"
    assert first.sequence == second.sequence
    assert extractor.cache_misses == 1
    assert extractor.cache_hits == 1
    assert second.from_cache is True


def test_table_extraction_returns_valid_or_fallback_for_each_perturbation(tmp_path: Path) -> None:
    fasta_path = _write_mock_fasta(tmp_path / "mock.fa")
    reference = GenomeReference.from_fasta(fasta_path)
    extractor = SequenceWindowExtractor(
        reference=reference,
        config=SequenceWindowConfig(window_size=1000),
    )

    perturb_table = pd.DataFrame(
        [
            {
                "perturbation_id": "p_valid_1",
                "chrom": "chr22",
                "start": 1400,
                "end": 1410,
                "strand": "+",
            },
            {
                "perturbation_id": "p_valid_2",
                "chrom": "chr22",
                "start": 2400,
                "end": 2410,
                "strand": "-",
            },
            {
                "perturbation_id": "p_fallback",
                "chrom": "chr22",
                "start": -1,
                "end": -1,
                "strand": "+",
            },
        ]
    )
    windows_df, summary = extractor.extract_from_perturbation_table(perturb_table)

    assert summary.n_total == 3
    assert summary.n_ok == 2
    assert summary.n_fallback == 1
    assert set(windows_df["status"].tolist()) == {"ok", "fallback"}
    fallback_reason = windows_df.loc[
        windows_df["perturbation_id"] == "p_fallback",
        "reason",
    ].iloc[0]
    assert "coordinates" in fallback_reason
