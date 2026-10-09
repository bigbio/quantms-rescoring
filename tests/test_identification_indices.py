"""Regression coverage for source identification indices in merged idparquet."""

from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pyopenms as oms
import pytest
from click.testing import CliRunner

from quantmsrescore.constants import PRIMARY_SCORE_IMPUTED
from quantmsrescore.idparquet_reader import ParquetRescoringReader
from quantmsrescore.psm_clean import psm_feature_clean
from quantmsrescore.utils import ParquetReader


def _write_source(directory, engine, candidates):
    directory.mkdir()
    schemas = ParquetReader(directory)
    score_type = "expect" if engine == "Comet" else "SpecEValue"
    rows = []
    for peptide, scan, identification, hit, rank, rt, mz in candidates:
        rows.append({
            "sequence": peptide,
            "peptidoform": peptide,
            "modifications": [],
            "precursor_charge": 2,
            "is_decoy": False,
            "observed_mz": mz,
            "reference_file_name": "run.mzML",
            "scan": scan,
            "rt": rt,
            "spectrum_reference": f"scan={scan}",
            "score": 0.01,
            "score_type": score_type,
            "higher_score_better": False,
            "hit_index": hit,
            "peptide_identification_index": identification,
            "psm_metavalues": [{"name": "rank", "value": str(rank), "value_type": "int"}],
            "spectrum_metavalues": [{"name": "source", "value": engine, "value_type": "string"}],
            "run_identifier": "shared_run",
        })
    search_params = {
        "run_identifier": "shared_run",
        "search_engine": engine,
        "search_engine_version": "test",
        "date": datetime(2026, 1, 1),
        "score_type": score_type,
        "higher_score_better": False,
        "mass_type": "monoisotopic",
        "precursor_mass_tolerance": 10.0,
        "precursor_mass_tolerance_ppm": True,
        "fragment_mass_tolerance": 0.02,
        "fragment_mass_tolerance_ppm": False,
        "missed_cleavages": 0,
        "fixed_modifications": [],
        "variable_modifications": [],
        "primary_ms_run_paths": ["run.mzML"],
        "metavalues": [],
        "sp_metavalues": [],
    }
    protein = {
        "accession": "P1", "score": 1.0, "rank": 1,
        "run_identifier": "shared_run", "metavalues": [],
    }
    for name, records, schema in [
        ("psms", rows, schemas.psm_schema),
        ("search_params", [search_params], schemas.search_params_schema),
        ("proteins", [protein], schemas.proteins_schema),
        ("protein_groups", [], schemas.protein_groups_schema),
    ]:
        pq.write_table(pa.Table.from_pylist(records, schema=schema), directory / f"{name}.parquet")
    return rows


@pytest.fixture
def identification_inputs(tmp_path):
    comet = tmp_path / "comet.idparquet"
    msgf = tmp_path / "msgf.idparquet"
    # Both directories deliberately reuse the run identifier and index 7.
    comet_rows = _write_source(comet, "Comet", [
        ("PEPTIDEK", 1, 7, 0, 1, 10.0, 500.0),
        ("ACDEFGK", 2, 8, 0, 1, 20.0, 600.0),
    ])
    msgf_rows = _write_source(msgf, "MS-GF+", [
        ("ACDEFGK", 2, 7, 0, 1, 20.125, 600.2),  # Duplicate: Comet metadata wins.
        ("GHIKLMNR", 2, 7, 2, 3, 20.125, 600.2),
        ("WYSKPEPR", 2, 7, 1, 2, 20.125, 600.2),  # Physical order differs from rank.
    ])

    experiment = oms.MSExperiment()
    for scan, rt, mz in [(1, 10.0, 500.0), (2, 20.0, 600.0)]:
        spectrum = oms.MSSpectrum()
        spectrum.setNativeID(f"scan={scan}")
        spectrum.setMSLevel(2)
        spectrum.setRT(rt)
        spectrum.set_peaks(([100.0, 200.0], [10.0, 20.0]))
        precursor = oms.Precursor()
        precursor.setMZ(mz)
        precursor.setCharge(2)
        spectrum.setPrecursors([precursor])
        experiment.addSpectrum(spectrum)
    mzml = tmp_path / "run.mzML"
    oms.MzMLFile().store(str(mzml), experiment)
    return comet, msgf, mzml, comet_rows, msgf_rows


def _clean(sources, mzml, output):
    args = [arg for source in sources for arg in ("--idparquet", str(source))]
    args.extend(["--mzml", str(mzml), "--output", str(output)])
    result = CliRunner().invoke(psm_feature_clean, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    return pq.read_table(output / "psms.parquet")


@pytest.mark.parametrize("reverse_sources", [False, True])
def test_merge_preserves_retained_source_identifications(
    tmp_path, identification_inputs, reverse_sources
):
    comet, msgf, mzml, comet_rows, msgf_rows = identification_inputs
    sources = [msgf, comet] if reverse_sources else [comet, msgf]
    table = _clean(sources, mzml, tmp_path / "merged.idparquet")
    rows = table.to_pylist()
    expected = comet_rows + msgf_rows[1:]

    # Check the full retained candidate sequence, including cross-engine deduplication.
    for field in ("peptidoform", "spectrum_reference", "rt", "observed_mz", "spectrum_metavalues"):
        assert [row[field] for row in rows] == [row[field] for row in expected]
    ranks = [
        next(mv["value"] for mv in row["psm_metavalues"] if mv["name"] == "rank")
        for row in rows
    ]
    assert ranks == ["1", "1", "3", "2"]

    ids = [row["peptide_identification_index"] for row in rows]
    # Distinct spectra must not collide; same-spectrum source groups also stay separate.
    assert len(set(ids[:3])) == 3
    assert ids[2] == ids[3]
    assert [row["hit_index"] for row in rows] == [0, 0, 0, 1]
    assert table.schema.field("peptide_identification_index").type == pa.int32()
    assert table.schema.field("hit_index").type == pa.int32()


def test_single_source_keeps_original_indices(tmp_path, identification_inputs):
    _, msgf, mzml, _, original = identification_inputs
    rows = _clean([msgf], mzml, tmp_path / "single.idparquet").to_pylist()
    for field in ("peptidoform", "peptide_identification_index", "hit_index", "psm_metavalues"):
        assert [row[field] for row in rows] == [row[field] for row in original]


def test_only_imputed_primary_scores_are_marked_for_calibration(identification_inputs):
    comet, msgf, mzml, _, _ = identification_inputs
    reader = ParquetRescoringReader([comet, msgf], mzml)
    # An observed worst score and its imputed replacement can be identical.
    # Eligibility must follow provenance, rather than a numeric cutoff.
    assert [psm.score for psm in reader.psms] == [0.01] * 4
    assert [psm.metadata.get(PRIMARY_SCORE_IMPUTED) for psm in reader.psms] == [
        None, None, "true", "true"
    ]
