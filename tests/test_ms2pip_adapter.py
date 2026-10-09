"""Regression tests for the OpenMS reader and MS2PIP 4.2 adapter."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from ms2pip.exceptions import NoMatchingSpectraFound
from ms2pip.result import ProcessingResult
from ms2pip.spectrum import ObservedSpectrum
from psm_utils import PSM, PSMList

from quantmsrescore import model_downloader
from quantmsrescore import ms2pip as adapter
from quantmsrescore.constants import PRIMARY_SCORE_IMPUTED
from quantmsrescore.openms import calculate_correlations


def make_psm(spectrum_id="1", rank=1, charge=2):
    peptide = "PEPTIDEK" + (f"/{charge}" if charge else "")
    return PSM(
        peptidoform=peptide,
        spectrum_id=spectrum_id,
        run="run",
        rank=rank,
        score=10.0,
        is_decoy=False,
    )


def make_spectrum():
    return ObservedSpectrum(
        identifier="controllerType=0 controllerNumber=1 scan=1",
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float32),
        intensity=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        precursor_mz=464.7,
        precursor_charge=2,
        retention_time=10.0,
    )


@pytest.mark.parametrize("unit,tolerance", [("Da", 0.02), ("ppm", 20.0)])
def test_matching_preserves_indices_duplicates_units_and_original_psms(
    monkeypatch, unit, tolerance
):
    psms = PSMList(psm_list=[make_psm("absent"), make_psm(), make_psm(rank=2)])
    monkeypatch.setattr(adapter, "read_spectrum_file", lambda _: iter([make_spectrum()]))
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    generator = adapter.MS2PIPAnnotator(ms2_tolerance_unit=unit, processes=1)
    assert "RAYON_NUM_THREADS" not in os.environ

    def correlate(**kwargs):
        matched = kwargs["psms"]
        assert len(matched) == 2
        assert [p.rank for p in matched] == [1, 2]
        assert kwargs["spectrum_file"] is None
        assert kwargs["ms2_tolerance_mode"] == unit
        assert kwargs["ms2_tolerance"] == tolerance
        assert kwargs["processes"] == 1
        assert os.environ["RAYON_NUM_THREADS"] == "1"
        assert all(p.spectrum is not None for p in matched)
        # Upstream may append invalid results out of input order.
        return [ProcessingResult(psm_index=i, psm=matched[i]) for i in [1, 0]]

    monkeypatch.setattr(adapter.ms2pip, "correlate", correlate)
    results = generator.custom_correlate(
        psms,
        "unused.mzML",
        spectrum_id_pattern=r"scan=(\d+)",
        ms2_tolerance=tolerance,
    )
    assert [result.psm_index for result in results] == [1, 2]
    assert results[0].psm is psms[1]
    assert results[1].psm is psms[2]
    assert all(p.spectrum is None for p in psms)
    assert "RAYON_NUM_THREADS" not in os.environ


def test_existing_rayon_thread_setting_is_kept(monkeypatch):
    monkeypatch.setattr(adapter, "read_spectrum_file", lambda _: iter([make_spectrum()]))
    monkeypatch.setenv("RAYON_NUM_THREADS", "3")

    def correlate(**kwargs):
        assert os.environ["RAYON_NUM_THREADS"] == "3"
        return [ProcessingResult(psm_index=0, psm=kwargs["psms"][0])]

    monkeypatch.setattr(adapter.ms2pip, "correlate", correlate)
    adapter.MS2PIPAnnotator(processes=8).custom_correlate(
        PSMList(psm_list=[make_psm()]), "unused.mzML", spectrum_id_pattern=r"scan=(\d+)"
    )
    assert os.environ["RAYON_NUM_THREADS"] == "3"


def test_charge_recovery_does_not_mutate_input(monkeypatch):
    psms = PSMList(psm_list=[make_psm(charge=None)])
    monkeypatch.setattr(adapter, "read_spectrum_file", lambda _: iter([make_spectrum()]))

    def correlate(**kwargs):
        psm = kwargs["psms"][0]
        assert psm.peptidoform.precursor_charge == 2
        return [ProcessingResult(psm_index=0, psm=psm)]

    monkeypatch.setattr(adapter.ms2pip, "correlate", correlate)
    adapter.MS2PIPAnnotator().custom_correlate(
        psms, "unused.mzML", spectrum_id_pattern=r"scan=(\d+)"
    )
    assert psms[0].peptidoform.precursor_charge is None


def test_empty_psms_fail_before_reading_spectra(monkeypatch):
    def unexpected_read(_):
        pytest.fail("Empty input must not scan a spectrum file")

    monkeypatch.setattr(adapter, "read_spectrum_file", unexpected_read)
    with pytest.raises(NoMatchingSpectraFound):
        adapter.MS2PIPAnnotator().custom_correlate(PSMList(psm_list=[]), "unused.mzML")


def test_ion_order_does_not_change_correlation():
    result = SimpleNamespace(
        predicted_intensity={"b": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])},
        observed_intensity={"y": np.array([3.0, 4.0]), "b": np.array([1.0, 2.0])},
    )
    calculate_correlations([result])
    assert result.correlation == pytest.approx(1.0)


@pytest.mark.parametrize("correlations", [[], [None], [np.nan], [0.9, None, np.nan]])
def test_missing_correlations_do_not_pass_model_validation(correlations):
    results = [
        ProcessingResult(psm_index=i, psm=make_psm(), correlation=value)
        for i, value in enumerate(correlations)
    ]
    assert not adapter.MS2PIPAnnotator()._validate_scores(results, 1.0, 0.7, 0.6, True)


def test_model_validation_still_accepts_valid_predictions():
    result = ProcessingResult(psm_index=0, psm=make_psm(), correlation=0.9)
    assert adapter.MS2PIPAnnotator()._validate_scores([result], 1.0, 0.7, 0.6, True)


@pytest.mark.parametrize("higher_better", [False, True])
def test_imputed_primary_scores_do_not_expand_calibration_set(higher_better):
    results = []
    for i in range(200):
        psm = make_psm(str(i))
        psm.score = -float(i) if higher_better else float(i)
        if i >= 100:
            psm.metadata[PRIMARY_SCORE_IMPUTED] = "true"
        results.append(ProcessingResult(
            psm_index=i, psm=psm, correlation=0.9 if i < 15 else 0.1
        ))
    generator = adapter.MS2PIPAnnotator()
    assert generator._validate_scores(results[:100], 0.15, 0.7, 0.7, higher_better)
    assert generator._validate_scores(results, 0.15, 0.7, 0.7, higher_better)
    assert len(results) == 200


def test_all_imputed_primary_scores_cannot_validate_model():
    psm = make_psm()
    psm.metadata[PRIMARY_SCORE_IMPUTED] = "true"
    result = ProcessingResult(psm_index=0, psm=psm, correlation=1.0)
    assert not adapter.MS2PIPAnnotator()._validate_scores([result], 1.0, 0.7, 0.7, False)


def test_features_attach_to_original_index():
    psms = PSMList(psm_list=[make_psm("absent"), make_psm()])
    result = ProcessingResult(
        psm_index=1,
        psm=psms[1],
        predicted_intensity={"b": np.array([-5.0, -3.0, -1.0]), "y": np.array([-6.0, -2.0, -4.0])},
        observed_intensity={"y": np.array([-6.0, -2.2, -4.0]), "b": np.array([-5.0, -3.1, -1.0])},
    )
    generator = adapter.MS2PIPAnnotator(processes=1)
    generator._calculate_features(psms, [result])
    assert not psms[0].rescoring_features
    assert set(psms[1].rescoring_features) == set(generator.feature_names)
    assert len(psms[1].rescoring_features) == 71


def test_model_downloader_uses_upstream_model_registry(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(model_downloader.ms2pip, "download_models", lambda **kw: calls.append(kw))
    model_downloader.download_ms2pip_models(tmp_path)
    assert calls == [{"model_dir": tmp_path}]
