"""Regression tests for ParquetRescoringReader.get_default_scores.

get_default_scores accumulates the per-engine worst primary score across PSMs.
It did `float(get_meta_features(psm_metavalues, "MS:1002252"))` for Comet, but
get_meta_features returns None when a PSM lacks that metavalue (seen with
ms2rescore-annotated phospho/ETD data), and float(None) raises TypeError,
killing the whole run at the SNR spectrum2feature step (idparquet_reader.py:802,
test_dda_id_fine_tuning). A single PSM missing its primary score must not crash.

Constructed via __new__ to avoid the pyopenms/mzML dependency of __init__.
"""

import math
import numpy as np
import pytest

from quantmsrescore.idparquet_reader import ParquetRescoringReader


def _reader():
    r = ParquetRescoringReader.__new__(ParquetRescoringReader)
    r.min_msgf_RawScore = np.inf
    r.max_msgf_EValue = -np.inf
    r.max_comet_expectation_value = -np.inf
    r.min_comet_xcorr = np.inf
    r.min_sage_hyperscore = np.inf
    return r


def _mv(name, value):
    return {"name": name, "value": value}


class TestAsFloat:
    def test_none_returns_none_not_raises(self):
        assert ParquetRescoringReader._as_float(None) is None

    def test_numbers_and_strings(self):
        assert ParquetRescoringReader._as_float("1.5") == 1.5
        assert ParquetRescoringReader._as_float(2) == 2.0

    def test_garbage_returns_none(self):
        assert ParquetRescoringReader._as_float("n/a") is None


class TestCometMissingXcorr:
    def test_missing_ms1002252_does_not_crash(self):
        r = _reader()
        # a Comet PSM WITHOUT MS:1002252 -> get_meta_features returns None
        r.get_default_scores(
            {"search_engine": "Comet"},
            [_mv("COMET:deltaCn", "0.3")],   # no MS:1002252
            {"score": "0.01"},
        )
        # xcorr contributes nothing; seed stays inf; expectation still tracked
        assert math.isinf(r.min_comet_xcorr)
        assert r.max_comet_expectation_value == 0.01

    def test_present_xcorr_is_tracked(self):
        r = _reader()
        r.get_default_scores(
            {"search_engine": "Comet"},
            [_mv("MS:1002252", "1.8")],
            {"score": "0.02"},
        )
        assert r.min_comet_xcorr == 1.8
        assert r.max_comet_expectation_value == 0.02

    def test_mixed_psms_take_the_worst_of_the_present_ones(self):
        r = _reader()
        for mv, score in (([_mv("MS:1002252", "2.0")], "0.01"),
                          ([_mv("COMET:deltaCn", "0.1")], "0.5"),   # missing xcorr
                          ([_mv("MS:1002252", "1.2")], "0.2")):
            r.get_default_scores({"search_engine": "Comet"}, mv, {"score": score})
        assert r.min_comet_xcorr == 1.2       # min over the two present values
        assert r.max_comet_expectation_value == 0.5


class TestMsgfMissingRawScore:
    def test_missing_ms1002049_does_not_crash(self):
        r = _reader()
        r.get_default_scores(
            {"search_engine": "MS-GF+"},
            [_mv("ExplainedIonCurrentRatio", "0.4")],   # no MS:1002049
            {"score": "1e-10"},
        )
        assert math.isinf(r.min_msgf_RawScore)
        assert r.max_msgf_EValue == 1e-10


class TestSage:
    def test_sage_uses_score_column(self):
        r = _reader()
        r.get_default_scores({"search_engine": "Sage"}, [], {"score": "3.2"})
        assert r.min_sage_hyperscore == 3.2

    def test_sage_none_score_does_not_crash(self):
        r = _reader()
        r.get_default_scores({"search_engine": "Sage"}, [], {"score": None})
        assert math.isinf(r.min_sage_hyperscore)


import pandas as pd


class TestWorstObservedScore:
    """Regression: _worst_observed_score must not crash on an object-dtype score
    column. np.isfinite raises on object arrays (even without a None), and a
    null score arrives as Python None -> object dtype. This is the last-resort
    sentinel fallback, so a rare degenerate merge must degrade, not crash.
    """

    def _reader_with_scores(self, values, high_score_better):
        r = ParquetRescoringReader.__new__(ParquetRescoringReader)
        r._psms_df = pd.DataFrame({"score": values})
        r.high_score_better = high_score_better
        return r

    def test_object_dtype_with_none_lower_is_better(self):
        r = self._reader_with_scores(pd.Series([0.01, float("inf"), None, 0.5], dtype=object), False)
        assert r._worst_observed_score() == 0.5   # worst (max) of finite, not a crash

    def test_object_dtype_with_none_higher_is_better(self):
        r = self._reader_with_scores(pd.Series([5.0, float("inf"), None, 2.0], dtype=object), True)
        assert r._worst_observed_score() == 2.0   # worst (min) of finite

    def test_all_sentinel_or_null_returns_zero(self):
        r = self._reader_with_scores(pd.Series([float("inf"), None], dtype=object), False)
        assert r._worst_observed_score() == 0.0

    def test_plain_float_column_still_works(self):
        r = self._reader_with_scores([0.01, float("inf"), 0.5], False)
        assert r._worst_observed_score() == 0.5

    def test_no_score_column_returns_zero(self):
        r = ParquetRescoringReader.__new__(ParquetRescoringReader)
        r._psms_df = pd.DataFrame({"other": [1, 2]})
        r.high_score_better = False
        assert r._worst_observed_score() == 0.0
