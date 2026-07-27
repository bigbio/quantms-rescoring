"""
Unit tests for the two-engine (comet + msgf) Percolator feature construction.

Covers the engine-source detection, the orientation-aware worst-case
accumulation, the per-feature imputation of the missing engine, the
always-defined engine indicators and the full ``build_consensus_features``
transform. Kept dependency-free (no pyopenms/ms2rescore import chain), mirroring
``quantmsrescore/consensus_features.py``.
"""

from quantmsrescore import consensus_features as CF


def _mv(name, value="1", value_type="double"):
    return {"name": name, "value": value, "value_type": value_type}


# A comet-only PSM carries comet markers + comet features (no msgf aux markers).
COMET_ONLY = [_mv("COMET:xcorr", "1.2"), _mv("COMET:deltaCn", "0.3"),
              _mv("COMET:spscore", "180"), _mv("MS:1002257", "0.01")]
# An msgf-only PSM carries msgf markers + msgf features (no comet markers).
MSGF_ONLY = [_mv("MS:1002050", "220", "int"), _mv("MS:1002049", "180", "int"),
             _mv("MS:1002052", "1e-12"), _mv("ExplainedIonCurrentRatio", "0.4")]
BOTH = COMET_ONLY + MSGF_ONLY


class TestEngineSourceDetection:
    def test_comet_only(self):
        assert CF.detect_engine_sources(COMET_ONLY) == (True, False)

    def test_msgf_only(self):
        assert CF.detect_engine_sources(MSGF_ONLY) == (False, True)

    def test_both_engines(self):
        assert CF.detect_engine_sources(BOTH) == (True, True)

    def test_empty_and_none(self):
        assert CF.detect_engine_sources([]) == (False, False)
        assert CF.detect_engine_sources(None) == (False, False)

    def test_imputed_primary_scores_do_not_count_as_presence(self):
        # A comet-only PSM after imputation carries MS:1002049 / MS:1002052 (msgf
        # primary scores). Those are imputed, not real msgf hits, so msgf must
        # still be detected as absent.
        imputed = COMET_ONLY + [_mv("MS:1002049", "-75", "int"), _mv("MS:1002052", "9e9")]
        assert CF.detect_engine_sources(imputed) == (True, False)


class TestWorstCase:
    def test_orientation_aware_min_and_max(self):
        worst = {}
        # higher-better feature (COMET:xcorr): worst = min
        # lower-better feature (MS:1002257 expect): worst = max
        CF.update_worst_case([_mv("COMET:xcorr", "2.0"), _mv("MS:1002257", "0.01")], worst)
        CF.update_worst_case([_mv("COMET:xcorr", "0.5"), _mv("MS:1002257", "5.0")], worst)
        assert worst["COMET:xcorr"] == 0.5      # min for higher-better
        assert worst["MS:1002257"] == 5.0       # max for lower-better

    def test_non_numeric_and_missing_ignored(self):
        worst = {}
        CF.update_worst_case([_mv("COMET:xcorr", "not_a_number")], worst)
        CF.update_worst_case([_mv("protein_references", "P1;P2")], worst)
        assert "COMET:xcorr" not in worst
        assert "protein_references" not in worst


class TestImputation:
    def test_missing_engine_features_filled_with_worst(self):
        worst = {name: 1.0 for name in CF.UNION_FEATURE_ORIENTATION}
        out = CF.impute_union_features(list(COMET_ONLY), worst)
        by_name = {m["name"]: m["value"] for m in out}
        # every union feature is now present
        for name in CF.UNION_FEATURE_ORIENTATION:
            assert name in by_name
        # a genuinely-present comet feature keeps its real value ...
        assert by_name["COMET:xcorr"] == "1.2"
        # ... while a missing msgf feature is imputed to its worst-case constant
        assert float(by_name["MS:1002049"]) == 1.0

    def test_idempotent(self):
        worst = {name: 0.0 for name in CF.UNION_FEATURE_ORIENTATION}
        once = CF.impute_union_features(list(MSGF_ONLY), worst)
        twice = CF.impute_union_features(list(once), worst)
        names_once = [m["name"] for m in once]
        names_twice = [m["name"] for m in twice]
        assert len(names_once) == len(names_twice)
        for name in CF.UNION_FEATURE_ORIENTATION:
            assert names_twice.count(name) == 1


class TestEngineIndicators:
    def test_indicators_for_comet_only(self):
        mvs, feats = CF.add_engine_source_features(list(COMET_ONLY))
        by_name = {m["name"]: m["value"] for m in mvs}
        assert by_name[CF.CONSENSUS_COMET_FEATURE] == "1"
        assert by_name[CF.CONSENSUS_MSGF_FEATURE] == "0"
        assert feats == {CF.CONSENSUS_COMET_FEATURE, CF.CONSENSUS_MSGF_FEATURE}

    def test_indicators_for_msgf_only(self):
        mvs, _ = CF.add_engine_source_features(list(MSGF_ONLY))
        by_name = {m["name"]: m["value"] for m in mvs}
        assert by_name[CF.CONSENSUS_COMET_FEATURE] == "0"
        assert by_name[CF.CONSENSUS_MSGF_FEATURE] == "1"

    def test_explicit_presence_overrides_detection(self):
        mvs, _ = CF.add_engine_source_features([], comet_present=False, msgf_present=True)
        by_name = {m["name"]: m["value"] for m in mvs}
        assert by_name[CF.CONSENSUS_COMET_FEATURE] == "0"
        assert by_name[CF.CONSENSUS_MSGF_FEATURE] == "1"

    def test_always_defined_never_missing(self):
        for mvs in ([], COMET_ONLY, MSGF_ONLY, BOTH):
            out, _ = CF.add_engine_source_features(list(mvs))
            names = {m["name"] for m in out}
            assert CF.CONSENSUS_COMET_FEATURE in names
            assert CF.CONSENSUS_MSGF_FEATURE in names

    def test_idempotent_no_duplicates(self):
        mvs, _ = CF.add_engine_source_features(list(COMET_ONLY))
        mvs, _ = CF.add_engine_source_features(mvs)
        names = [m["name"] for m in mvs]
        assert names.count(CF.CONSENSUS_COMET_FEATURE) == 1
        assert names.count(CF.CONSENSUS_MSGF_FEATURE) == 1


class TestBuildConsensusFeatures:
    def test_comet_only_end_to_end(self):
        worst = {name: 0.0 for name in CF.UNION_FEATURE_ORIENTATION}
        out = CF.build_consensus_features(list(COMET_ONLY), worst)
        by_name = {m["name"]: m["value"] for m in out}
        # every union feature + both indicators are defined
        for name in CF.union_feature_names():
            assert name in by_name
        # engine indicators reflect comet-only origin
        assert by_name[CF.CONSENSUS_COMET_FEATURE] == "1"
        assert by_name[CF.CONSENSUS_MSGF_FEATURE] == "0"
        # real comet feature preserved; msgf feature imputed
        assert by_name["COMET:xcorr"] == "1.2"
        assert float(by_name["MS:1002049"]) == 0.0

    def test_both_engines_preserves_real_values(self):
        worst = {name: -999.0 for name in CF.UNION_FEATURE_ORIENTATION}
        out = CF.build_consensus_features(list(BOTH), worst)
        by_name = {m["name"]: m["value"] for m in out}
        assert by_name["COMET:xcorr"] == "1.2"
        assert by_name["MS:1002049"] == "180"      # not overwritten by worst-case
        assert by_name[CF.CONSENSUS_COMET_FEATURE] == "1"
        assert by_name[CF.CONSENSUS_MSGF_FEATURE] == "1"


class TestMergedEngineDetection:
    def test_labels(self):
        assert CF.is_merged_engine("quantms-rescoring") is True
        assert CF.is_merged_engine("quantms-consensus-rescoring") is True
        assert CF.is_merged_engine("Comet") is False
        assert CF.is_merged_engine("MS-GF+") is False
        assert CF.is_merged_engine(None) is False


class TestUnionFeatureNames:
    def test_includes_features_and_indicators(self):
        names = CF.union_feature_names()
        assert CF.CONSENSUS_COMET_FEATURE in names
        assert CF.CONSENSUS_MSGF_FEATURE in names
        assert "COMET:xcorr" in names
        assert "MS:1002049" in names
        # no string/categorical metavalues leak in
        assert "protein_references" not in names
        assert "AssumedDissociationMethod" not in names


# ---------------------------------------------------------------------------
# SAGE / N-engine coverage (bigbio/quantms#717)
#
# Before the registry refactor the consensus path was gated on the engine set
# being exactly {Comet, MS-GF+}, so every SAGE-involving merge silently fell
# back to the primary-score fill -- including comet,msgf,sage, which meant
# adding SAGE to a working two-engine run regressed it to the collapsed feature
# table that returns zero proteins at 1% protein FDR.
# ---------------------------------------------------------------------------

# A sage-only PSM carries sage markers + sage features.
SAGE_ONLY = [_mv("SAGE:matched_peaks", "14", "int"), _mv("SAGE:longest_b", "6", "int"),
             _mv("SAGE:longest_y", "8", "int"), _mv("SAGE:ln(delta_next)", "1.4"),
             _mv("ln(hyperscore)", "3.2"), _mv("SAGE:scored_candidates", "500", "int")]
COMET_SAGE = ("Comet", "Sage")
MSGF_SAGE = ("MS-GF+", "Sage")
ALL_THREE = ("Comet", "MS-GF+", "Sage")


class TestEngineRegistry:
    def test_sage_is_registered(self):
        assert "Sage" in CF.supported_engines()
        assert CF.ENGINE_REGISTRY["Sage"].indicator == "CONSENSUS:sage"

    def test_all_registered_combinations_are_supported(self):
        for engines in (("Sage",), COMET_SAGE, MSGF_SAGE, ALL_THREE,
                        ("Comet", "MS-GF+")):
            assert CF.is_supported_engine_set(engines), engines

    def test_unknown_engine_disables_consensus_path(self):
        # An unregistered engine must switch the whole merge off rather than
        # produce a union that silently omits its features.
        assert not CF.is_supported_engine_set(("Comet", "Andromeda"))
        assert not CF.is_supported_engine_set([])


class TestSageDetection:
    def test_sage_only(self):
        found = CF.detect_engines(SAGE_ONLY, engine_labels=ALL_THREE)
        assert found == {"Comet": False, "MS-GF+": False, "Sage": True}

    def test_comet_plus_sage(self):
        found = CF.detect_engines(COMET_ONLY + SAGE_ONLY, engine_labels=COMET_SAGE)
        assert found == {"Comet": True, "Sage": True}

    def test_msgf_plus_sage(self):
        found = CF.detect_engines(MSGF_ONLY + SAGE_ONLY, engine_labels=MSGF_SAGE)
        assert found == {"MS-GF+": True, "Sage": True}

    def test_three_engine_single_engine_subsets(self):
        for mvs, expect in ((COMET_ONLY, "Comet"), (MSGF_ONLY, "MS-GF+"), (SAGE_ONLY, "Sage")):
            found = CF.detect_engines(mvs, engine_labels=ALL_THREE)
            assert found[expect] is True
            assert sum(found.values()) == 1, found

    def test_imputed_sage_primary_does_not_count_as_presence(self):
        # ln(hyperscore) is imputed onto non-sage PSMs, so it must NOT be a
        # presence marker -- otherwise every PSM looks sage-identified.
        imputed = COMET_ONLY + [_mv("ln(hyperscore)", "0.01")]
        found = CF.detect_engines(imputed, engine_labels=COMET_SAGE)
        assert found == {"Comet": True, "Sage": False}


class TestSageImputation:
    def test_sage_features_imputed_on_comet_only_psm(self):
        orientation = CF.union_feature_orientation(COMET_SAGE)
        worst = {}
        for mvs in (COMET_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)
        out = CF.impute_union_features(list(COMET_ONLY), worst, orientation)
        names = {m["name"] for m in out}
        # every sage feature is now defined on a comet-only PSM
        assert set(CF.SAGE_FEATURE_ORIENTATION).issubset(names)

    def test_orientation_aware_worst_for_sage(self):
        orientation = CF.union_feature_orientation(("Sage",))
        worst = {}
        CF.update_worst_case([_mv("SAGE:matched_peaks", "20")], worst, orientation)
        CF.update_worst_case([_mv("SAGE:matched_peaks", "5")], worst, orientation)
        # higher-is-better -> worst is the minimum
        assert worst["SAGE:matched_peaks"] == 5.0

    def test_unordered_feature_uses_midpoint_not_an_extreme(self):
        # scored_candidates has no confidence ordering; imputing either extreme
        # would assert a signal we have not measured.
        orientation = CF.union_feature_orientation(("Sage",))
        worst = {}
        for v in ("100", "300", "500"):
            CF.update_worst_case([_mv("SAGE:scored_candidates", v)], worst, orientation)
        assert CF.SAGE_FEATURE_ORIENTATION["SAGE:scored_candidates"] is None
        assert worst["SAGE:scored_candidates"] == 300.0  # midpoint of [100, 500]

    def test_no_inf_or_nan_leaks_into_imputed_values(self):
        orientation = CF.union_feature_orientation(ALL_THREE)
        worst = {}
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY):
            out = CF.build_consensus_features(
                list(mvs), worst, orientation=orientation,
                presence=CF.detect_engines(mvs, engine_labels=ALL_THREE))
            for item in out:
                value = float(item["value"])
                assert value == value, item          # not NaN
                assert abs(value) != float("inf"), item


class TestSageIndicators:
    def test_three_indicators_present_and_consistent(self):
        orientation = CF.union_feature_orientation(ALL_THREE)
        worst = {}
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)
        out = CF.build_consensus_features(
            list(SAGE_ONLY), worst, orientation=orientation,
            presence=CF.detect_engines(SAGE_ONLY, engine_labels=ALL_THREE))
        flags = {m["name"]: m["value"] for m in out if m["name"].startswith("CONSENSUS:")}
        assert flags == {"CONSENSUS:comet": "0", "CONSENSUS:msgf": "0", "CONSENSUS:sage": "1"}

    def test_union_feature_names_cover_sage_and_all_indicators(self):
        names = CF.union_feature_names(engine_labels=ALL_THREE)
        assert set(CF.SAGE_FEATURE_ORIENTATION).issubset(names)
        assert {"CONSENSUS:comet", "CONSENSUS:msgf", "CONSENSUS:sage"}.issubset(names)

    def test_every_declared_feature_is_defined_on_every_psm(self):
        # The contract that makes the merge safe: whatever we declare in
        # extra_features must exist on EVERY merged PSM, whichever engine found it.
        orientation = CF.union_feature_orientation(ALL_THREE)
        declared = CF.union_feature_names(orientation, engine_labels=ALL_THREE)
        worst = {}
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY, COMET_ONLY + SAGE_ONLY):
            out = CF.build_consensus_features(
                list(mvs), worst, orientation=orientation,
                presence=CF.detect_engines(mvs, engine_labels=ALL_THREE))
            names = {m["name"] for m in out}
            assert declared.issubset(names), declared - names

    def test_two_engine_default_unchanged_by_sage_support(self):
        # Regression guard: the comet+msgf union validated on MSV000085836 must
        # not silently gain sage features when sage is not configured.
        names = CF.union_feature_names()
        assert "CONSENSUS:sage" not in names
        assert not any(n.startswith("SAGE:") for n in names)
