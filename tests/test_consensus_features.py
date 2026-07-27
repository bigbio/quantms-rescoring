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
# Mirrors the metavalue names Sage actually emits, verified against
# tests/test_data/..._sage_ms2rescore.idXML. Note ln(hyperscore) is absent --
# Sage's primary score lives in the `score` column, not in metavalues -- and
# SAGE:ln(delta_best) is present in the data but deliberately unregistered
# (constant 0.0 at num_hits=1).
SAGE_ONLY = [_mv("SAGE:matched_peaks", "14", "int"), _mv("SAGE:longest_b", "6", "int"),
             _mv("SAGE:longest_y", "8", "int"), _mv("SAGE:ln(delta_next)", "1.4"),
             _mv("SAGE:ln(-poisson)", "2.1"), _mv("SAGE:scored_candidates", "500", "int"),
             _mv("SAGE:longest_y_pct", "0.7"), _mv("SAGE:ln(matched_intensity_pct)", "-0.3"),
             _mv("SAGE:ln(delta_best)", "0.0")]
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
        # No registry feature uses None today (scored_candidates was measured
        # into the higher-is-better direction), but the mechanism must stay
        # correct: a feature whose direction cannot be established is filled
        # with a non-informative midpoint rather than a fabricated extreme.
        orientation = {"UNORDERED:feature": None}
        worst = {}
        for v in ("100", "300", "500"):
            CF.update_worst_case([_mv("UNORDERED:feature", v)], worst, orientation)
        assert worst["UNORDERED:feature"] == 300.0  # midpoint of [100, 500]

    def test_scored_candidates_is_higher_better_per_measurement(self):
        # Measured on the Sage idXML fixture: target mean 193.0 vs decoy 113.7.
        assert CF.SAGE_FEATURE_ORIENTATION["SAGE:scored_candidates"] is True

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
        worst = {}
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)
        declared = CF.union_feature_names(orientation, engine_labels=ALL_THREE, worst=worst)
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


class TestOrientationDefaultsFollowEngineLabels:
    """Regression: build_consensus_features must not emit an engine's indicator
    while leaving that engine's features undefined.

    Passing engine_labels without an explicit orientation used to fall back to
    the Comet/MS-GF+ default union, so a three-engine call emitted
    CONSENSUS:sage but imputed no SAGE feature -- every SAGE name ended up
    declared in extra_features yet missing on non-SAGE PSMs, which is the
    missing-value defect the module exists to prevent.
    """

    def test_engine_labels_alone_impute_that_engines_features(self):
        orientation = CF.union_feature_orientation(ALL_THREE)
        worst = {}
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)

        # NOTE: orientation deliberately omitted -- derived from engine_labels.
        out = CF.build_consensus_features(list(COMET_ONLY), worst, engine_labels=ALL_THREE)
        names = {m["name"] for m in out}
        declared = CF.union_feature_names(engine_labels=ALL_THREE, worst=worst)
        assert declared.issubset(names), declared - names

    def test_presence_alone_also_derives_orientation(self):
        orientation = CF.union_feature_orientation(ALL_THREE)
        worst = {}
        for mvs in (COMET_ONLY, MSGF_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)
        presence = CF.detect_engines(COMET_ONLY, engine_labels=ALL_THREE)
        out = CF.build_consensus_features(list(COMET_ONLY), worst, presence=presence)
        names = {m["name"] for m in out}
        assert CF.union_feature_names(engine_labels=ALL_THREE, worst=worst).issubset(names)

    def test_no_labels_still_defaults_to_two_engine_union(self):
        worst = {}
        for mvs in (COMET_ONLY, MSGF_ONLY):
            CF.update_worst_case(mvs, worst)
        out = CF.build_consensus_features(list(COMET_ONLY), worst)
        names = {m["name"] for m in out}
        assert CF.union_feature_names(worst=worst).issubset(names)
        assert not any(n.startswith("SAGE:") for n in names)


class TestNonFiniteValuesAreRejected:
    """Regression: float() parses "inf"/"-inf"/"nan", so a single degenerate
    metavalue could become THE worst-case constant and be written onto every
    missing-engine PSM. NaN was worse -- min()/max() propagate it depending on
    argument order, so one NaN could poison the accumulator for a whole run.
    The reader's np.isfinite repair only guards the `score` column, never
    features, so update_worst_case is the only place this can be caught.
    """

    def test_negative_inf_does_not_become_the_worst_case(self):
        worst = {}
        CF.update_worst_case([_mv("COMET:xcorr", "2.0")], worst)
        CF.update_worst_case([_mv("COMET:xcorr", "-inf")], worst)
        assert worst["COMET:xcorr"] == 2.0

    def test_positive_inf_ignored_for_lower_is_better(self):
        worst = {}
        CF.update_worst_case([_mv("MS:1002052", "1e-12")], worst)
        CF.update_worst_case([_mv("MS:1002052", "inf")], worst)
        assert worst["MS:1002052"] == 1e-12

    def test_nan_does_not_poison_regardless_of_order(self):
        # NaN first was the poisoning order: min(nan, x) returns nan.
        worst = {}
        CF.update_worst_case([_mv("COMET:xcorr", "nan")], worst)
        CF.update_worst_case([_mv("COMET:xcorr", "2.0")], worst)
        assert worst["COMET:xcorr"] == 2.0

        worst2 = {}
        CF.update_worst_case([_mv("COMET:xcorr", "2.0")], worst2)
        CF.update_worst_case([_mv("COMET:xcorr", "nan")], worst2)
        assert worst2["COMET:xcorr"] == 2.0

    def test_unordered_feature_range_ignores_non_finite(self):
        worst = {}
        orientation = {"UNORDERED:feature": None}
        for v in ("100", "inf", "500", "nan"):
            CF.update_worst_case([_mv("UNORDERED:feature", v)], worst, orientation)
        assert worst["UNORDERED:feature"] == 300.0  # midpoint of [100, 500]

    def test_non_finite_never_reaches_imputed_metavalues(self):
        orientation = CF.union_feature_orientation(ALL_THREE)
        worst = {}
        # a degenerate PSM for every engine
        CF.update_worst_case(COMET_ONLY + [_mv("COMET:xcorr", "-inf")], worst, orientation)
        CF.update_worst_case(MSGF_ONLY + [_mv("MS:1002049", "nan")], worst, orientation)
        CF.update_worst_case(SAGE_ONLY + [_mv("SAGE:longest_b", "inf")], worst, orientation)
        out = CF.build_consensus_features(
            [], worst, orientation=orientation,
            presence={"Comet": False, "MS-GF+": False, "Sage": False})
        for item in out:
            value = float(item["value"])
            assert value == value, item                    # not NaN
            assert abs(value) != float("inf"), item        # not +/-inf


# Comet metavalues as they ACTUALLY appear in
# tests/test_data/..._comet.idparquet: MS:1002252/3/5/6/8/9 are present but
# MS:1002257 (expectation) is NOT -- Comet keeps expect in the `score` column
# (score_type=expect, higher_score_better=false). The registry nonetheless
# declares MS:1002257, which is what made the fabricated 0.0 reachable.
COMET_NO_EXPECT = [_mv("COMET:xcorr", "1.2"), _mv("COMET:deltaCn", "0.3"),
                   _mv("COMET:spscore", "180"), _mv("MS:1002252", "1.2"),
                   _mv("MS:1002258", "12"), _mv("MS:1002259", "40")]


class TestDeclaredFeatureInvariant:
    """Regression: a feature the data never provided must not be fabricated.

    impute_union_features used ``worst.get(name, 0.0)``. The registry declares
    Comet MS:1002257 (expectation value), but Comet stores its expectation in
    the ``score`` column and emits no such metavalue -- verified against
    tests/test_data/..._comet.idparquet, whose metavalue names include
    MS:1002252/3/5/6/8/9 but NOT MS:1002257. The accumulator therefore never saw
    it and every PSM was handed MS:1002257 = 0.0, the BEST possible value for a
    lower-is-better expectation, while it was still advertised in
    extra_features.
    """

    def test_feature_never_observed_is_not_fabricated(self):
        worst = {}
        CF.update_worst_case(COMET_NO_EXPECT, worst)     # no MS:1002257 anywhere
        assert "MS:1002257" not in worst
        out = CF.impute_union_features([], worst)
        names = {m["name"] for m in out}
        assert "MS:1002257" not in names

    def test_feature_never_observed_is_not_advertised(self):
        worst = {}
        CF.update_worst_case(COMET_NO_EXPECT, worst)
        declared = CF.union_feature_names(worst=worst)
        assert "MS:1002257" not in declared

    def test_observed_features_are_still_imputed_and_advertised(self):
        worst = {}
        CF.update_worst_case(COMET_NO_EXPECT, worst)
        declared = CF.union_feature_names(worst=worst)
        out = {m["name"] for m in CF.impute_union_features([], worst)}
        for name in ("COMET:xcorr", "COMET:deltaCn", "COMET:spscore"):
            assert name in declared
            assert name in out

    def test_indicators_always_advertised_even_with_no_constants(self):
        # Indicators are computed per PSM (0/1), never imputed, so an empty
        # accumulator must not strip them.
        declared = CF.union_feature_names(worst={})
        assert {"CONSENSUS:comet", "CONSENSUS:msgf"}.issubset(declared)

    def test_unconstrained_reports_exactly_what_was_dropped(self):
        worst = {}
        CF.update_worst_case(COMET_NO_EXPECT, worst)
        dropped = CF.unconstrained_features(worst)
        assert "MS:1002257" in dropped
        assert "COMET:xcorr" not in dropped
        assert dropped == set(CF.UNION_FEATURE_ORIENTATION) - CF.constrained_features(worst)

    def test_non_finite_only_feature_is_dropped_not_fabricated(self):
        # Every observed value non-finite -> no usable constant -> must not be
        # advertised, and must not fall back to 0.0.
        worst = {}
        CF.update_worst_case([_mv("COMET:xcorr", "inf")], worst)
        CF.update_worst_case([_mv("COMET:xcorr", "nan")], worst)
        assert "COMET:xcorr" not in CF.constrained_features(worst)
        assert "COMET:xcorr" not in CF.union_feature_names(worst=worst)
        assert "COMET:xcorr" not in {m["name"] for m in CF.impute_union_features([], worst)}

    def test_declared_set_is_defined_on_every_psm_for_all_engines(self):
        # The end-to-end contract, now with a registry entry that the data never
        # supplies (MS:1002257) present in the orientation map.
        orientation = CF.union_feature_orientation(ALL_THREE)
        worst = {}
        for mvs in (COMET_NO_EXPECT, MSGF_ONLY, SAGE_ONLY):
            CF.update_worst_case(mvs, worst, orientation)
        declared = CF.union_feature_names(orientation, engine_labels=ALL_THREE, worst=worst)
        assert "MS:1002257" not in declared
        for mvs in (COMET_NO_EXPECT, MSGF_ONLY, SAGE_ONLY, COMET_NO_EXPECT + SAGE_ONLY):
            out = CF.build_consensus_features(
                list(mvs), worst, orientation=orientation,
                presence=CF.detect_engines(mvs, engine_labels=ALL_THREE))
            names = {m["name"] for m in out}
            assert declared.issubset(names), declared - names


class TestSageRegistryMatchesRealData:
    """Data-driven guard: the Sage registry must match the metavalue names Sage
    actually emits, checked against the committed Sage idXML fixture.

    This exists because the registry was originally written from the ms2rescore
    feature table and was wrong in three ways: it missed SAGE:ln(-poisson),
    it declared ln(hyperscore) which is not a metavalue at all (Sage's primary
    score lives in the `score` column), and it guessed the direction of
    SAGE:scored_candidates.
    """

    IDXML = ("tests/test_data/TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01"
             "_sage_ms2rescore.idXML")

    def _sage_names(self):
        import os
        import re
        if not os.path.exists(self.IDXML):
            import pytest
            pytest.skip("Sage idXML fixture not available")
        text = open(self.IDXML, encoding="utf-8", errors="ignore").read()
        return {m for m in re.findall(r'name="(SAGE:[^"]+)"', text)}

    def test_every_registered_sage_feature_exists_in_the_data(self):
        emitted = self._sage_names()
        registered = set(CF.SAGE_FEATURE_ORIENTATION)
        assert registered <= emitted, registered - emitted

    def test_hyperscore_is_not_registered_as_a_metavalue(self):
        # Sage's primary score is score_type="hyperscore" in the `score` column.
        # Note the idXML->idparquet conversion DROPS the primary score's
        # UserParam entirely: the comet idXML carries MS:1002257 but the comet
        # idparquet does not. Registering a primary score therefore declares a
        # feature that cannot exist on any merged PSM.
        assert "ln(hyperscore)" not in CF.SAGE_FEATURE_ORIENTATION
        assert "hyperscore" not in CF.SAGE_FEATURE_ORIENTATION

    def test_registry_matches_a_real_sageadapter_idparquet(self):
        """Names verified by running SageAdapter for real (OpenMS
        openms-tools-thirdparty 2026.07.02) on an MSV000085836 TMT mzML against
        the decoy database from that run: 66 PSMs, 67 matched proteins
        (55% target / 44% decoy). The psm_metavalues names it produced were:

            DeltaMass, PTM, SAGE:ln(-poisson), SAGE:ln(delta_best),
            SAGE:ln(delta_next), SAGE:ln(matched_intensity_pct),
            SAGE:longest_b, SAGE:longest_y, SAGE:longest_y_pct,
            SAGE:matched_peaks, SAGE:scored_candidates, protein_references,
            spectrum_q

        with score_type="ln(hyperscore)" and higher_score_better=True.

        Two consequences are locked in below: every registered name really is
        emitted, and ln(hyperscore) is a SCORE TYPE rather than a metavalue.
        """
        emitted_by_sageadapter = {
            "SAGE:ln(-poisson)", "SAGE:ln(delta_best)", "SAGE:ln(delta_next)",
            "SAGE:ln(matched_intensity_pct)", "SAGE:longest_b", "SAGE:longest_y",
            "SAGE:longest_y_pct", "SAGE:matched_peaks", "SAGE:scored_candidates",
        }
        registered = set(CF.SAGE_FEATURE_ORIENTATION)
        assert registered <= emitted_by_sageadapter, registered - emitted_by_sageadapter
        assert emitted_by_sageadapter - registered == {"SAGE:ln(delta_best)"}

    def test_spectrum_q_is_not_registered_as_a_feature(self):
        # SageAdapter also emits `spectrum_q`, a q-value computed from
        # target/decoy competition. Feeding a label-derived quantity to
        # Percolator as a feature would be genuine leakage, so it must stay out
        # of the registry.
        assert not any("spectrum_q" in n for n in CF.SAGE_FEATURE_ORIENTATION)

    def test_deliberately_unregistered_names_are_documented_not_forgotten(self):
        emitted = self._sage_names()
        unregistered = emitted - set(CF.SAGE_FEATURE_ORIENTATION)
        # ln(delta_best) is constant 0.0 at num_hits=1, so it is left out on
        # purpose. Anything else appearing here is drift that needs a decision.
        assert unregistered == {"SAGE:ln(delta_best)"}, unregistered


class TestMetavalueContainerNormalisation:
    """Regression: parquet hands back numpy arrays, which have no unambiguous
    truth value.

    ``search_params["sp_metavalues"] or []`` raised

        ValueError: The truth value of an array with more than one element is
        ambiguous. Use a.any() or a.all()

    which surfaced as test_psm_clean_multi_engine failing with exit_code 1 on a
    real comet+msgf merge -- after the consensus features had already been
    built, so the whole run was lost at the very last step.
    """

    class _FakeArray:
        """Stands in for numpy's ndarray: iterable, but truth-testing raises."""

        def __init__(self, items):
            self._items = list(items)

        def tolist(self):
            return list(self._items)

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

        def __bool__(self):
            raise ValueError(
                "The truth value of an array with more than one element is "
                "ambiguous. Use a.any() or a.all()"
            )

    def test_array_like_is_normalised_without_truth_testing(self):
        arr = self._FakeArray([_mv("COMET:xcorr", "1.2"), _mv("MS:1002258", "12")])
        out = CF.as_metavalue_list(arr)
        assert isinstance(out, list)
        assert [m["name"] for m in out] == ["COMET:xcorr", "MS:1002258"]

    def test_the_old_or_idiom_really_would_have_raised(self):
        # Guards the premise: if this stops raising, the fake no longer models
        # numpy and the regression test above is worthless.
        import pytest
        arr = self._FakeArray([_mv("a", "1"), _mv("b", "2")])
        with pytest.raises(ValueError, match="truth value"):
            arr or []

    def test_none_and_plain_list_pass_through(self):
        assert CF.as_metavalue_list(None) == []
        items = [_mv("x", "1")]
        assert CF.as_metavalue_list(items) is items

    def test_empty_array_like_is_empty_list(self):
        assert CF.as_metavalue_list(self._FakeArray([])) == []
