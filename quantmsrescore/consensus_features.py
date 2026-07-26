"""
Two-engine (comet + msgf) Percolator feature construction for the merged
(consensus) idparquet.

This module is intentionally dependency-free (no pyopenms / ms2rescore / deeplc
imports) so the two-engine merge logic can be unit-tested in isolation, the same
way the original engine-source indicators were.

The defect this fixes
---------------------
When quantms searches with two engines (``comet,msgf``) their PSMs are merged
into a single idparquet and rescored by *one* joint PercolatorAdapter. The
previous merged representation was defective in two ways:

1. **Feature collapse.** ``extra_features`` (the list PercolatorAdapter uses)
   was reduced to only the four primary raw scores
   (``MS:1002049`` msgf RawScore, ``MS:1002052`` msgf SpecEValue,
   ``MS:1002252`` comet xcorr, ``MS:1002257`` comet expect). The ~30 rich
   per-engine features that OpenMS already stores in ``psm_metavalues``
   (comet deltaCn/spscore/lnExpect..., MS-GF+ ExplainedIonCurrentRatio,
   error statistics, ...) were dropped.

2. **Global worst-case imputation of the primaries.** For a single-engine PSM
   the *other* engine's two primary scores were filled with one global
   worst-case constant, so the feature space was half constant and Percolator
   could not tell "engine did not identify this PSM" from "engine scored it
   badly".

~80% of PSMs in a two-engine run are single-engine, and single-engine PSMs are
~45% decoy (vs ~8% decoy for PSMs found by *both* engines). With a half-constant
4-feature space Percolator cannot separate that single-engine decoy flood, so at
the protein level the target:decoy ratio never clears 1% and picked protein FDR
(EPIFANY) returns zero proteins ("No proteins left after FDR filtering", exit 8).

The fix (evidence-based; see the experiment writeup)
----------------------------------------------------
Feed the *union* of both engines' rich, numeric features to the single joint
Percolator, with:

* **orientation-aware, per-feature worst-case imputation** for the missing
  engine (each feature imputed to its own worst value, not one global constant);
* **always-defined engine-source indicators** (``CONSENSUS:comet`` /
  ``CONSENSUS:msgf``, 0/1 on every PSM) so Percolator has an explicit,
  never-missing signal for which engine(s) identified the PSM.

On MSV000085836_v2 this feature union recovered more target proteins at 1%
protein FDR (classic *and* picked) than the comet-only reference and clearly
more than the collapsed 4-feature baseline, while the collapsed baseline scored
*below* comet-only. String/categorical metavalues (``protein_references``,
``AssumedDissociationMethod``) are intentionally excluded so PercolatorAdapter
receives a purely numeric feature table.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

# Percolator feature names for the engine-source indicators.
CONSENSUS_COMET_FEATURE = "CONSENSUS:comet"
CONSENSUS_MSGF_FEATURE = "CONSENSUS:msgf"

# Metavalues that are ONLY present when the respective engine genuinely
# identified the PSM. They are never produced by imputation, so they are safe
# presence markers even after imputation has run.
COMET_PRESENCE_MARKERS: Tuple[str, ...] = ("COMET:xcorr", "COMET:deltaCn", "COMET:spscore")
MSGF_PRESENCE_MARKERS: Tuple[str, ...] = ("MS:1002050", "ExplainedIonCurrentRatio")

# Curated, non-redundant numeric feature -> higher_is_better orientation.
# Only numeric metavalues are listed; anything not here is left untouched.
# (These are the exact feature sets validated in the two-engine experiment.)
COMET_FEATURE_ORIENTATION: Dict[str, bool] = {
    "COMET:xcorr": True,          # MS:1002252 (primary)
    "COMET:deltaCn": True,        # MS:1002253
    "COMET:deltaLCn": True,
    "COMET:spscore": True,        # MS:1002255
    "COMET:sprank": False,        # MS:1002256
    "COMET:lnExpect": False,
    "COMET:IonFrac": True,
    "COMET:lnNumSP": True,
    "COMET:lnRankSP": False,
    "Comet:lnrSp": True,
    "num_matched_peptides": True,
    "MS:1002257": False,          # comet expectation value (primary)
    "MS:1002258": True,           # matched ions
    "MS:1002259": True,           # total ions
}
MSGF_FEATURE_ORIENTATION: Dict[str, bool] = {
    "MS:1002049": True,           # RawScore (primary)
    "MS:1002050": True,           # DeNovoScore
    "MS:1002052": False,          # SpecEValue (primary)
    "ExplainedIonCurrentRatio": True,
    "MS2IonCurrent": True,
    "NTermIonCurrentRatio": True,
    "CTermIonCurrentRatio": True,
    "NumMatchedMainIons": True,
    "MeanErrorTop7": False,
    "StdevErrorTop7": False,
    "MeanRelErrorTop7": False,
    "StdevRelErrorTop7": False,
    "MeanErrorAll": False,
    "StdevErrorAll": False,
}
UNION_FEATURE_ORIENTATION: Dict[str, bool] = {
    **COMET_FEATURE_ORIENTATION,
    **MSGF_FEATURE_ORIENTATION,
}


def is_merged_engine(engine_label) -> bool:
    """Return True when the idparquet is a merged multi-engine (consensus) run.

    Matches both the historical ``quantms-rescoring`` label and the deployed
    ``quantms-consensus-rescoring`` label.
    """
    if engine_label is None:
        return False
    label = str(engine_label).lower()
    return label == "quantms-rescoring" or "consensus" in label


def _to_list(psm_metavalues) -> List:
    if psm_metavalues is None:
        return []
    if hasattr(psm_metavalues, "tolist"):
        return psm_metavalues.tolist()
    return list(psm_metavalues)


def _has_meta(psm_metavalues, name: str) -> bool:
    for item in _to_list(psm_metavalues):
        if isinstance(item, dict) and item.get("name") == name:
            return True
    return False


def _meta_value(psm_metavalues, name: str):
    for item in _to_list(psm_metavalues):
        if isinstance(item, dict) and item.get("name") == name:
            return item.get("value")
    return None


def detect_engine_sources(psm_metavalues) -> Tuple[bool, bool]:
    """Detect which engine(s) genuinely identified a PSM.

    Relies on auxiliary, never-imputed markers, so it stays correct even after
    imputation has added the missing engine's features.

    Returns
    -------
    (comet_present, msgf_present) : tuple[bool, bool]
    """
    comet = any(_has_meta(psm_metavalues, m) for m in COMET_PRESENCE_MARKERS)
    msgf = any(_has_meta(psm_metavalues, m) for m in MSGF_PRESENCE_MARKERS)
    return comet, msgf


def update_worst_case(
    psm_metavalues,
    worst: Dict[str, float],
    orientation: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    """Accumulate per-feature orientation-aware worst-case values in-place.

    ``worst`` maps feature name -> the worst (least confident) value seen so far:
    the minimum for higher-is-better features, the maximum otherwise. Only
    values genuinely present on a PSM contribute; imputed constants never do
    because this must be run in a first pass over the *raw* merged metavalues.
    """
    if orientation is None:
        orientation = UNION_FEATURE_ORIENTATION
    for name, higher_better in orientation.items():
        raw = _meta_value(psm_metavalues, name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if name not in worst:
            worst[name] = value
        else:
            worst[name] = min(worst[name], value) if higher_better else max(worst[name], value)
    return worst


def impute_union_features(
    psm_metavalues,
    worst: Dict[str, float],
    orientation: Optional[Dict[str, bool]] = None,
) -> List:
    """Ensure every union feature is defined on ``psm_metavalues``.

    Any union feature missing from this PSM (i.e. belonging to the engine that
    did not identify it) is filled with that feature's own worst-case constant
    from ``worst``. Idempotent: features already present are left untouched.
    """
    if orientation is None:
        orientation = UNION_FEATURE_ORIENTATION
    psm_metavalues = _to_list(psm_metavalues)
    present = {item.get("name") for item in psm_metavalues if isinstance(item, dict)}
    for name in orientation:
        if name in present:
            continue
        value = worst.get(name, 0.0)
        psm_metavalues.append(
            {"name": name, "value": repr(float(value)), "value_type": "double"}
        )
    return psm_metavalues


def add_engine_source_features(
    psm_metavalues,
    comet_present: Optional[bool] = None,
    msgf_present: Optional[bool] = None,
) -> Tuple[List, Set[str]]:
    """Append the two engine-source indicator features to ``psm_metavalues``.

    The indicators are always defined (0 or 1) for every PSM, so Percolator never
    sees a missing value for them. Idempotent.

    Returns
    -------
    (psm_metavalues, feature_names) : tuple[list, set[str]]
    """
    if comet_present is None or msgf_present is None:
        detected_comet, detected_msgf = detect_engine_sources(psm_metavalues)
        if comet_present is None:
            comet_present = detected_comet
        if msgf_present is None:
            msgf_present = detected_msgf

    psm_metavalues = _to_list(psm_metavalues)
    existing = {item.get("name") for item in psm_metavalues if isinstance(item, dict)}
    for name, present in (
        (CONSENSUS_COMET_FEATURE, comet_present),
        (CONSENSUS_MSGF_FEATURE, msgf_present),
    ):
        if name not in existing:
            psm_metavalues.append(
                {"name": name, "value": "1" if present else "0", "value_type": "int"}
            )
    return psm_metavalues, {CONSENSUS_COMET_FEATURE, CONSENSUS_MSGF_FEATURE}


def build_consensus_features(
    psm_metavalues,
    worst: Dict[str, float],
    comet_present: Optional[bool] = None,
    msgf_present: Optional[bool] = None,
    orientation: Optional[Dict[str, bool]] = None,
) -> List:
    """Full per-PSM transform: detect engine origin, impute the missing engine's
    union features at their worst-case, and add the always-defined indicators.

    Detection is done on the raw metavalues BEFORE imputation (imputation adds
    the missing engine's features, so it must not be used as a presence signal).
    """
    if comet_present is None or msgf_present is None:
        dc, dm = detect_engine_sources(psm_metavalues)
        comet_present = dc if comet_present is None else comet_present
        msgf_present = dm if msgf_present is None else msgf_present
    psm_metavalues = impute_union_features(psm_metavalues, worst, orientation)
    psm_metavalues, _ = add_engine_source_features(
        psm_metavalues, comet_present=comet_present, msgf_present=msgf_present
    )
    return psm_metavalues


def union_feature_names(orientation: Optional[Dict[str, bool]] = None) -> Set[str]:
    """All Percolator feature names this module guarantees on every merged PSM."""
    if orientation is None:
        orientation = UNION_FEATURE_ORIENTATION
    return set(orientation) | {CONSENSUS_COMET_FEATURE, CONSENSUS_MSGF_FEATURE}
