"""Multi-engine (consensus) Percolator feature construction for the merged
idparquet.

This module is intentionally dependency-free (no pyopenms / ms2rescore / deeplc
imports) so the merge logic can be unit-tested in isolation, the same way the
original engine-source indicators were.

The defect this fixes
---------------------
When quantms searches with more than one engine (``comet,msgf``, ``comet,sage``,
``msgf,sage``, ``comet,msgf,sage``) their PSMs are merged into a single
idparquet and rescored by *one* joint PercolatorAdapter. The previous merged
representation was defective in two ways:

1. **Feature collapse.** ``extra_features`` (the list PercolatorAdapter uses)
   was reduced to only the handful of primary raw scores
   (``MS:1002049`` msgf RawScore, ``MS:1002052`` msgf SpecEValue,
   ``MS:1002252`` comet xcorr, ``MS:1002257`` comet expect,
   ``ln(hyperscore)`` sage). The ~30 rich per-engine features that OpenMS
   already stores in ``psm_metavalues`` (comet deltaCn/spscore/lnExpect...,
   MS-GF+ ExplainedIonCurrentRatio, Sage longest_b/longest_y/matched_peaks,
   error statistics, ...) were dropped.

2. **Global worst-case imputation of the primaries.** For a single-engine PSM
   the *other* engine's primary scores were filled with one global worst-case
   constant, so the feature space was largely constant and Percolator could not
   tell "engine did not identify this PSM" from "engine scored it badly".

~80% of PSMs in a two-engine run are single-engine, and single-engine PSMs are
~45% decoy (vs ~8% decoy for PSMs found by *both* engines). With a
half-constant 4-feature space Percolator cannot separate that single-engine
decoy flood, so at the protein level the target:decoy ratio never clears 1% and
picked protein FDR (EPIFANY) returns zero proteins ("No proteins left after FDR
filtering", exit 8).

The fix (evidence-based; see the experiment writeup)
----------------------------------------------------
Feed the *union* of every configured engine's rich, numeric features to the
single joint Percolator, with:

* **orientation-aware, per-feature worst-case imputation** for the missing
  engine (each feature imputed to its own worst value, not one global constant);
* **always-defined engine-source indicators** (``CONSENSUS:comet`` /
  ``CONSENSUS:msgf`` / ``CONSENSUS:sage``, 0/1 on every PSM) so Percolator has
  an explicit, never-missing signal for which engine(s) identified the PSM.

On MSV000085836_v2 this feature union recovered more target proteins at 1%
protein FDR (classic *and* picked) than the comet-only reference and clearly
more than the collapsed 4-feature baseline, while the collapsed baseline scored
*below* comet-only. String/categorical metavalues (``protein_references``,
``AssumedDissociationMethod``) are intentionally excluded so PercolatorAdapter
receives a purely numeric feature table.

Engines are described by :data:`ENGINE_REGISTRY`. Adding an engine means adding
one :class:`EngineSpec`; nothing else in the pipeline needs to change.

Feature orientation
-------------------
``orientation`` maps a feature name to ``True`` (higher is better), ``False``
(lower is better) or ``None`` (**no confidence ordering**). The first two are
imputed to the worst value observed for that feature. ``None`` features are
imputed to the *midpoint* of their observed range — a deliberately
non-informative fill, because imputing an extreme for a feature that does not
encode confidence would inject a false signal in whichever direction we
guessed. This is used for Sage's ``scored_candidates`` (a property of the
spectrum's search space, not of PSM quality), where an extreme fill in either
direction would be an unjustified claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Suffixes used to stash the running range of an unordered (``None``)
# orientation feature inside the same accumulator dict, so the accumulator stays
# a plain ``dict`` and callers need no special type.
_MIN_SUFFIX = "::__min__"
_MAX_SUFFIX = "::__max__"


@dataclass(frozen=True)
class EngineSpec:
    """Everything the consensus builder needs to know about one search engine.

    Parameters
    ----------
    label
        The engine name as it appears in ``idparquet_reader.merge_search_engines``
        (e.g. ``"MS-GF+"``).
    key
        Short lowercase token used to build the indicator feature name
        (``CONSENSUS:<key>``).
    presence_markers
        Metavalues that indicate this engine genuinely identified the PSM.
        Presence detection always runs on the *raw* metavalues before
        imputation, so any of the engine's own features are valid markers.
    orientation
        Curated numeric feature -> ``True`` (higher better) / ``False`` (lower
        better) / ``None`` (unordered, midpoint-imputed).
    """

    label: str
    key: str
    presence_markers: Tuple[str, ...]
    orientation: Dict[str, Optional[bool]]

    @property
    def indicator(self) -> str:
        return f"CONSENSUS:{self.key}"


COMET_FEATURE_ORIENTATION: Dict[str, Optional[bool]] = {
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
MSGF_FEATURE_ORIENTATION: Dict[str, Optional[bool]] = {
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
# Sage feature names as emitted into psm_metavalues by quantms (the ``psm_file``
# rows of Sage's feature table; see ``sage_feature.py``). ``ln(hyperscore)`` is
# the primary score and is written without the ``SAGE:`` prefix by the merger.
SAGE_FEATURE_ORIENTATION: Dict[str, Optional[bool]] = {
    "ln(hyperscore)": True,               # primary
    "SAGE:ln(delta_next)": True,          # bigger gap to the runner-up = better
    "SAGE:ln(matched_intensity_pct)": True,
    "SAGE:matched_peaks": True,
    "SAGE:longest_b": True,
    "SAGE:longest_y": True,
    "SAGE:longest_y_pct": True,
    # Number of candidates scored for the spectrum. This describes the search
    # space, not the quality of the match, so it has no confidence ordering:
    # imputing either extreme would assert something we have not measured.
    # Midpoint-imputed instead (see module docstring).
    "SAGE:scored_candidates": None,
}

ENGINE_REGISTRY: Dict[str, EngineSpec] = {
    "Comet": EngineSpec(
        label="Comet",
        key="comet",
        presence_markers=("COMET:xcorr", "COMET:deltaCn", "COMET:spscore"),
        orientation=COMET_FEATURE_ORIENTATION,
    ),
    "MS-GF+": EngineSpec(
        label="MS-GF+",
        key="msgf",
        presence_markers=("MS:1002050", "ExplainedIonCurrentRatio"),
        orientation=MSGF_FEATURE_ORIENTATION,
    ),
    "Sage": EngineSpec(
        label="Sage",
        key="sage",
        presence_markers=(
            "SAGE:matched_peaks",
            "SAGE:longest_b",
            "SAGE:longest_y",
            "SAGE:ln(delta_next)",
        ),
        orientation=SAGE_FEATURE_ORIENTATION,
    ),
}

DEFAULT_ENGINES: Tuple[str, ...] = ("Comet", "MS-GF+")

# Backwards-compatible aliases (the two-engine API predates the registry).
CONSENSUS_COMET_FEATURE = ENGINE_REGISTRY["Comet"].indicator
CONSENSUS_MSGF_FEATURE = ENGINE_REGISTRY["MS-GF+"].indicator
CONSENSUS_SAGE_FEATURE = ENGINE_REGISTRY["Sage"].indicator
COMET_PRESENCE_MARKERS: Tuple[str, ...] = ENGINE_REGISTRY["Comet"].presence_markers
MSGF_PRESENCE_MARKERS: Tuple[str, ...] = ENGINE_REGISTRY["MS-GF+"].presence_markers
SAGE_PRESENCE_MARKERS: Tuple[str, ...] = ENGINE_REGISTRY["Sage"].presence_markers

UNION_FEATURE_ORIENTATION: Dict[str, Optional[bool]] = {
    **COMET_FEATURE_ORIENTATION,
    **MSGF_FEATURE_ORIENTATION,
}


def supported_engines() -> Set[str]:
    """Engine labels this module can build consensus features for."""
    return set(ENGINE_REGISTRY)


def is_supported_engine_set(engine_labels: Iterable[str]) -> bool:
    """True when every configured engine has an :class:`EngineSpec`.

    A merge containing an unknown engine must NOT take the consensus path: we
    would impute only the known engines' features and silently leave the unknown
    engine's features missing on the other engines' PSMs.
    """
    labels = list(engine_labels or [])
    return bool(labels) and set(labels) <= supported_engines()


def _specs(engine_labels: Optional[Sequence[str]] = None) -> List[EngineSpec]:
    labels = list(engine_labels) if engine_labels else list(DEFAULT_ENGINES)
    return [ENGINE_REGISTRY[label] for label in labels if label in ENGINE_REGISTRY]


def union_feature_orientation(
    engine_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Optional[bool]]:
    """Merged feature -> orientation map for the configured engines."""
    merged: Dict[str, Optional[bool]] = {}
    for spec in _specs(engine_labels):
        merged.update(spec.orientation)
    return merged


def indicator_names(engine_labels: Optional[Sequence[str]] = None) -> Set[str]:
    """``CONSENSUS:<engine>`` indicator names for the configured engines."""
    return {spec.indicator for spec in _specs(engine_labels)}


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


def detect_engines(
    psm_metavalues,
    engine_labels: Optional[Sequence[str]] = None,
) -> Dict[str, bool]:
    """Detect which of the configured engines genuinely identified a PSM.

    Must be called on the RAW metavalues, before :func:`impute_union_features`
    adds the missing engines' features.

    Returns
    -------
    dict
        engine label -> ``True``/``False``.
    """
    return {
        spec.label: any(_has_meta(psm_metavalues, m) for m in spec.presence_markers)
        for spec in _specs(engine_labels)
    }


def detect_engine_sources(psm_metavalues) -> Tuple[bool, bool]:
    """Two-engine compatibility wrapper returning ``(comet, msgf)``.

    Prefer :func:`detect_engines`, which supports any registered engine set.
    """
    found = detect_engines(psm_metavalues, engine_labels=("Comet", "MS-GF+"))
    return found["Comet"], found["MS-GF+"]


def update_worst_case(
    psm_metavalues,
    worst: Dict[str, float],
    orientation: Optional[Dict[str, Optional[bool]]] = None,
) -> Dict[str, float]:
    """Accumulate per-feature imputation constants in-place.

    For ordered features ``worst`` maps feature name -> the worst (least
    confident) value seen so far: the minimum for higher-is-better features, the
    maximum otherwise. For unordered (``None``) features it maps to the midpoint
    of the observed range, tracked via two private companion keys.

    Only values genuinely present on a PSM contribute; imputed constants never
    do, because this must be run in a first pass over the *raw* merged
    metavalues.
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
        if higher_better is None:
            lo_key, hi_key = name + _MIN_SUFFIX, name + _MAX_SUFFIX
            lo = min(worst.get(lo_key, value), value)
            hi = max(worst.get(hi_key, value), value)
            worst[lo_key], worst[hi_key] = lo, hi
            worst[name] = (lo + hi) / 2.0
        elif name not in worst:
            worst[name] = value
        else:
            worst[name] = min(worst[name], value) if higher_better else max(worst[name], value)
    return worst


def impute_union_features(
    psm_metavalues,
    worst: Dict[str, float],
    orientation: Optional[Dict[str, Optional[bool]]] = None,
) -> List:
    """Ensure every union feature is defined on ``psm_metavalues``.

    Any union feature missing from this PSM (i.e. belonging to an engine that
    did not identify it) is filled with that feature's own constant from
    ``worst``. Idempotent: features already present are left untouched.
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
    presence: Optional[Dict[str, bool]] = None,
    engine_labels: Optional[Sequence[str]] = None,
) -> Tuple[List, Set[str]]:
    """Append the engine-source indicator features to ``psm_metavalues``.

    The indicators are always defined (0 or 1) for every PSM, so Percolator
    never sees a missing value for them. Idempotent.

    ``presence`` (engine label -> bool) is the N-engine form. The
    ``comet_present`` / ``msgf_present`` keyword arguments are the original
    two-engine API and still work; when both forms are given, ``presence`` wins
    for the engines it names.
    """
    specs = _specs(engine_labels if presence is None else list(presence))
    resolved: Dict[str, bool] = {}
    if presence:
        resolved.update({k: bool(v) for k, v in presence.items()})
    legacy = {"Comet": comet_present, "MS-GF+": msgf_present}
    for label, value in legacy.items():
        if value is not None and label not in resolved:
            resolved[label] = bool(value)

    missing = [s.label for s in specs if s.label not in resolved]
    if missing:
        detected = detect_engines(psm_metavalues, engine_labels=missing)
        resolved.update(detected)

    psm_metavalues = _to_list(psm_metavalues)
    existing = {item.get("name") for item in psm_metavalues if isinstance(item, dict)}
    names: Set[str] = set()
    for spec in specs:
        names.add(spec.indicator)
        if spec.indicator not in existing:
            psm_metavalues.append(
                {
                    "name": spec.indicator,
                    "value": "1" if resolved.get(spec.label) else "0",
                    "value_type": "int",
                }
            )
    return psm_metavalues, names


def build_consensus_features(
    psm_metavalues,
    worst: Dict[str, float],
    comet_present: Optional[bool] = None,
    msgf_present: Optional[bool] = None,
    orientation: Optional[Dict[str, Optional[bool]]] = None,
    presence: Optional[Dict[str, bool]] = None,
    engine_labels: Optional[Sequence[str]] = None,
) -> List:
    """Full per-PSM transform: detect engine origin, impute the missing engines'
    union features at their imputation constants, and add the always-defined
    indicators.

    Detection is done on the raw metavalues BEFORE imputation (imputation adds
    the missing engines' features, so it must not be used as a presence signal).
    """
    labels = list(presence) if presence else (list(engine_labels) if engine_labels else None)
    # Derive the orientation from the configured engines when the caller did not
    # supply one. Without this, asking for e.g. ("Comet","MS-GF+","Sage") while
    # omitting `orientation` emitted the CONSENSUS:sage indicator but imputed
    # only the default Comet/MS-GF+ union, leaving every Sage feature declared in
    # extra_features yet undefined on non-Sage PSMs -- the exact missing-value
    # defect this module exists to prevent.
    if orientation is None and labels:
        orientation = union_feature_orientation(labels)
    if presence is None:
        detected = detect_engines(psm_metavalues, engine_labels=labels)
        presence = dict(detected)
        if comet_present is not None:
            presence["Comet"] = bool(comet_present)
        if msgf_present is not None:
            presence["MS-GF+"] = bool(msgf_present)
    psm_metavalues = impute_union_features(psm_metavalues, worst, orientation)
    psm_metavalues, _ = add_engine_source_features(
        psm_metavalues, presence=presence, engine_labels=labels
    )
    return psm_metavalues


def union_feature_names(
    orientation: Optional[Dict[str, Optional[bool]]] = None,
    engine_labels: Optional[Sequence[str]] = None,
) -> Set[str]:
    """All Percolator feature names this module guarantees on every merged PSM."""
    if orientation is None:
        orientation = (
            union_feature_orientation(engine_labels)
            if engine_labels
            else UNION_FEATURE_ORIENTATION
        )
    return set(orientation) | indicator_names(engine_labels)
