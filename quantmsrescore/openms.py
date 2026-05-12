import re
from pathlib import Path
from typing import List, Union, Optional, Tuple, Generator, Dict, Any
from warnings import filterwarnings

filterwarnings(
    "ignore",
    message="OPENMS_DATA_PATH environment variable already exists",
    category=UserWarning,
    module="pyopenms",
)

import numpy as np
import pyopenms as oms
import pandas as pd
from packaging import version
from psm_utils import PSM, PSMList
from pyopenms import (
    SpectrumLookup,
    TheoreticalSpectrumGenerator,
)

from quantmsrescore.constants import (
    DEEPLC_FEATURES,
    MS2PIP_FEATURES,
    OPENMS_DISSOCIATION_METHODS_PATCH_3_3_0,
    OPENMS_DISSOCIATION_METHODS_PATCH_3_1_0,
)
from quantmsrescore.exceptions import MzMLNotUnixException
from quantmsrescore.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

OPENMS_DECOY_FIELD = "target_decoy"
SPECTRUM_PATTERN = r"(spectrum|scan)=(\d+)"

# =============================================================================
# Caching infrastructure for performance
# =============================================================================

# Spectrum file cache (LRU bounded)
_SPECTRUM_FILE_CACHE: Dict[str, Tuple[oms.MSExperiment, SpectrumLookup]] = {}
_SPECTRUM_FILE_ACCESS_ORDER: List[str] = []  # Track access order for LRU eviction
MAX_CACHE_SIZE = 3  # Maximum number of mzML files to keep in cache

# Compiled regex cache to avoid repeated compilation
_REGEX_CACHE: Dict[str, re.Pattern] = {}


def get_compiled_regex(pattern: Optional[str]) -> re.Pattern:
    """
    Get a compiled regex pattern from cache or compile and cache it.

    This prevents repeated regex compilation in loops, which is a common
    performance issue when processing many spectra.

    Parameters
    ----------
    pattern : Optional[str]
        The regex pattern to compile. If None or empty, defaults to "(.*)".

    Returns
    -------
    re.Pattern
        The compiled regex pattern.
    """
    if not pattern:
        pattern = r"(.*)"

    if pattern not in _REGEX_CACHE:
        try:
            _REGEX_CACHE[pattern] = re.compile(pattern)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}. Using default.")
            pattern = r"(.*)"
            if pattern not in _REGEX_CACHE:
                _REGEX_CACHE[pattern] = re.compile(pattern)

    return _REGEX_CACHE[pattern]


def get_cached_spectrum_data(
    mzml_file: Union[str, Path],
    force_reload: bool = False
) -> Tuple[oms.MSExperiment, SpectrumLookup]:
    """
    Get spectrum data from cache or load it.

    This function prevents duplicate loading of the same mzML file when
    multiple feature generators (MS2PIP, AlphaPeptDeep, DeepLC) process
    the same spectrum file.

    The cache is bounded to MAX_CACHE_SIZE (default: 3) entries. When the
    limit is reached, the least recently used entry is evicted. This prevents
    unbounded memory growth when processing many spectrum files sequentially.

    Parameters
    ----------
    mzml_file : Union[str, Path]
        Path to the mzML file.
    force_reload : bool, optional
        If True, reload the file even if cached. Default is False.

    Returns
    -------
    Tuple[oms.MSExperiment, SpectrumLookup]
        Cached or freshly loaded experiment and lookup.

    Notes
    -----
    For explicit memory management, call clear_spectrum_cache() after
    completing annotation on a file or batch of files.
    """
    mzml_path = str(mzml_file) if isinstance(mzml_file, Path) else mzml_file

    if force_reload or mzml_path not in _SPECTRUM_FILE_CACHE:
        logger.debug(f"Loading mzML file into cache: {mzml_path}")

        # Evict oldest entry if cache is full (LRU eviction)
        if len(_SPECTRUM_FILE_CACHE) >= MAX_CACHE_SIZE:
            if _SPECTRUM_FILE_ACCESS_ORDER:
                oldest = _SPECTRUM_FILE_ACCESS_ORDER.pop(0)
                if oldest in _SPECTRUM_FILE_CACHE:
                    del _SPECTRUM_FILE_CACHE[oldest]
                    logger.debug(f"Evicted from cache (LRU): {oldest}")

        exp, lookup = OpenMSHelper.get_spectrum_lookup_indexer(mzml_path)
        _SPECTRUM_FILE_CACHE[mzml_path] = (exp, lookup)
        _SPECTRUM_FILE_ACCESS_ORDER.append(mzml_path)
    else:
        logger.debug(f"Using cached mzML data: {mzml_path}")
        # Move to end of access order (most recently used)
        if mzml_path in _SPECTRUM_FILE_ACCESS_ORDER:
            _SPECTRUM_FILE_ACCESS_ORDER.remove(mzml_path)
            _SPECTRUM_FILE_ACCESS_ORDER.append(mzml_path)

    return _SPECTRUM_FILE_CACHE[mzml_path]


def clear_spectrum_cache(mzml_file: Optional[Union[str, Path]] = None) -> None:
    """
    Clear the spectrum file cache to free memory.

    Call this function after completing annotation on a file or batch of files
    to reclaim memory. The annotator calls this automatically at the end of
    annotation.

    Parameters
    ----------
    mzml_file : Optional[Union[str, Path]], optional
        Specific file to remove from cache. If None, clears entire cache.
    """
    if mzml_file is None:
        _SPECTRUM_FILE_CACHE.clear()
        _SPECTRUM_FILE_ACCESS_ORDER.clear()
        logger.debug("Cleared entire spectrum cache")
    else:
        mzml_path = str(mzml_file) if isinstance(mzml_file, Path) else mzml_file
        if mzml_path in _SPECTRUM_FILE_CACHE:
            del _SPECTRUM_FILE_CACHE[mzml_path]
            if mzml_path in _SPECTRUM_FILE_ACCESS_ORDER:
                _SPECTRUM_FILE_ACCESS_ORDER.remove(mzml_path)
            logger.debug(f"Removed from spectrum cache: {mzml_path}")


# =============================================================================
# Shared utility functions for MS2PIP and AlphaPeptDeep
# =============================================================================
# These functions are used by both alphapeptdeep.py and ms2pip.py to avoid
# code duplication.


def organize_psms_by_spectrum_id(
    enumerated_psm_list:  Union[PSMList, List[Any]]
) -> Dict[str, List[Tuple[int, Any]]]:
    """
    Organize PSMs by spectrum ID for efficient lookup.

    This function creates a dictionary mapping spectrum IDs to lists of
    (index, PSM) tuples, enabling O(1) lookup when iterating through spectra.

    Parameters
    ----------
    enumerated_psm_list : List[Any]
        List of PSMs. Can be either:
        - PSMList: Will be enumerated internally
        - List[Tuple[int, PSM]]: Already enumerated tuples

    Returns
    -------
    Dict[str, List[Tuple[int, Any]]]
        Dictionary mapping spectrum IDs to lists of (index, PSM) tuples.
    """
    from collections import defaultdict
    psms_by_specid = defaultdict(list)

    for item in enumerated_psm_list:
        # Handle both enumerated tuples and raw PSM objects
        if isinstance(item, tuple) and len(item) == 2:
            psm_index, psm = item
        else:
            # Assume it's a PSM object, enumerate on the fly
            psm_index = enumerated_psm_list.psm_list.index(item)
            psm = item

        psms_by_specid[str(psm.spectrum_id)].append((psm_index, psm))

    return psms_by_specid


def calculate_correlations(results: List[Any]) -> None:
    """
    Calculate and add Pearson correlations to list of ProcessingResult objects.

    This function modifies the results in-place, adding a correlation
    attribute to each result based on predicted vs observed intensities.

    Parameters
    ----------
    results : List[Any]
        List of ProcessingResult objects with predicted_intensity and
        observed_intensity attributes.
    """
    for result in results:
        if result.predicted_intensity and result.observed_intensity:
            pred_int = np.concatenate([i for i in result.predicted_intensity.values()])
            obs_int = np.concatenate([i for i in result.observed_intensity.values()])
            result.correlation = np.corrcoef(pred_int, obs_int)[0][1]
        else:
            result.correlation = None
            logger.debug(f"Empty intensities for result: {result}")


class OpenMSHelper:
    """
    This class should contain methods to help with OpenMS operations on PSM lists
    Features, parameters.
    """

    @staticmethod
    def count_decoys_targets(df: pd.DataFrame) -> tuple[int, int]:
        """
        Count target/decoy from parquet DataFrame.
        """
        if "is_decoy" not in df.columns:
            logger.warning("No is_decoy column found")
            return 0, 0

        decoy = int(df["is_decoy"].fillna(False).sum())
        target = len(df) - decoy

        logger.info(
            f"Decoy percentage: {decoy / max(len(df), 1) * 100:.2f}%, "
            f"targets {target}, decoys {decoy}"
        )
        return decoy, target

    @staticmethod
    def get_psm_count(df: pd.DataFrame) -> int:
        """Count rows in PSM parquet table."""
        return len(df)

    @staticmethod
    def get_spectrum_lookup_indexer(
        mzml_file: Union[str, Path]
    ) -> tuple[oms.MSExperiment, SpectrumLookup]:
        """
        Create a SpectrumLookup indexer from an mzML file.

        This method loads an mzML file into an MSExperiment object and
        initializes a SpectrumLookup object to read spectra using a
        specified regular expression pattern for scan numbers.

        Parameters
        ----------
        mzml_file : str
        The path to the mzML file to be loaded.

        Returns
        -------
        tuple: A tuple containing the MSExperiment object with the loaded
        """

        if isinstance(mzml_file, Path):
            mzml_file = str(mzml_file)

        exp = oms.MSExperiment()
        oms.MzMLFile().load(mzml_file, exp)

        lookup = SpectrumLookup()
        if "spectrum=" in exp.getSpectrum(0).getNativeID():
            lookup.readSpectra(exp, "spectrum=(?<SCAN>\\d+)")
        else:
            lookup.readSpectra(exp, "scan=(?<SCAN>\\d+)")

        return exp, lookup

    @staticmethod
    def get_spectrum_for_psm(
        psm: Union[PSM, dict, pd.Series], exp: oms.MSExperiment, lookup: SpectrumLookup
    ) -> Union[None, oms.MSSpectrum]:

        """
            Retrieve MS spectrum for a PSM from Parquet-based identification data.

            Notes
            -----
            It expects a spectrum reference string stored in Parquet (e.g. scan=1234 or spectrum=1234).
            """

        # -------- 1. extract spectrum reference --------
        if isinstance(psm, dict):
            spectrum_reference = psm.get("spectrum_reference") or psm.get("spectrum_id")
        else:
            spectrum_reference = getattr(psm, "spectrum_id", None) or getattr(psm, "spectrum_reference", None)

        if spectrum_reference is None:
            logger.warning(f"Missing spectrum reference for PSM: {psm}")
            return None

        # -------- 2. parse scan number --------
        matches = re.findall(r"(?:spectrum|scan)=(\d+)", str(spectrum_reference))
        if not matches:
            logger.warning(f"Invalid spectrum reference format: {spectrum_reference}")
            return None

        scan_number = int(matches[0])

        # -------- 3. retrieve spectrum --------
        try:
            index = lookup.findByScanNumber(scan_number)
            spectrum = exp.getSpectrum(index)
            return spectrum

        except Exception as e:
            logger.error(
                "Error retrieving spectrum for scan=%s, reference=%s: %s",
                scan_number,
                spectrum_reference,
                e,
            )
            return None

    @staticmethod
    def get_peaks_by_scan(
        scan_number: int, exp: oms.MSExperiment, lookup: SpectrumLookup
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get spectrum deeplc_models for a given scan number

        Parameters
        ----------
            scan_number: The scan number to look up
            exp: The MSExperiment object containing the spectra
            lookup: The SpectrumLookup object to find the spectrum

        Returns
        -------
            Tuple of (mz_array, intensity_array) if found, None if not found
        """
        try:
            index = lookup.findByScanNumber(scan_number)
            spectrum = exp.getSpectrum(index)
            return spectrum.get_peaks()
        except IndexError:
            logger.warning(f"Scan number {scan_number} not found")
            return None

    @staticmethod
    def get_ms_level(
            psm: Union[PSM, dict, pd.Series],
            spec_lookup: oms.SpectrumLookup,
            exp: oms.MSExperiment,
    ) -> int:
        """
        Retrieve MS level for a Parquet-based PSM.

        Parameters
        ----------
        psm : Union[PSM, dict, pd.Series]
            PSM object or parquet row containing spectrum reference.
        spec_lookup : oms.SpectrumLookup
            OpenMS spectrum lookup index.
        exp : oms.MSExperiment
            Loaded mzML experiment.

        Returns
        -------
        int
            MS level, or -1 if unavailable.
        """

        spectrum = OpenMSHelper.get_spectrum_for_psm(
            psm,
            exp,
            spec_lookup
        )

        if spectrum is None:
            return -1

        try:
            return int(spectrum.getMSLevel())
        except Exception as e:
            logger.warning(f"Failed to retrieve MS level: {e}")
            return -1

    @staticmethod
    def get_psm_hash_unique_id(row: dict) -> str:
        """
        Generate unique PSM key from parquet row.
        """
        return f"{row.get('spectrum_ref')}_{row.get('sequence')}_{row.get('retention_time')}_{row.get('charge')}_{row.get('rank')}"

    @staticmethod
    def get_str_metavalue_round(metavalue: float):
        """
        Get a string representation of a metadata value, rounded to 4 decimal places.

        Parameters
        ----------
        metavalue : float
            The metadata value to be converted to a string.

        Returns
        -------
        str
            A string representation of the metadata value, rounded to 4 decimal places.
        """
        if np.isnan(metavalue) or np.isinf(metavalue):
            return "0.0"
        return "{:.4f}".format(metavalue)

    @staticmethod
    def get_canonical_feature(feature: str) -> Union[str, None]:
        """
        Retrieve the canonical feature name for a given feature.

        This method searches through predefined feature mappings to find
        the canonical name corresponding to the provided feature string.
        It first checks the DEEPLC_FEATURES mapping, followed by the
        MS2PIP_FEATURES mapping.

        Parameters
        ----------
        feature : str
            The feature name to be converted to its canonical form.

        Returns
        -------
        str
            The canonical feature name if found, otherwise None.
        """
        if feature is None:
            return None

        canonical_feature = next((k for k, v in DEEPLC_FEATURES.items() if v == feature), None)
        if canonical_feature is not None:
            return canonical_feature

        canonical_feature = next((k for k, v in MS2PIP_FEATURES.items() if v == feature), None)
        return canonical_feature

    @staticmethod
    def validate_features(original_features: List[str]) -> List[str]:
        """
        This function make sures that the given features are supported by the tool
        in DEEPLC_FEATURES and MS2PIP_FEATURES.

        Parameters
        ----------

        original_features: List[str]
            The list of features to validate.

        Returns
        -------

        List[str]
            The list of validated features.
        """

        validated_features = []
        for feature in original_features:
            feature = feature.strip()  # Remove leading and trailing whitespaces
            if feature in DEEPLC_FEATURES.values() or feature in MS2PIP_FEATURES.values():
                validated_features.append(feature)
            else:
                logger.warning(f"Feature {feature} not supported by quantms rescoring")
        return validated_features

    @staticmethod
    def get_pyopenms_version():
        """
        Get the version of the pyopenms library.

        Returns
        -------
        str
            The version of the pyopenms library.
        """
        oms_version = oms.VersionInfo.getVersionStruct()
        return (
            f"{oms_version.version_major}.{oms_version.version_minor}.{oms_version.version_patch}"
        )

    @staticmethod
    def get_pyopenms_dissociation_matrix() -> Union[List[dict], None]:
        """
        Retrieve the dissociation methods matrix based on the pyopenms version.

        This method checks the current version of pyopenms and returns the
        appropriate dissociation methods matrix. If the version is not supported,
        a warning is logged and None is returned.

        Returns
        -------
        Union[List[dict], None]
            A list of dissociation methods if the version is supported, otherwise None.
        """
        oms_version = version.parse(OpenMSHelper.get_pyopenms_version())
        dissociation_methods = []
        if oms_version < version.parse("3.2.0"):
            dissociation_methods = OPENMS_DISSOCIATION_METHODS_PATCH_3_1_0
        if oms_version >= version.parse("3.2.0"):
            dissociation_methods = OPENMS_DISSOCIATION_METHODS_PATCH_3_3_0
        if not dissociation_methods:
            logger.warning("OpenMS version not supported, can't find the dissociation method.")
            return None
        return dissociation_methods

    @staticmethod
    def get_dissociation_method(method_index: int, matrix: List[dict] = None) -> Union[str, None]:
        """
        Retrieve the dissociation method name based on the method index and pyopenms version.

        This method determines the appropriate dissociation method list to use
        based on the current pyopenms version and retrieves the method name
        corresponding to the provided index.

        Parameters
        ----------
        method_index : int
            The index of the dissociation method to retrieve.
        matrix : List[dict], optional
            The dissociation methods matrix to use. If not provided, the default
            matrix for the current pyopenms version is used.

        Returns
        -------
        Union[str, None]
            The name of the dissociation method if found, otherwise None.
        """
        if matrix is None:
            dissociation_methods = OpenMSHelper.get_pyopenms_dissociation_matrix()
        else:
            dissociation_methods = matrix
        if dissociation_methods is None:
            return None
        if method_index < 0 or method_index >= len(dissociation_methods):
            logger.warning("Invalid dissociation method index.")
            return None
        return list(dissociation_methods[method_index].keys())[0]

    @classmethod
    def check_unix_compatibility(cls, mzml_path: Union[str, Path]):
        """
        Check if an mzML file has Unix-style line endings.

        This method verifies whether the specified mzML file uses Unix-style
        LF line endings. If Windows-style CRLF line endings are detected,
        an MzMLNotUnixException is raised.

        This is necessary because ms2rescore-rs is using a rust dependency to read mzML files, that if the file
        fails with Windows-style CRLF line endings, do not give an error message, just fail silently.

        Parameters
        ----------
        mzml_path : Union[str, Path]
            The path to the mzML file to be checked.

        Raises
        ------
        MzMLNotUnixException
            If the file contains Windows-style CRLF line endings.
        """

        if isinstance(mzml_path, Path):
            mzml_path = str(mzml_path)

        with open(mzml_path, "rb") as f:
            content = f.read()
            if b"\r\n" in content:
                raise MzMLNotUnixException(
                    f"File {mzml_path} has Windows-style CRLF line endings. Please convert to LF (Unix-style) using `dos2unix` or similar."
                )
            else:
                logger.info(f"File {mzml_path} has the correct Unix-style LF line endings.")

    @staticmethod
    def get_ms_tolerance(
            search_params: dict,
    ) -> Tuple[float, Optional[str]]:
        """
        Get fragment mass tolerance from Parquet search parameters.

        Parameters
        ----------
        search_params : dict
            Dictionary loaded from search_params.parquet.

        Returns
        -------
        Tuple[float, Optional[str]]
            Mass tolerance and unit ("ppm" or "Da").
        """

        if not search_params:
            return 0.0, None

        tolerance = search_params.get("fragment_mass_tolerance", 0.0)

        # OpenMS sometimes encodes ppm flag explicitly
        if search_params.get("fragment_mass_tolerance_ppm", False):
            return float(tolerance), "ppm"

        # fallback
        unit = search_params.get("fragment_mass_tolerance_unit", "Da")
        return float(tolerance), unit

    @staticmethod
    def generate_theoretical_spectrum(peptide_sequence: str, charge: int):
        """
        Generate a theoretical spectrum for a given peptide sequence.

        Parameters:
        - peptide (str): Peptide sequence (e.g., "PEPTIDE").
        - charge (int): Charge state of the fragments.

        Returns:
        - theoretical_mzs (list): List of theoretical fragment m/z values.
        - ion_labels (list): Corresponding fragment ion labels.
        """

        tsg = TheoreticalSpectrumGenerator()
        spec = oms.MSSpectrum()
        peptide = oms.AASequence.fromString(peptide_sequence)

        # Generate b- and y-ions (can be extended for other ion types)
        tsg.getSpectrum(spec, peptide, 1, charge)

        theoretical_mzs = [peak.getMZ() for peak in spec]
        return theoretical_mzs

    @staticmethod
    def get_predicted_ms_tolerance(
        exp: oms.MSExperiment, ppm_tolerance: float
    ) -> Tuple[float, str]:
        """
        Calculate the predicted mass tolerance in Daltons for an MS experiment.

        This method computes the maximum fragment mass from the spectra in the
        given MSExperiment and calculates the mass tolerance in Daltons based
        on the provided parts-per-million (ppm) tolerance.

        Parameters
        ----------
        exp : oms.MSExperiment
            The MSExperiment object containing the spectra.
        ppm_tolerance : float
            The mass tolerance in parts-per-million.

        Returns
        -------
        Tuple[float, str]
            A tuple containing the calculated mass tolerance in Daltons and
            the unit "Da".
        """
        max_frag_mass = 0
        for spec in exp:
            if spec.getMSLevel() == 2:
                spec.updateRanges()
                if spec.getMaxMZ() > max_frag_mass:
                    max_frag_mass = spec.getMaxMZ()

        tol_da = max_frag_mass * ppm_tolerance / 1e6
        tol_da = round(tol_da, 4)
        return tol_da, "Da"

    @staticmethod
    def get_mslevel_spectra(
        file_name: Union[str, Path],
        ms_level: int,
        use_cache: bool = True
    ) -> List[oms.MSSpectrum]:
        """
        Get spectra of a specific MS level from an mzML file.

        Parameters
        ----------
        file_name : Union[str, Path]
            Path to the mzML file.
        ms_level : int
            MS level to filter (e.g., 2 for MS2).
        use_cache : bool, optional
            If True, use the global spectrum cache. Default is True.

        Returns
        -------
        List[oms.MSSpectrum]
            List of spectra at the specified MS level.
        """
        if use_cache:
            exp, _ = get_cached_spectrum_data(file_name)
        else:
            exp = OpenMSHelper.get_spectrum_lookup_indexer(str(file_name))[0]

        spectra = []
        for spec in exp:
            if spec.getMSLevel() == ms_level:
                spectra.append(spec)
        return spectra

    @staticmethod
    def iter_mslevel_spectra(
        file_name: Union[str, Path],
        ms_level: int,
        use_cache: bool = True
    ) -> Generator[oms.MSSpectrum, None, None]:
        """
        Iterate over spectra of a specific MS level (memory-efficient generator).

        This is more memory-efficient than get_mslevel_spectra() when you don't
        need all spectra at once.

        Parameters
        ----------
        file_name : Union[str, Path]
            Path to the mzML file.
        ms_level : int
            MS level to filter (e.g., 2 for MS2).
        use_cache : bool, optional
            If True, use the global spectrum cache. Default is True.

        Yields
        ------
        oms.MSSpectrum
            Spectra at the specified MS level.
        """
        if use_cache:
            exp, _ = get_cached_spectrum_data(file_name)
        else:
            exp = OpenMSHelper.get_spectrum_lookup_indexer(str(file_name))[0]

        for spec in exp:
            if spec.getMSLevel() == ms_level:
                yield spec

    @staticmethod
    def get_instrument(exp: oms.MSExperiment):
        instrument = exp.getInstrument().getName()
        return instrument

    @staticmethod
    def get_nce_psm(
            psm: Union[PSM, dict],
            exp: oms.MSExperiment,
            lookup: oms.SpectrumLookup
    ) -> Optional[float]:
        """
        Extract collision energy (NCE).
        """

        # -------- spectrum metadata --------
        spectrum = OpenMSHelper.get_spectrum_for_psm(psm, exp, lookup)
        if spectrum is None:
            return None

        meta = spectrum.getMetaValue("filter string")

        if not meta:
            return None

        # -------- robust regex extraction --------
        match = re.search(r"@([A-Za-z]+)(\d+(?:\.\d+)?)", str(meta))
        if not match:
            return None

        return float(match.group(2))