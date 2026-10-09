import os
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import ms2pip
import ms2pip.exceptions as exceptions
import numpy as np
from ms2pip._utils.psm_input import read_psms
from ms2pip.exceptions import NoMatchingSpectraFound
from ms2pip.result import ProcessingResult
from ms2pip.spectrum import ObservedSpectrum
from ms2rescore.feature_generators import MS2PIPFeatureGenerator
from ms2rescore.feature_generators.base import FeatureGeneratorException
from ms2rescore.utils import infer_spectrum_path
from ms2rescore_rs import MS2Spectrum, Precursor
from psm_utils import Peptidoform, PSMList

from quantmsrescore.constants import PRIMARY_SCORE_IMPUTED, SUPPORTED_MODELS_MS2PIP
from quantmsrescore.logging_config import get_logger
from quantmsrescore.openms import (
    OpenMSHelper,
    calculate_correlations,
    get_compiled_regex,
    organize_psms_by_spectrum_id,
)
from quantmsrescore.utils import worker_pool

logger = get_logger(__name__)


@contextmanager
def _rayon_threads(threads: int):
    """Cap MS2PIP's Rust threads for one call unless RAYON_NUM_THREADS is already set.

    MS2PIP annotates spectra in Rust before its own thread setting, and Rayon
    sizes its thread pool at the first parallel call, so the cap must come first.
    The variable is removed afterwards so later stages and workers do not inherit it.
    """
    if "RAYON_NUM_THREADS" in os.environ:
        yield
        return
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    try:
        yield
    finally:
        os.environ.pop("RAYON_NUM_THREADS", None)


class MS2PIPAnnotator(MS2PIPFeatureGenerator):

    def __init__(
        self,
        *args,
        model: str = "HCD",
        ms2_tolerance: float = 0.02,
        ms2_tolerance_unit: str = "Da",
        spectrum_path: Optional[str] = None,
        spectrum_id_pattern: str = "(.*)",
        model_dir: Optional[str] = None,
        processes: int = 1,
        calibration_set_size: Optional[float] = 0.20,
        valid_correlations_size: Optional[float] = 0.70,
        correlation_threshold: Optional[float] = 0.6,
        higher_score_better: bool = True,
        force_model: bool = False,
        **kwargs,
    ):
        if ms2_tolerance_unit not in {"Da", "ppm"}:
            raise ValueError("MS2 tolerance unit must be 'Da' or 'ppm'.")
        if processes < 1:
            raise ValueError("processes must be a positive integer.")
        self.ms2_tolerance_unit = ms2_tolerance_unit
        super().__init__(
            *args,
            model=model,
            ms2_tolerance=ms2_tolerance,
            spectrum_path=spectrum_path,
            spectrum_id_pattern=spectrum_id_pattern,
            model_dir=model_dir,
            processes=processes,
            **kwargs,
        )
        self._calibration_set_size: float = calibration_set_size
        self._valid_correlations_size: float = valid_correlations_size
        self._correlation_threshold: float = correlation_threshold
        self._higher_score_better: bool = higher_score_better
        self._force_model: bool = force_model

    def validate_features(self, psm_list: PSMList, model: str = None) -> bool:
        """
        This method is used to validate a model for a given PSM list.
        It checks if the model is valid for the given PSM list and returns a boolean value.

        Parameters
        ----------
        psm_list : PSMList
            The PSM list to validate the model for.
        model : str, optional
            The model to validate. If not provided, the default model is used.

        """
        logger.info("Adding MS²PIP-derived features to PSMs.")
        psm_dict = psm_list.get_psm_dict()
        current_run = 1
        valid_correlation = None
        if model is None:
            model = self.model
        for runs in psm_dict.values():
            for run, psms in runs.items():
                psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                spectrum_filename = infer_spectrum_path(self.spectrum_path, run)
                logger.debug(f"Using spectrum file `{spectrum_filename}`")
                try:
                    ms2pip_results = self.custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=model,
                        ms2_tolerance=self.ms2_tolerance,
                        ms2_tolerance_unit=self.ms2_tolerance_unit,
                        compute_correlations=True,
                        model_dir=self.model_dir,
                        processes=self.processes,
                    )
                except NoMatchingSpectraFound as e:
                    raise FeatureGeneratorException(
                        f"Could not find any matching spectra for PSMs from run `{run}`. "
                        "Please check that the `spectrum_id_pattern` and `psm_id_pattern` "
                        "options are configured correctly. See "
                        "https://ms2rescore.readthedocs.io/en/latest/userguide/configuration/#mapping-psms-to-spectra"
                        " for more information."
                    ) from e
                valid_correlation = self._validate_scores(
                    ms2pip_results=ms2pip_results,
                    calibration_set_size=self._calibration_set_size,
                    valid_correlations_size=self._valid_correlations_size,
                    correlation_threshold=self._correlation_threshold,
                    higher_score_better=self._higher_score_better,
                )
                current_run += 1
        return valid_correlation

    def add_features(self, psm_list: PSMList) -> None:
        """
        Add MS²PIP-derived features to PSMs.

        Parameters
        ----------
        psm_list
        PSMs to add features to.
        """
        logger.info("Adding MS²PIP-derived features to PSMs.")
        psm_dict = psm_list.get_psm_dict()
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())

        for runs in psm_dict.values():
            for run, psms in runs.items():
                logger.info(
                    f"Running MS²PIP {self.model} for PSMs from run ({current_run}/{total_runs}) `{run}`..."
                )
                psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                spectrum_filename = infer_spectrum_path(self.spectrum_path, run)
                logger.debug(f"Using spectrum file `{spectrum_filename}`")
                try:
                    ms2pip_results = self.custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=self.model,
                        ms2_tolerance=self.ms2_tolerance,
                        ms2_tolerance_unit=self.ms2_tolerance_unit,
                        compute_correlations=True,
                        model_dir=self.model_dir,
                        processes=self.processes,
                    )
                except NoMatchingSpectraFound as e:
                    raise FeatureGeneratorException(
                        f"Could not find any matching spectra for PSMs from run `{run}`. "
                        "Please check that the `spectrum_id_pattern` and `psm_id_pattern` "
                        "options are configured correctly. See "
                        "https://ms2rescore.readthedocs.io/en/latest/userguide/configuration/#mapping-psms-to-spectra"
                        " for more information."
                    ) from e
                self._calculate_features(psm_list_run, ms2pip_results)
                current_run += 1

    def _validate_scores(
        self,
        ms2pip_results,
        calibration_set_size,
        valid_correlations_size,
        correlation_threshold,
        higher_score_better,
    ) -> bool:
        """
        Validate MS²PIP results based on score and correlation criteria.

        This method checks if the MS²PIP results meet the specified correlation
        threshold and score criteria. It first filters out decoy PSMs, sorts the
        results based on the PSM score, and selects a calibration set. The method
        then verifies if at least 80% of the calibration set has a correlation
        above the given threshold.

        Parameters
        ----------
        ms2pip_results : list
            List of MS²PIP results to validate.
        calibration_set_size : float
            Fraction of the results to use for calibration.
        valid_correlations_size: float
            Fraction of the valid PSM.
        correlation_threshold : float
            Minimum correlation value required for a result to be considered valid.
        higher_score_better : bool
            Indicates if a higher PSM score is considered better.

        Returns
        -------
        bool
            True if the results are valid based on the criteria, False otherwise.
        """
        if not ms2pip_results:
            return False

        ms2pip_results_copy = (
            ms2pip_results.copy()
        )  # Copy ms2pip results to avoid modifying the original list

        # The calibration fraction is relative to observed primary scores.
        # Other engines' candidates receive an imputed primary score during
        # merging; counting them would enlarge this engine's top-scoring set.
        # They remain in the full PSM list for feature generation and rescoring.
        ms2pip_results_copy = [
            result
            for result in ms2pip_results_copy
            if not result.psm.is_decoy and result.psm.rank == 1
            and (result.psm.metadata or {}).get(PRIMARY_SCORE_IMPUTED) != "true"
        ]
        # Sort ms2pip results by PSM score and lower score is better
        ms2pip_results_copy.sort(key=lambda x: x.psm.score, reverse=higher_score_better)

        # Get a calibration set, the % of psms to be used for calibrarion is defined by calibration_set_size
        calibration_set = ms2pip_results_copy[
            : int(len(ms2pip_results_copy) * calibration_set_size)
        ]

        if not calibration_set:
            logger.info("No target PSMs available in the model calibration set.")
            return False

        # Invalid predictions remain in the denominator, rather than making a
        # model appear better by silently dropping failed PSMs.
        valid_correlation = [
            psm
            for psm in calibration_set
            if psm.correlation is not None
            and np.isfinite(psm.correlation)
            and psm.correlation >= correlation_threshold
        ]

        logger.info(
            f"The percentage of PSMs in the top {calibration_set_size * 100}% with a correlation greater than {correlation_threshold} is: "
            f"{(len(valid_correlation) / len(calibration_set)) * 100:.2f}%"
        )

        if len(valid_correlation) < len(calibration_set) * valid_correlations_size:
            return False

        return True

    def _find_best_ms2pip_model(
        self, batch_psms: PSMList, known_fragmentation: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Find the best MS²PIP model for a batch of PSMs.

        This method finds the best MS²PIP model for a batch of PSMs by
        comparing the correlation of the PSMs with the different models.

        Parameters
        ----------
        batch_psms : list
            List of PSMs to find the best model for.

        Returns
        -------
        Tuple
            Tuple containing the best model and the correlation value.
        """
        best_model = None
        best_correlation = 0

        filtered_models = SUPPORTED_MODELS_MS2PIP

        if known_fragmentation:
            filtered_models = {
                known_fragmentation: SUPPORTED_MODELS_MS2PIP.get(known_fragmentation)
            }

        for fragment_types in filtered_models:
            for model in filtered_models[fragment_types]:
                logger.info(f"Running MS²PIP for model `{model}`...")
                ms2pip_results = self.custom_correlate(
                    psms=batch_psms,
                    spectrum_file=self.spectrum_path,
                    spectrum_id_pattern=self.spectrum_id_pattern,
                    model=model,
                    ms2_tolerance=self.ms2_tolerance,
                    ms2_tolerance_unit=self.ms2_tolerance_unit,
                    compute_correlations=True,
                    model_dir=self.model_dir,
                    processes=self.processes,
                )
                correlation = self._calculate_correlation(ms2pip_results)
                if correlation > best_correlation and correlation >= 0.4:
                    best_model = model
                    best_correlation = correlation

        return best_model, best_correlation

    @staticmethod
    def _calculate_correlation(ms2pip_results: List[ProcessingResult]) -> float:
        """
        Calculate the average correlation from MS²PIP results.

        This method computes the average correlation score from a list of
        MS²PIP results, where each result contains a correlation attribute.

        Parameters
        ----------
        ms2pip_results : list
            List of MS²PIP results, each containing a correlation score.

        Returns
        -------
        float
            The average correlation score of the provided MS²PIP results.
        """
        if not ms2pip_results:
            return 0.0
        total_correlation = sum(
            [
                psm.correlation
                for psm in ms2pip_results
                if psm.correlation is not None and np.isfinite(psm.correlation)
            ]
        )
        return total_correlation / len(ms2pip_results)

    def _calculate_features(
        self, psm_list: PSMList, ms2pip_results: List[ProcessingResult]
    ) -> None:
        """Keep upstream features, without forking an initialized Rust thread pool."""
        failed = 0
        with worker_pool(self.processes) as pool:
            features = pool.imap(
                self._calculate_features_single, ms2pip_results, chunksize=1000
            )
            for result, values in zip(ms2pip_results, features):
                if values:
                    psm = psm_list[result.psm_index]
                    if psm.rescoring_features is None:
                        psm.rescoring_features = {}
                    psm.rescoring_features.update(values)
                else:
                    failed += 1
        if failed:
            logger.warning(f"Failed to calculate features for {failed} PSMs")

    def custom_correlate(
        self,
        psms: Union[PSMList, str, Path],
        spectrum_file: Union[str, Path],
        psm_filetype: Optional[str] = None,
        spectrum_id_pattern: Optional[str] = None,
        compute_correlations: bool = False,
        add_retention_time: bool = False,
        add_ion_mobility: bool = False,
        model: str = "HCD",
        model_dir: Optional[Union[str, Path]] = None,
        ms2_tolerance: float = 0.02,
        ms2_tolerance_unit: Optional[str] = None,
        processes: Optional[int] = None,
    ) -> List[ProcessingResult]:
        """Read spectra with OpenMS and use MS2PIP's public, unit-aware API."""
        psm_list = read_psms(psms, filetype=psm_filetype)
        if not psm_list:
            raise NoMatchingSpectraFound("No PSMs to match to spectra.")
        if len(psm_list.collections) != 1 or len(psm_list.runs) != 1:
            raise exceptions.InvalidInputError("PSMs should be for a single run and collection.")
        spectrum_id_pattern = spectrum_id_pattern or "(.*)"
        spectrum_id_regex = get_compiled_regex(spectrum_id_pattern)
        by_spectrum = organize_psms_by_spectrum_id(list(enumerate(psm_list)))
        matched = {}
        for spectrum in read_spectrum_file(str(spectrum_file)):
            match = spectrum_id_regex.search(spectrum.identifier)
            try:
                spectrum_id = match[1]
            except (TypeError, IndexError) as exc:
                raise exceptions.TitlePatternError(
                    f"Spectrum title pattern `{spectrum_id_pattern}` could not be matched to "
                    f"spectrum ID `{spectrum.identifier}`. A capturing group is required."
                ) from exc
            if spectrum_id not in by_spectrum:
                continue
            raw_spectrum = MS2Spectrum(
                identifier=spectrum.identifier,
                mz=spectrum.mz,
                intensity=spectrum.intensity,
                precursor=Precursor(
                    mz=spectrum.precursor_mz,
                    charge=int(spectrum.precursor_charge),
                    rt=spectrum.retention_time,
                ),
            )
            for index, psm in by_spectrum[spectrum_id]:
                copied = psm.model_copy(update={"spectrum": raw_spectrum})
                if not copied.peptidoform.precursor_charge:
                    copied.peptidoform = Peptidoform(
                        f"{copied.peptidoform.modified_sequence}/{int(spectrum.precursor_charge)}"
                    )
                matched[index] = copied
        if not matched:
            raise NoMatchingSpectraFound("No spectra matching spectrum IDs from PSM list.")

        indices = sorted(matched)
        workers = processes if processes is not None else self.processes
        with _rayon_threads(workers):
            results = ms2pip.correlate(
                psms=PSMList(psm_list=[matched[index] for index in indices]),
                spectrum_file=None,
                model=model,
                model_dir=model_dir,
                ms2_tolerance=ms2_tolerance,
                ms2_tolerance_mode=ms2_tolerance_unit or self.ms2_tolerance_unit,
                processes=workers,
                compute_correlations=False,
                add_retention_time=add_retention_time,
                add_ion_mobility=add_ion_mobility,
            )
        for result in results:
            result.psm_index = indices[result.psm_index]
            # Do not retain Rust spectrum objects in feature worker arguments.
            result.psm = psm_list[result.psm_index]
        results.sort(key=lambda result: result.psm_index)
        if compute_correlations:
            calculate_correlations(results)
            correlations = [
                r.correlation
                for r in results
                if r.correlation is not None and np.isfinite(r.correlation)
            ]
            if correlations:
                logger.info(f"Median correlation: {np.median(correlations)}, model {model}")
        return results


def read_spectrum_file(
    spec_file: str, use_cache: bool = True
) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from a supported file format; inferring the type from the filename extension.

    This function uses a global cache to prevent loading the same mzML file
    multiple times when both MS2PIP and AlphaPeptDeep process the same file.

    Parameters
    ----------
    spec_file : str
        Path to MGF or mzML file.
    use_cache : bool, optional
        If True, use the global spectrum cache. Default is True.
        This prevents duplicate file loading when multiple feature generators
        process the same spectrum file.

    Yields
    ------
    ObservedSpectrum

    Raises
    ------
    UnsupportedSpectrumFiletypeError
        If the file extension is not supported.
    """
    try:
        # Use iterator version for memory efficiency
        spectra = OpenMSHelper.iter_mslevel_spectra(
            file_name=str(spec_file), ms_level=2, use_cache=use_cache
        )
    except ValueError:
        raise exceptions.UnsupportedSpectrumFiletypeError(Path(spec_file).suffixes)

    for spectrum in spectra:
        mz, intensities = spectrum.get_peaks()
        precursors = spectrum.getPrecursors()
        obs_spectrum = None
        if len(precursors) > 0:
            precursor = precursors[0]
            charge_state = precursor.getCharge()
            exp_mz = precursor.getMZ()
            rt = spectrum.getRT()
            spec_id = spectrum.getNativeID()

            obs_spectrum = ObservedSpectrum(
                mz=np.array(mz, dtype=np.float32),
                intensity=np.array(intensities, dtype=np.float32),
                identifier=str(spec_id),
                precursor_mz=float(exp_mz),
                precursor_charge=float(charge_state),
                retention_time=float(rt),
            )
        if (
            obs_spectrum is None
            or obs_spectrum.identifier == ""
            or obs_spectrum.mz.shape[0] == 0
            or obs_spectrum.intensity.shape[0] == 0
        ):
            continue
        yield obs_spectrum
