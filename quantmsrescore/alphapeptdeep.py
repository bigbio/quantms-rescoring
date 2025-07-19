import os.path

import pandas as pd
from peptdeep.pretrained_models import ModelManager
from peptdeep.mass_spec.match import match_centroid_mz
from ms2rescore.feature_generators.base import FeatureGeneratorBase, FeatureGeneratorException
from typing import Optional, Tuple, List, Union, Generator, Dict

from psm_utils import PSMList, PSM
from quantmsrescore.logging_config import get_logger
from quantmsrescore.openms import OpenMSHelper
from ms2rescore.utils import infer_spectrum_path
import ms2pip.exceptions as exceptions
import numpy as np

from ms2pip._utils.psm_input import read_psms
from ms2pip.exceptions import NoMatchingSpectraFound
from ms2pip.result import ProcessingResult
from ms2pip.spectrum import ObservedSpectrum
import multiprocessing
import re
from collections import defaultdict
from itertools import chain
from pathlib import Path
from rich.progress import track
import warnings

# Get logger for this module
logger = get_logger(__name__)


class AlphaPeptDeepFeatureGenerator(FeatureGeneratorBase):
    """Generate AlphaPeptDeep-based features."""

    def __init__(
            self,
            *args,
            model: str = "generic",
            ms2_tolerance: float = 0.02,
            spectrum_path: Optional[str] = None,
            spectrum_id_pattern: str = "(.*)",
            model_dir: Optional[str] = None,
            mask_modloss: bool = True,
            processes: 1,
            **kwargs,
    ) -> None:
        """
        Generate AlphaPeptDeep-based features.

        Parameters
        ----------
        model
            AlphaPeptDeep prediction model to use. Defaults to :py:const:`generic`.
        ms2_tolerance
            MS2 mass tolerance in Da. Defaults to :py:const:`0.02`.
        spectrum_path
            Path to spectrum file or directory with spectrum files. If None, inferred from ``run``
            field in PSMs. Defaults to :py:const:`None`.
        spectrum_id_pattern : str, optional
            Regular expression pattern to extract spectrum ID from spectrum file. Defaults to
            :py:const:`.*`.
        model_dir
            Directory containing AlphaPeptDeep models. Defaults to :py:const:`None` (use AlphaPeptDeep default).
        processes : int, optional
            Number of processes to use. Defaults to 1.

        Attributes
        ----------
        feature_names: list[str]
            Names of the features that will be added to the PSMs.

        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.ms2_tolerance = ms2_tolerance
        self.spectrum_path = spectrum_path
        self.spectrum_id_pattern = spectrum_id_pattern
        self.model_dir = model_dir
        self.processes = processes
        self.mask_modloss = mask_modloss

    @property
    def feature_names(self):
        return [
            "spec_pearson_norm",
            "ionb_pearson_norm",
            "iony_pearson_norm",
            "spec_mse_norm",
            "ionb_mse_norm",
            "iony_mse_norm",
            "min_abs_diff_norm",
            "max_abs_diff_norm",
            "abs_diff_Q1_norm",
            "abs_diff_Q2_norm",
            "abs_diff_Q3_norm",
            "mean_abs_diff_norm",
            "std_abs_diff_norm",
            "ionb_min_abs_diff_norm",
            "ionb_max_abs_diff_norm",
            "ionb_abs_diff_Q1_norm",
            "ionb_abs_diff_Q2_norm",
            "ionb_abs_diff_Q3_norm",
            "ionb_mean_abs_diff_norm",
            "ionb_std_abs_diff_norm",
            "iony_min_abs_diff_norm",
            "iony_max_abs_diff_norm",
            "iony_abs_diff_Q1_norm",
            "iony_abs_diff_Q2_norm",
            "iony_abs_diff_Q3_norm",
            "iony_mean_abs_diff_norm",
            "iony_std_abs_diff_norm",
            "dotprod_norm",
            "dotprod_ionb_norm",
            "dotprod_iony_norm",
            "cos_norm",
            "cos_ionb_norm",
            "cos_iony_norm",
            "spec_pearson",
            "ionb_pearson",
            "iony_pearson",
            "spec_spearman",
            "ionb_spearman",
            "iony_spearman",
            "spec_mse",
            "ionb_mse",
            "iony_mse",
            "min_abs_diff_iontype",
            "max_abs_diff_iontype",
            "min_abs_diff",
            "max_abs_diff",
            "abs_diff_Q1",
            "abs_diff_Q2",
            "abs_diff_Q3",
            "mean_abs_diff",
            "std_abs_diff",
            "ionb_min_abs_diff",
            "ionb_max_abs_diff",
            "ionb_abs_diff_Q1",
            "ionb_abs_diff_Q2",
            "ionb_abs_diff_Q3",
            "ionb_mean_abs_diff",
            "ionb_std_abs_diff",
            "iony_min_abs_diff",
            "iony_max_abs_diff",
            "iony_abs_diff_Q1",
            "iony_abs_diff_Q2",
            "iony_abs_diff_Q3",
            "iony_mean_abs_diff",
            "iony_std_abs_diff",
            "dotprod",
            "dotprod_ionb",
            "dotprod_iony",
            "cos",
            "cos_ionb",
            "cos_iony",
        ]

    def add_features(self, psm_list: PSMList, psms_df: pd.DataFrame) -> None:
        """
        Add AlphaPeptDeep-derived features to PSMs.

        Parameters
        ----------
        psm_list
            PSMs to add features to.

        """
        logger.info("Adding AlphaPeptDeep-derived features to PSMs.")
        psm_dict = psm_list.get_psm_dict()
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())

        for runs in psm_dict.values():
            for run, psms in runs.items():
                logger.info(
                    f"Running AlphaPeptDeep for PSMs from run ({current_run}/{total_runs}) `{run}`..."
                )
                psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                spectrum_filename = infer_spectrum_path(self.spectrum_path, run)
                logger.debug(f"Using spectrum file `{spectrum_filename}`")
                try:
                    alphapeptdeep_results = custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        psms_df=psms_df,
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=self.model,
                        ms2_tolerance=self.ms2_tolerance,
                        compute_correlations=False,
                        model_dir=self.model_dir,
                        processes=self.processes,
                        mask_modloss=self.mask_modloss
                    )
                except NoMatchingSpectraFound as e:
                    raise FeatureGeneratorException(
                        f"Could not find any matching spectra for PSMs from run `{run}`. "
                        "Please check that the `spectrum_id_pattern` and `psm_id_pattern` "
                        "options are configured correctly. See "
                        "https://ms2rescore.readthedocs.io/en/latest/userguide/configuration/#mapping-psms-to-spectra"
                        " for more information."
                    ) from e
                self._calculate_features(psm_list_run, alphapeptdeep_results)
                current_run += 1

    def _calculate_features(
            self, psm_list: PSMList, alphapeptdeep_results: List[ProcessingResult]
    ) -> None:
        """Calculate features from all AlphaPeptDeep results and add to PSMs."""
        logger.debug("Calculating features from predicted spectra")
        with multiprocessing.Pool(int(self.processes)) as pool:
            # Use imap, so we can use a progress bar
            counts_failed = 0
            for result, features in zip(
                    alphapeptdeep_results,
                    track(
                        pool.imap(self._calculate_features_single, alphapeptdeep_results, chunksize=1000),
                        total=len(alphapeptdeep_results),
                        description="Calculating features...",
                        transient=True,
                    ),
            ):
                if features:
                    # Cannot use result.psm directly, as it is a copy from AlphaPeptDeep multiprocessing
                    try:
                        psm_list[result.psm_index]["rescoring_features"].update(features)
                    except (AttributeError, TypeError):
                        psm_list[result.psm_index]["rescoring_features"] = features
                else:
                    counts_failed += 1

        if counts_failed > 0:
            logger.warning(f"Failed to calculate features for {counts_failed} PSMs")

    def _calculate_features_single(self, processing_result: ProcessingResult) -> Union[dict, None]:
        """Calculate AlphaPeptDeep-based features for single PSM."""
        if (
                processing_result.observed_intensity is None
                or processing_result.predicted_intensity is None
        ):
            return None

        # Suppress RuntimeWarnings about invalid values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Convert intensities to arrays, The predictions of the AlphaPeptDeep model will be truncated at 1e-4.
            target_b_unlog = processing_result.predicted_intensity["b"]
            target_y_unlog = processing_result.predicted_intensity["y"]
            target_all_unlog = np.concatenate([target_b_unlog, target_y_unlog])
            prediction_b_unlog = processing_result.observed_intensity["b"]
            prediction_y_unlog = processing_result.observed_intensity["y"]
            prediction_all_unlog = np.concatenate([prediction_b_unlog, prediction_y_unlog])

            # Prepare 'logged' intensity arrays
            target_b = np.log2(target_b_unlog.clip(0.0001))
            target_y = np.log2(target_y_unlog.clip(0.0001))
            target_all = np.concatenate([target_b, target_y])
            prediction_b = np.log2(prediction_b_unlog.clip(0.0001))
            prediction_y = np.log2(prediction_y_unlog.clip(0.0001))
            prediction_all = np.concatenate([prediction_b, prediction_y])

            # Calculate absolute differences
            abs_diff_b = np.abs(target_b - prediction_b)
            abs_diff_y = np.abs(target_y - prediction_y)
            abs_diff_all = np.abs(target_all - prediction_all)
            abs_diff_b_unlog = np.abs(target_b_unlog - prediction_b_unlog)
            abs_diff_y_unlog = np.abs(target_y_unlog - prediction_y_unlog)
            abs_diff_all_unlog = np.abs(target_all_unlog - prediction_all_unlog)

            # Compute features
            feature_values = [
                # Features between spectra in log space
                np.corrcoef(target_all, prediction_all)[0][1],  # Pearson all ions
                np.corrcoef(target_b, prediction_b)[0][1],  # Pearson b ions
                np.corrcoef(target_y, prediction_y)[0][1],  # Pearson y ions
                _mse(target_all, prediction_all),  # MSE all ions
                _mse(target_b, prediction_b),  # MSE b ions
                _mse(target_y, prediction_y),  # MSE y ions
                np.min(abs_diff_all),  # min_abs_diff_norm
                np.max(abs_diff_all),  # max_abs_diff_norm
                np.quantile(abs_diff_all, 0.25),  # abs_diff_Q1_norm
                np.quantile(abs_diff_all, 0.5),  # abs_diff_Q2_norm
                np.quantile(abs_diff_all, 0.75),  # abs_diff_Q3_norm
                np.mean(abs_diff_all),  # mean_abs_diff_norm
                np.std(abs_diff_all),  # std_abs_diff_norm
                np.min(abs_diff_b),  # ionb_min_abs_diff_norm
                np.max(abs_diff_b),  # ionb_max_abs_diff_norm
                np.quantile(abs_diff_b, 0.25),  # ionb_abs_diff_Q1_norm
                np.quantile(abs_diff_b, 0.5),  # ionb_abs_diff_Q2_norm
                np.quantile(abs_diff_b, 0.75),  # ionb_abs_diff_Q3_norm
                np.mean(abs_diff_b),  # ionb_mean_abs_diff_norm
                np.std(abs_diff_b),  # ionb_std_abs_diff_norm
                np.min(abs_diff_y),  # iony_min_abs_diff_norm
                np.max(abs_diff_y),  # iony_max_abs_diff_norm
                np.quantile(abs_diff_y, 0.25),  # iony_abs_diff_Q1_norm
                np.quantile(abs_diff_y, 0.5),  # iony_abs_diff_Q2_norm
                np.quantile(abs_diff_y, 0.75),  # iony_abs_diff_Q3_norm
                np.mean(abs_diff_y),  # iony_mean_abs_diff_norm
                np.std(abs_diff_y),  # iony_std_abs_diff_norm
                np.dot(target_all, prediction_all),  # Dot product all ions
                np.dot(target_b, prediction_b),  # Dot product b ions
                np.dot(target_y, prediction_y),  # Dot product y ions
                _cosine_similarity(target_all, prediction_all),  # Cos similarity all ions
                _cosine_similarity(target_b, prediction_b),  # Cos similarity b ions
                _cosine_similarity(target_y, prediction_y),  # Cos similarity y ions
                # Same features in normal space
                np.corrcoef(target_all_unlog, prediction_all_unlog)[0][1],  # Pearson all
                np.corrcoef(target_b_unlog, prediction_b_unlog)[0][1],  # Pearson b
                np.corrcoef(target_y_unlog, prediction_y_unlog)[0][1],  # Pearson y
                _spearman(target_all_unlog, prediction_all_unlog),  # Spearman all ions
                _spearman(target_b_unlog, prediction_b_unlog),  # Spearman b ions
                _spearman(target_y_unlog, prediction_y_unlog),  # Spearman y ions
                _mse(target_all_unlog, prediction_all_unlog),  # MSE all ions
                _mse(target_b_unlog, prediction_b_unlog),  # MSE b ions
                _mse(target_y_unlog, prediction_y_unlog),  # MSE y ions,
                # Ion type with min absolute difference
                0 if np.min(abs_diff_b_unlog) <= np.min(abs_diff_y_unlog) else 1,
                # Ion type with max absolute difference
                0 if np.max(abs_diff_b_unlog) >= np.max(abs_diff_y_unlog) else 1,
                np.min(abs_diff_all_unlog),  # min_abs_diff
                np.max(abs_diff_all_unlog),  # max_abs_diff
                np.quantile(abs_diff_all_unlog, 0.25),  # abs_diff_Q1
                np.quantile(abs_diff_all_unlog, 0.5),  # abs_diff_Q2
                np.quantile(abs_diff_all_unlog, 0.75),  # abs_diff_Q3
                np.mean(abs_diff_all_unlog),  # mean_abs_diff
                np.std(abs_diff_all_unlog),  # std_abs_diff
                np.min(abs_diff_b_unlog),  # ionb_min_abs_diff
                np.max(abs_diff_b_unlog),  # ionb_max_abs_diff_norm
                np.quantile(abs_diff_b_unlog, 0.25),  # ionb_abs_diff_Q1
                np.quantile(abs_diff_b_unlog, 0.5),  # ionb_abs_diff_Q2
                np.quantile(abs_diff_b_unlog, 0.75),  # ionb_abs_diff_Q3
                np.mean(abs_diff_b_unlog),  # ionb_mean_abs_diff
                np.std(abs_diff_b_unlog),  # ionb_std_abs_diff
                np.min(abs_diff_y_unlog),  # iony_min_abs_diff
                np.max(abs_diff_y_unlog),  # iony_max_abs_diff
                np.quantile(abs_diff_y_unlog, 0.25),  # iony_abs_diff_Q1
                np.quantile(abs_diff_y_unlog, 0.5),  # iony_abs_diff_Q2
                np.quantile(abs_diff_y_unlog, 0.75),  # iony_abs_diff_Q3
                np.mean(abs_diff_y_unlog),  # iony_mean_abs_diff
                np.std(abs_diff_y_unlog),  # iony_std_abs_diff
                np.dot(target_all_unlog, prediction_all_unlog),  # Dot product all ions
                np.dot(target_b_unlog, prediction_b_unlog),  # Dot product b ions
                np.dot(target_y_unlog, prediction_y_unlog),  # Dot product y ions
                _cosine_similarity(target_all_unlog, prediction_all_unlog),  # Cos similarity all
                _cosine_similarity(target_b_unlog, prediction_b_unlog),  # Cos similarity b ions
                _cosine_similarity(target_y_unlog, prediction_y_unlog),  # Cos similarity y ions
            ]

        features = dict(
            zip(
                self.feature_names,
                [0.0 if np.isnan(ft) else ft for ft in feature_values],
            )
        )

        return features


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    x = np.array(x)
    y = np.array(y)
    x_rank = pd.Series(x).rank()
    y_rank = pd.Series(y).rank()
    return np.corrcoef(x_rank, y_rank)[0][1]


def _mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error"""
    x = np.array(x)
    y = np.array(y)
    return np.mean((x - y) ** 2)


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity"""
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))


class AlphaPeptDeepAnnotator(AlphaPeptDeepFeatureGenerator):

    def __init__(
            self,
            *args,
            model: str = "generic",
            ms2_tolerance: float = 0.02,
            spectrum_path: Optional[str] = None,
            spectrum_id_pattern: str = "(.*)",
            model_dir: Optional[str] = None,
            processes: int = 1,
            calibration_set_size: Optional[float] = 0.20,
            valid_correlations_size: Optional[float] = 0.70,
            correlation_threshold: Optional[float] = 0.6,
            higher_score_better: bool = True,
            force_model: bool = False,
            mask_modloss: bool = True,
            **kwargs,
    ):
        super().__init__(
            args,
            model=model,
            ms2_tolerance=ms2_tolerance,
            spectrum_path=spectrum_path,
            spectrum_id_pattern=spectrum_id_pattern,
            model_dir=model_dir,
            processes=processes,
            kwargs=kwargs,
        )
        self._calibration_set_size: float = calibration_set_size
        self._valid_correlations_size: float = valid_correlations_size
        self._correlation_threshold: float = correlation_threshold
        self._higher_score_better: bool = higher_score_better
        self._force_model: bool = force_model
        self._mask_modloss: bool = mask_modloss

    def validate_features(self, psm_list: PSMList, psms_df: pd.DataFrame, model: str = None) -> bool:
        """
        This method is used to validate a model for a given PSM list.
        It checks if the model is valid for the given PSM list and returns a boolean value.

        Parameters
        ----------
        psm_list : PSMList
            The PSM list to validate the model for.
        model : str, optional
            The model to validate. If not provided, the default model is used.
        psms_df:
            The PSM Dataframe to validate the model for.
        """
        logger.info("Adding AlphaPeptDeep-derived features to PSMs.")
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
                    AlphaPeptDeep_results = custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        psms_df=psms_df,
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=model,
                        ms2_tolerance=self.ms2_tolerance,
                        compute_correlations=True,
                        model_dir=self.model_dir,
                        processes=self.processes,
                        mask_modloss=self._mask_modloss,
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
                    alphapeptdeep_results=AlphaPeptDeep_results,
                    calibration_set_size=self._calibration_set_size,
                    valid_correlations_size=self._valid_correlations_size,
                    correlation_threshold=self._correlation_threshold,
                    higher_score_better=self._higher_score_better,
                )
                current_run += 1
        return valid_correlation

    def add_features(self, psm_list: PSMList, psms_df: pd.DataFrame) -> None:
        """
        Add AlphaPeptDeep-derived features to PSMs.

        Parameters
        ----------
        psm_list
        PSMs to add features to.
        """
        logger.info("Adding AlphaPeptDeep-derived features to PSMs.")
        psm_dict = psm_list.get_psm_dict()
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())

        for runs in psm_dict.values():
            for run, psms in runs.items():

                logger.info(
                    f"Running AlphaPeptDeep {self.model} for PSMs from run ({current_run}/{total_runs}) `{run}`..."
                )
                psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                spectrum_filename = infer_spectrum_path(self.spectrum_path, run)
                logger.debug(f"Using spectrum file `{spectrum_filename}`")
                try:
                    alphapeptdeep_results = custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        psms_df=psms_df,
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=self.model,
                        ms2_tolerance=self.ms2_tolerance,
                        compute_correlations=True,
                        model_dir=self.model_dir,
                        processes=self.processes,
                        mask_modloss=self._mask_modloss
                    )
                except NoMatchingSpectraFound as e:
                    raise FeatureGeneratorException(
                        f"Could not find any matching spectra for PSMs from run `{run}`. "
                        "Please check that the `spectrum_id_pattern` and `psm_id_pattern` "
                        "options are configured correctly. See "
                        "https://ms2rescore.readthedocs.io/en/latest/userguide/configuration/#mapping-psms-to-spectra"
                        " for more information."
                    ) from e
                self._calculate_features(psm_list_run, alphapeptdeep_results)
                current_run += 1

    def _validate_scores(
            self,
            alphapeptdeep_results,
            calibration_set_size,
            valid_correlations_size,
            correlation_threshold,
            higher_score_better,
    ) -> bool:
        """
        Validate AlphaPeptDeep results based on score and correlation criteria.

        This method checks if the AlphaPeptDeep results meet the specified correlation
        threshold and score criteria. It first filters out decoy PSMs, sorts the
        results based on the PSM score, and selects a calibration set. The method
        then verifies if at least 80% of the calibration set has a correlation
        above the given threshold.

        Parameters
        ----------
        alphapeptdeep_results : list
            List of AlphaPeptDeep results to validate.
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
        if not alphapeptdeep_results:
            return False

        alphapeptdeep_results_copy = (
            alphapeptdeep_results.copy()
        )  # Copy alphapeptdeep results to avoid modifying the original list

        # Select only PSMs that are target and not decoys
        alphapeptdeep_results_copy = [
            result
            for result in alphapeptdeep_results_copy
            if not result.psm.is_decoy and result.psm.rank == 1
        ]
        # Sort alphapeptdeep results by PSM score and lower score is better
        alphapeptdeep_results_copy.sort(key=lambda x: x.psm.score, reverse=higher_score_better)

        # Get a calibration set, the % of psms to be used for calibrarion is defined by calibration_set_size
        calibration_set = alphapeptdeep_results_copy[
                          : int(len(alphapeptdeep_results_copy) * calibration_set_size)
                          ]

        # Select the results with correlation above the threshold
        valid_correlation = [
            psm for psm in calibration_set if psm.correlation >= correlation_threshold
        ]

        logger.info(
            f"The percentage of PSMs in the top {calibration_set_size * 100}% with a correlation greater than {correlation_threshold} is: "
            f"{(len(valid_correlation) / len(calibration_set)) * 100:.2f}%"
        )

        if len(valid_correlation) < len(calibration_set) * valid_correlations_size:
            return False

        return True

    def _find_best_ms2_model(
            self, batch_psms: PSMList, psms_df: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Find the best MS2 model for a batch of PSMs.

        This method finds the best MS2 model for a batch of PSMs by
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

        # AlphaPeptDeep has generic model, So force it to default.
        logger.info(f"Running AlphaPeptDeep for model `{self.model}`...")
        alphapeptdeep_results = custom_correlate(
            psms=batch_psms,
            psms_df=psms_df,
            spectrum_file=self.spectrum_path,
            spectrum_id_pattern=self.spectrum_id_pattern,
            model=self.model,
            ms2_tolerance=self.ms2_tolerance,
            compute_correlations=True,
            model_dir=self.model_dir,
            processes=self.processes,
            mask_modloss=self._mask_modloss
        )

        correlation = self._calculate_correlation(alphapeptdeep_results)
        if correlation >= 0.4:
            best_model = self.model
            best_correlation = correlation

        return best_model, best_correlation

    @staticmethod
    def _calculate_correlation(alphapeptdeep_results: List[ProcessingResult]) -> float:
        """
        Calculate the average correlation from AlphaPeptDeep results.

        This method computes the average correlation score from a list of
        AlphaPeptDeep results, where each result contains a correlation attribute.

        Parameters
        ----------
        alphapeptdeep_results : list
            List of AlphaPeptDeep results, each containing a correlation score.

        Returns
        -------
        float
            The average correlation score of the provided AlphaPeptDeep results.
        """
        total_correlation = sum(
            [
                psm.correlation
                for psm in alphapeptdeep_results
                if psm.correlation is not None and not np.isnan(psm.correlation)
            ]
        )
        return total_correlation / len(alphapeptdeep_results)


def custom_correlate(
        psms: Union[PSMList, str, Path],
        psms_df: pd.DataFrame,
        spectrum_file: Union[str, Path],
        psm_filetype: Optional[str] = None,
        spectrum_id_pattern: Optional[str] = None,
        compute_correlations: bool = False,
        model: Optional[str] = "generic",
        model_dir: Optional[Union[str, Path]] = None,
        ms2_tolerance: float = 0.02,
        mask_modloss: bool = True,
        processes: Optional[int] = None,
) -> List[ProcessingResult]:
    """
    Custom implementation of correlate that uses our custom spectrum reader.
    """
    psm_list = read_psms(psms, filetype=psm_filetype)
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    results = make_prediction(psm_list, psms_df, spectrum_file, spectrum_id_pattern,
                            model, model_dir, ms2_tolerance, processes, mask_modloss)

    # Correlations also requested
    if compute_correlations:
        calculate_correlations(results)
        logger.info(
            f"Median correlation: {np.median([r.correlation for r in results if r.correlation is not None and not np.isnan(r.correlation)])}, model {model}"
        )

    return results


def make_prediction(enumerated_psm_list, psms_df, spec_file, spectrum_id_pattern, model, model_dir,
                    ms2_tolerance, processes, mask_modloss):

    model_mgr = ModelManager(mask_modloss=mask_modloss, device="cpu")
    if model_dir is not None and os.path.exists(os.path.join(model_dir, "ms2.pth")):
        model_mgr.load_external_models(
            ms2_model_file=os.path.join(model_dir, "ms2.pth"))
    else:
        model_mgr.load_installed_models(model)

    results = []
    predictions = model_mgr.predict_all(precursor_df=psms_df, predict_items=["ms2"],
                                        frag_types=['b_z1', 'y_z1', 'b_z2', 'y_z2'],
                                        process_num=processes)
    precusor_df, predict_int_df, theoretical_mz_df = predictions["precursor_df"], predictions[
        "fragment_intensity_df"], predictions["fragment_mz_df"]
    precusor_df = precusor_df.set_index("provenance_data")

    b_cols = [col for col in theoretical_mz_df.columns if col.startswith('b')]
    y_cols = [col for col in theoretical_mz_df.columns if col.startswith('y')]
    # Compile regex for spectrum ID matching
    try:
        spectrum_id_regex = re.compile(spectrum_id_pattern)
    except TypeError:
        spectrum_id_regex = re.compile(r"(.*)")

    # Organize PSMs by spectrum ID
    psms_by_specid = _organize_psms_by_spectrum_id(enumerated_psm_list)

    # Process each spectrum
    for spectrum in read_spectrum_file(spec_file):

        # Match spectrum ID with provided regex
        match = spectrum_id_regex.search(spectrum.identifier)
        try:
            spectrum_id = match[1]
        except (TypeError, IndexError):
            raise exceptions.TitlePatternError(
                f"Spectrum title pattern `{spectrum_id_pattern}` could not be matched to "
                f"spectrum ID `{spectrum.identifier}`. "
                " Are you sure that the regex contains a capturing group?"
            )

        # Skip if no matching PSMs
        if spectrum_id not in psms_by_specid:
            continue

        # # Preprocess spectrum
        # _preprocess_spectrum(spectrum, model)

        # Process each PSM for this spectrum
        for psm_idx, psm in psms_by_specid[spectrum_id]:

            row = precusor_df.loc[next(iter(psm.provenance_data.keys()))]
            mz = theoretical_mz_df.iloc[row["frag_start_idx"]:row["frag_stop_idx"], ]
            b_array_1d = mz[b_cols].values.flatten()
            y_array_1d = mz[y_cols].values.flatten()

            b_targets, y_targets = _get_targets_for_psm(
                b_array_1d, y_array_1d, spectrum, ms2_tolerance
            )
            predict_intensity = predict_int_df.iloc[row["frag_start_idx"]:row["frag_stop_idx"], ]

            results.append(ProcessingResult(
                psm_index=psm_idx,
                psm=psm,
                theoretical_mz={"b": b_array_1d, "y": y_array_1d},
                predicted_intensity={"b": predict_intensity[b_cols].values.flatten(),
                                     "y": predict_intensity[y_cols].values.flatten()},
                observed_intensity={"b": b_targets, "y": y_targets},
                correlation=None,
                feature_vectors=None
            ))


    return results


def calculate_correlations(results: List[ProcessingResult]) -> None:
    """Calculate and add Pearson correlations to list of results."""
    for result in results:
        if result.predicted_intensity and result.observed_intensity:
            pred_int = np.concatenate([i for i in result.predicted_intensity.values()])
            obs_int = np.concatenate([i for i in result.observed_intensity.values()])
            result.correlation = np.corrcoef(pred_int, obs_int)[0][1]
        else:
            result.correlation = None
            logger.info("Results {} is empty".format(result))


def read_spectrum_file(spec_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from a supported file format; inferring the type from the filename extension.

    Parameters
    ----------
    spec_file:
        Path to MGF or mzML file.

    Yields
    ------
    ObservedSpectrum

    Raises
    ------
    UnsupportedSpectrumFiletypeError
        If the file extension is not supported.

    """
    try:
        spectra = OpenMSHelper.get_mslevel_spectra(file_name=str(spec_file), ms_level=2)
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


def _organize_psms_by_spectrum_id(
        enumerated_psm_list: List[PSM]
) -> Dict[str, List[Tuple[int, PSM]]]:
    """
    Organize PSMs by spectrum ID for efficient lookup.

    Parameters
    ----------
    enumerated_psm_list
        List of tuples of (index, PSM) for each PSM in the input file.

    Returns
    -------
    Dict[str, List[Tuple[int, PSM]]]
        Dictionary mapping spectrum IDs to lists of (index, PSM) tuples.
    """
    psms_by_specid = defaultdict(list)
    for index, psm in enumerate(enumerated_psm_list):
        psms_by_specid[str(psm.spectrum_id)].append((index, psm))
    return psms_by_specid


def _preprocess_spectrum(spectrum: ObservedSpectrum, model: str) -> None:
    """
    Preprocess a spectrum by removing reporter ions, normalizing, and transforming.

    Parameters
    ----------
    spectrum
        The spectrum to preprocess.
    model
        The model name, used to determine if reporter ions should be removed.
    """
    # Remove reporter ions if needed
    for label_type in ["iTRAQ", "TMT"]:
        if label_type in model:
            spectrum.remove_reporter_ions(label_type)

    spectrum.tic_norm()
    spectrum.log2_transform()


def _get_targets_for_psm(
        b_frag_mzs: np.array,
        y_frag_mzs: np.array,
        spectrum: ObservedSpectrum,
        ms2_tolerance: float
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Get targets for a PSM from a spectrum.

    Parameters
    ----------
    psm
        The PSM to get targets for.
    spectrum
        The spectrum to get targets from.
    encoder
        The encoder to use for peptide and peptidoform encoding.
    ms2_tolerance
        The MS2 tolerance to use.
    model
        The model name.
    ion_types
        The ion types to use.

    Returns
    -------
    Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]
        A tuple containing the encoded peptidoform and the targets.
    """

    # if ms2_tolerance:
    #     spec_mz_tols = spec_mzs * tol * 1e-6
    # else:
    # print(spectrum.mz)
    # print(ms2_tolerance)  # 0.05
    spec_mz_tols = np.full_like(spectrum.mz, ms2_tolerance)

    b_matched_idxes = match_centroid_mz(spectrum.mz, b_frag_mzs, spec_mz_tols).reshape(-1)
    b_matched_intens = spectrum.intensity[b_matched_idxes]
    b_matched_intens[b_matched_idxes == -1] = 0

    y_matched_idxes = match_centroid_mz(spectrum.mz, y_frag_mzs, spec_mz_tols).reshape(-1)
    y_matched_intens = spectrum.intensity[y_matched_idxes]
    y_matched_intens[y_matched_idxes == -1] = 0
    max_intensity = max(np.max(b_matched_intens), np.max(y_matched_intens))

    if max_intensity == 0:
        return b_matched_intens, y_matched_intens
    return b_matched_intens / max_intensity, y_matched_intens / max_intensity
