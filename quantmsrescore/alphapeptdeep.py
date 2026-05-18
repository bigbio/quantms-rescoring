import pandas as pd
from peptdeep.pretrained_models import ModelManager
from peptdeep.mass_spec.match import match_centroid_mz
from alphabase.peptide.fragment import create_fragment_mz_dataframe
from ms2rescore.feature_generators.base import FeatureGeneratorBase, FeatureGeneratorException
from typing import Optional, Tuple, List, Union, Generator, Dict, Any

from psm_utils import PSMList, PSM
from quantmsrescore.logging_config import get_logger, configure_worker_process
from concurrent.futures import ThreadPoolExecutor, as_completed
from quantmsrescore.openms import (
    OpenMSHelper,
    get_compiled_regex,
    organize_psms_by_spectrum_id,
    calculate_correlations,
)
from quantmsrescore.ms2_model_manager import MS2ModelManager
from ms2rescore.utils import infer_spectrum_path
import ms2pip.exceptions as exceptions
import numpy as np
from math import ceil
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
            ms2_tolerance_unit: str = "Da",
            spectrum_path: Optional[str] = None,
            spectrum_id_pattern: str = "(.*)",
            model_dir: Optional[str] = None,
            consider_modloss: bool = False,
            processes: int = 1,
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
        self.ms2_tolerance_unit = ms2_tolerance_unit
        self.spectrum_path = spectrum_path
        self.spectrum_id_pattern = spectrum_id_pattern
        self.model_dir = model_dir
        self.processes = processes
        self.consider_modloss = consider_modloss

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
                    model_mgr = MS2ModelManager(
                        mask_modloss=not self._consider_modloss,
                        device="cpu",
                        model_dir=self.model_dir
                    )

                    alphapeptdeep_results = custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        psms_df=psms_df,
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=model_mgr,
                        ms2_tolerance=self.ms2_tolerance,
                        ms2_tolerance_unit=self.ms2_tolerance_unit,
                        compute_correlations=False,
                        processes=self.processes,
                        consider_modloss=self.consider_modloss
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

    def _get_pool(self):
        """Get multiprocessing pool with recursion/daemon protection."""
        processes = int(self.processes)
        logger.debug(f"Starting workers (processes={processes})...")

        if multiprocessing.current_process().daemon:
            logger.warning(
                "Running in a daemon process. Disabling multiprocessing as daemonic "
                "processes cannot have children."
            )
            return multiprocessing.dummy.Pool(1)

        if processes == 1:
            logger.debug("Using dummy multiprocessing pool.")
            return multiprocessing.dummy.Pool(1)

        # Check if already inside a worker process
        if multiprocessing.parent_process() is not None:
            logger.warning(
                "Attempting to create a pool inside a worker process! "
                "Returning a dummy pool instead."
            )
            return multiprocessing.dummy.Pool(1)

        return multiprocessing.get_context("spawn").Pool(
            processes,
            initializer=configure_worker_process
        )

    def _calculate_features(
            self, psm_list: PSMList, alphapeptdeep_results: List[ProcessingResult]
    ) -> None:
        """Calculate features from all AlphaPeptDeep results and add to PSMs."""
        logger.debug("Calculating features from predicted spectra")
        with self._get_pool() as pool:
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
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"Initializing rescoring_features for PSM {result.psm_index}: {e}")
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
            consider_modloss: bool = False,
            transfer_learning: bool = False,
            transfer_learning_test_ratio: float = 0.3,
            save_retrain_model: bool = False,
            epoch_to_train_ms2: int = 20,
            **kwargs,
    ):
        super().__init__(
            args,
            model=model,
            ms2_tolerance=ms2_tolerance,
            ms2_tolerance_unit=ms2_tolerance_unit,
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
        self._consider_modloss: bool = consider_modloss
        self._peptdeep_model = None
        self._transfer_learning = transfer_learning
        self._transfer_learning_test_ratio = transfer_learning_test_ratio
        self._save_retrain_model = save_retrain_model
        self._epoch_to_train_ms2 = epoch_to_train_ms2

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
                    model_mgr, transfer_learning = self._get_model_manager_and_transfer_flag()

                    AlphaPeptDeep_results, model_weights = custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        psms_df=psms_df,
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=model_mgr,
                        ms2_tolerance=self.ms2_tolerance,
                        ms2_tolerance_unit=self.ms2_tolerance_unit,
                        compute_correlations=True,
                        processes=self.processes,
                        consider_modloss=self._consider_modloss,
                        higher_score_better=self._higher_score_better,
                        calibration_set_size=self._calibration_set_size,
                        transfer_learning=transfer_learning,
                        transfer_learning_test_ratio=self._transfer_learning_test_ratio,
                        epoch_to_train_ms2=self._epoch_to_train_ms2
                    )
                    self._peptdeep_model = model_weights
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
                    model_mgr, transfer_learning = self._get_model_manager_and_transfer_flag()

                    alphapeptdeep_results, model_weights = custom_correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        psms_df=psms_df,
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=model_mgr,
                        ms2_tolerance=self.ms2_tolerance,
                        ms2_tolerance_unit=self.ms2_tolerance_unit,
                        compute_correlations=True,
                        processes=self.processes,
                        higher_score_better=self._higher_score_better,
                        calibration_set_size=self._calibration_set_size,
                        transfer_learning=transfer_learning,
                        transfer_learning_test_ratio=self._transfer_learning_test_ratio,
                        epoch_to_train_ms2=self._epoch_to_train_ms2
                    )
                    if self._save_retrain_model:
                        model_weights.save_ms2_model()

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

        model_mgr, transfer_learning = self._get_model_manager_and_transfer_flag()

        alphapeptdeep_results, model_weights = custom_correlate(
            psms=batch_psms,
            psms_df=psms_df,
            spectrum_file=self.spectrum_path,
            spectrum_id_pattern=self.spectrum_id_pattern,
            model=model_mgr,
            ms2_tolerance=self.ms2_tolerance,
            ms2_tolerance_unit=self.ms2_tolerance_unit,
            compute_correlations=True,
            processes=self.processes,
            consider_modloss=self._consider_modloss,
            higher_score_better=self._higher_score_better,
            calibration_set_size=self._calibration_set_size,
            transfer_learning=transfer_learning,
            transfer_learning_test_ratio=self._transfer_learning_test_ratio,
            epoch_to_train_ms2=self._epoch_to_train_ms2
        )
        self._peptdeep_model = model_weights

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

    def _get_model_manager_and_transfer_flag(self):
        if self._peptdeep_model:
            return self._peptdeep_model, False
        else:
            model_mgr = MS2ModelManager(
                mask_modloss=not self._consider_modloss,
                device="cpu",
                model_dir=self.model_dir
            )
            return model_mgr, self._transfer_learning


def custom_correlate(
        psms: Union[PSMList, str, Path],
        psms_df: pd.DataFrame,
        spectrum_file: Union[str, Path],
        psm_filetype: Optional[str] = None,
        spectrum_id_pattern: Optional[str] = None,
        compute_correlations: bool = False,
        model: ModelManager = None,
        ms2_tolerance: float = 0.02,
        ms2_tolerance_unit: str = "Da",
        consider_modloss: bool = False,
        processes: Optional[int] = None,
        higher_score_better: bool = True,
        calibration_set_size: Optional[float] = 0.20,
        transfer_learning: bool = True,
        transfer_learning_test_ratio: float = 0.3,
        epoch_to_train_ms2: int = 20
) -> Tuple[Union[List[ProcessingResult], Any], Any]:
    """
    Custom implementation of correlate that uses our custom spectrum reader.
    """
    psm_list = read_psms(psms, filetype=psm_filetype)
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    results, model_weights = ms2_fine_tune(psm_list, psms_df, spectrum_file, spectrum_id_pattern,
                                           model, ms2_tolerance, ms2_tolerance_unit, processes,
                                           consider_modloss, higher_score_better, calibration_set_size,
                                           transfer_learning, transfer_learning_test_ratio, epoch_to_train_ms2)

    # Correlations also requested
    if compute_correlations:
        calculate_correlations(results)
        logger.info(
            f"Median correlation: {np.median([r.correlation for r in results if r.correlation is not None and not np.isnan(r.correlation)])}, model {model}"
        )

    return results, model_weights


def ms2_fine_tune(enumerated_psm_list, psms_df, spec_file, spectrum_id_pattern, model_mgr,
                  ms2_tolerance, ms2_tolerance_unit, processes, consider_modloss, higher_score_better,
                  calibration_set_size, transfer_learning, transfer_learning_test_ratio, epoch_to_train_ms2):
    """
    Fine-tunes the MS2 prediction model using transfer learning on a calibration set of PSMs.

    This function selects a subset of high-confidence PSMs, splits them into calibration and test sets,
    and performs transfer learning on the MS2 prediction model. It then predicts MS2 spectra for all PSMs
    using the (optionally) fine-tuned model.

    Args:
        enumerated_psm_list (List[PSM]): List of PSM objects to use for calibration and prediction.
        psms_df (pd.DataFrame): DataFrame containing precursor information for all PSMs.
        spec_file (str or Path): Path to the spectrum file containing observed spectra.
        spectrum_id_pattern (str): Regex pattern to extract spectrum IDs from the spectrum file.
        model_mgr (ModelManager): The MS2 prediction model manager instance.
        ms2_tolerance (float): Tolerance for matching fragment ions (in Da or ppm).
        ms2_tolerance_unit (str): Unit for ms2_tolerance ("Da" or "ppm").
        processes (int or None): Number of processes to use for parallel prediction.
        consider_modloss (bool): Whether to consider modification losses in fragment ion types.
        higher_score_better (bool): If True, higher PSM scores are better; otherwise, lower scores are better.
        calibration_set_size (float): Fraction of PSMs to use for calibration (transfer learning).
        transfer_learning (bool): Whether to perform transfer learning on the MS2 model.
        transfer_learning_test_ratio (float): Fraction of calibration set to use as test set during transfer learning.
        epoch_to_train_ms2 (int): Number of epochs to train the MS2 model during transfer learning.

    Returns:
        Tuple[List[ProcessingResult], Any]: A tuple containing:
            - List of ProcessingResult objects for each PSM, including predicted and observed spectra and scores.
            - Model weights or state after (optional) transfer learning.

    Raises:
        Exception: If any error occurs during transfer learning or prediction.

    """
    if consider_modloss:
        frag_types = ['b_z1', 'y_z1', 'b_z2', 'y_z2',
                      'b_modloss_z1', 'b_modloss_z2',
                      'y_modloss_z1', 'y_modloss_z2']
    else:
        frag_types = ['b_z1', 'y_z1', 'b_z2', 'y_z2']

    psms_df.rename(columns={"precursor_charge": "charge"}, inplace=True)

    if transfer_learning:
        enumerated_psm_list_copy = enumerated_psm_list.copy()
        # Select only PSMs that are target and not decoys
        enumerated_psm_list_copy = [
            psm
            for psm in enumerated_psm_list_copy
            if not psm.is_decoy and psm.rank == 1
        ]
        # Sort alphapeptdeep results by PSM score and lower score is better
        enumerated_psm_list_copy.sort(key=lambda x: x.score, reverse=higher_score_better)

        # Get a calibration set, the % of psms to be used for calibration is defined by calibration_set_size
        calibration_set = enumerated_psm_list_copy[
                          : int(len(enumerated_psm_list_copy) * calibration_set_size)
                          ]

        precursor_df = psms_df[
            psms_df["provenance_data"].isin([next(iter(psm.provenance_data.keys())) for psm in calibration_set])]

        theoretical_mz_df = create_fragment_mz_dataframe(precursor_df, frag_types)

        precursor_df = precursor_df.set_index("provenance_data")

        # Get cached compiled regex for spectrum ID matching
        spectrum_id_regex = get_compiled_regex(spectrum_id_pattern)

        # Organize PSMs by spectrum ID
        psms_by_specid = organize_psms_by_spectrum_id(calibration_set)

        match_intensity_df = []
        current_index = 0
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

            # Process each PSM for this spectrum
            for psm_idx, psm in psms_by_specid[spectrum_id]:
                row_index = next(iter(psm.provenance_data.keys()))
                row = precursor_df.loc[row_index]
                mz = theoretical_mz_df.iloc[row["frag_start_idx"]:row["frag_stop_idx"], ]
                match_intensity = _get_targets_df_for_psm(
                    mz, spectrum, ms2_tolerance, ms2_tolerance_unit
                )
                fragment_len = match_intensity.shape[0]
                precursor_df.loc[row_index, "match_start_idx"] = current_index
                precursor_df.loc[row_index, "match_stop_idx"] = current_index + fragment_len
                match_intensity_df.append(match_intensity)
                current_index += fragment_len

        match_intensity_df = pd.concat(match_intensity_df, ignore_index=True)

        psm_num_to_train_ms2 = int(len(calibration_set) * (1 - transfer_learning_test_ratio))
        psm_num_to_test_ms2 = len(calibration_set) - psm_num_to_train_ms2

        precursor_df.drop(columns=["frag_start_idx", "frag_stop_idx"], inplace=True)
        precursor_df.rename(columns={
            "match_start_idx": "frag_start_idx",
            "match_stop_idx": "frag_stop_idx"
        }, inplace=True)
        precursor_df["frag_start_idx"] = precursor_df["frag_start_idx"].astype("int64")
        precursor_df["frag_stop_idx"] = precursor_df["frag_stop_idx"].astype("int64")

        model_mgr.ms2_fine_tuning(precursor_df,
                                  match_intensity_df,
                                  psm_num_to_train_ms2=psm_num_to_train_ms2,
                                  psm_num_to_test_ms2=psm_num_to_test_ms2,
                                  train_verbose=True,
                                  epoch_to_train_ms2=epoch_to_train_ms2)

    predictions = model_mgr.predict_all(precursor_df=psms_df, predict_items=["ms2"],
                                        frag_types=frag_types,
                                        process_num=processes)
    precursor_df, predict_int_df, theoretical_mz_df = predictions["precursor_df"], predictions[
        "fragment_intensity_df"], predictions["fragment_mz_df"]

    results = []

    precursor_df = precursor_df.set_index("provenance_data")
    precursor_dict = precursor_df.to_dict("index")
    psms_df.rename(columns={"charge": "precursor_charge"}, inplace=True)

    b_cols = [col for col in theoretical_mz_df.columns if col.startswith('b')]
    y_cols = [col for col in theoretical_mz_df.columns if col.startswith('y')]
    b_idx = [theoretical_mz_df.columns.tolist().index(c) for c in b_cols]
    y_idx = [theoretical_mz_df.columns.tolist().index(c) for c in y_cols]

    # Convert DataFrame once
    theoretical_mz_df = theoretical_mz_df.values
    predict_int_df = predict_int_df.values

    # Split by spectrum_id to keep PSMs for same spectrum together
    def _enumerated_psm_list_by_spectrum_id(psm_list, spectrum_ids_chunk):
        selected_indices = np.flatnonzero(np.isin(psm_list["spectrum_id"], spectrum_ids_chunk))
        return [(i, psm_list.psm_list[i]) for i in selected_indices]

    def get_chunk_size(n_items, n_processes):
        """Get optimal chunk size for multiprocessing."""
        if n_items < 5000:
            return n_items
        max_chunk_size = 50000
        n_chunks = ceil(ceil(n_items / n_processes) / max_chunk_size) * n_processes
        return ceil(n_items / n_chunks)

    def to_chunks(_list, chunk_size):
        """Split _list into chunks of size chunk_size."""

        def _generate_chunks():
            for i in range(0, len(_list), chunk_size):
                yield _list[i: i + chunk_size]

        _list = list(_list)
        return list(_generate_chunks())

    spectrum_ids = set(enumerated_psm_list["spectrum_id"])
    chunk_size = get_chunk_size(len(spectrum_ids), processes)
    psm_chunks = [
        _enumerated_psm_list_by_spectrum_id(enumerated_psm_list, spectrum_ids_chunk)
        for spectrum_ids_chunk in to_chunks(spectrum_ids, chunk_size)
    ]
    logger.info(f"Processing {len(psm_chunks)} chunk(s) of ~{chunk_size} entries each.")

    with ThreadPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(
                process_psm_chunk,
                chunk,
                spec_file,
                precursor_dict,
                theoretical_mz_df,
                predict_int_df,
                b_idx,
                y_idx,
                ms2_tolerance,
                ms2_tolerance_unit,
                spectrum_id_pattern,
            )
            for chunk in psm_chunks
        ]

        for future in as_completed(futures):
            results.extend(future.result())

    return results, model_mgr


def process_psm_chunk(
        psm_chunk,
        spec_file,
        precursor_dict,
        theoretical_mz_np,
        predict_int_np,
        b_cols,
        y_cols,
        ms2_tolerance,
        ms2_tolerance_unit,
        spectrum_id_pattern
):
    # Organize PSMs by spectrum ID

    # Get cached compiled regex for spectrum ID matching
    spectrum_id_regex = get_compiled_regex(spectrum_id_pattern)

    psms_by_specid = organize_psms_by_spectrum_id(psm_chunk)
    local_results = []

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
            row = precursor_dict[next(iter(psm.provenance_data.keys()))]
            start_idx = row["frag_start_idx"]
            stop_idx = row["frag_stop_idx"]
            mz = theoretical_mz_np[start_idx:stop_idx]

            b_array_1d = mz[:, b_cols].reshape(-1)
            y_array_1d = mz[:, y_cols].reshape(-1)

            b_mask = b_array_1d != 0.0
            y_mask = y_array_1d != 0.0

            b_targets, y_targets = _get_targets_for_psm(
                b_array_1d[b_mask], y_array_1d[y_mask], spectrum, ms2_tolerance, ms2_tolerance_unit
            )
            predict_intensity = predict_int_np[start_idx:stop_idx]

            b_pred = predict_intensity[:, b_cols].reshape(-1)[b_mask]
            y_pred = predict_intensity[:, y_cols].reshape(-1)[y_mask]

            local_results.append(ProcessingResult(
                psm_index=psm_idx,
                psm=psm,
                theoretical_mz={"b": b_array_1d[b_mask], "y": y_array_1d[y_mask]},
                predicted_intensity={"b": b_pred,
                                     "y": y_pred},
                observed_intensity={"b": b_targets, "y": y_targets},
                correlation=None,
                feature_vectors=None
            ))
    return local_results


def read_spectrum_file(spec_file: str, use_cache: bool = True) -> Generator[ObservedSpectrum, None, None]:
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
        ms2_tolerance: float,
        ms2_tolerance_unit: str
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

    if ms2_tolerance_unit == "ppm":
        spec_mz_tols = spectrum.mz * ms2_tolerance * 1e-6
    else:
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


def _get_targets_df_for_psm(
        mz_df: pd.DataFrame,
        spectrum: ObservedSpectrum,
        ms2_tolerance: float,
        ms2_tolerance_unit: str
) -> pd.DataFrame:
    """
    Get normalized target intensities for theoretical fragment ions from a spectrum.

    Parameters
    ----------
    mz_df : pd.DataFrame
        DataFrame containing theoretical m/z values for fragment ions.
    spectrum : ObservedSpectrum
        The observed spectrum to match against.
    ms2_tolerance : float
        The MS2 tolerance to use for matching.
    ms2_tolerance_unit : str
        The unit of MS2 tolerance ('ppm' or 'Da').
    Returns
    -------
    pd.DataFrame
        DataFrame of normalized intensities for matched fragment ions.
    """

    theo_mz = mz_df.values.flatten()

    if ms2_tolerance_unit == "ppm":
        spec_mz_tols = spectrum.mz * ms2_tolerance * 1e-6
    else:
        spec_mz_tols = np.full_like(spectrum.mz, ms2_tolerance)
    # match
    idx = match_centroid_mz(spectrum.mz, theo_mz, spec_mz_tols).reshape(-1)
    intens = np.where(idx >= 0, spectrum.intensity[idx], 0.0)

    # reshape  DataFrame
    out = pd.DataFrame(intens.reshape(mz_df.shape), columns=mz_df.columns)

    # normalize
    max_intensity = out.values.max()
    if max_intensity > 0:
        out = out / max_intensity

    return out
