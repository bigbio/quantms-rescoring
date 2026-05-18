import copy
import gc
from pathlib import Path
from typing import Optional, Set, Union
from psm_utils import PSMList, PSM
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from quantmsrescore.deeplc import DeepLCAnnotator
from quantmsrescore.idparquet_reader import ParquetRescoringReader
from quantmsrescore.logging_config import get_logger
from quantmsrescore.ms2pip import MS2PIPAnnotator
from quantmsrescore.openms import OpenMSHelper, clear_spectrum_cache
from quantmsrescore.alphapeptdeep import AlphaPeptDeepAnnotator

# Get logger for this module
logger = get_logger(__name__)


def _shallow_copy_psm_list(psm_list: PSMList) -> PSMList:
    """
    Create a shallow copy of a PSMList with fresh rescoring_features dicts.

    This is much more memory-efficient than copy.deepcopy() because it only
    creates new containers for mutable data that will be modified (rescoring_features),
    while sharing immutable data like peptide sequences and spectrum IDs.

    Parameters
    ----------
    psm_list : PSMList
        The PSMList to copy.

    Returns
    -------
    PSMList
        A new PSMList with shallow-copied PSMs.
    """
    copied_psms = []
    for psm in psm_list.psm_list:
        # Create a new PSM with the same attributes but a fresh rescoring_features dict
        # PSM attributes are mostly immutable (strings, numbers, tuples)
        new_psm = PSM(
            peptidoform=psm.peptidoform,
            spectrum_id=psm.spectrum_id,
            run=psm.run,
            collection=psm.collection,
            spectrum=psm.spectrum,
            is_decoy=psm.is_decoy,
            score=psm.score,
            qvalue=psm.qvalue,
            pep=psm.pep,
            precursor_mz=psm.precursor_mz,
            retention_time=psm.retention_time,
            ion_mobility=psm.ion_mobility,
            protein_list=psm.protein_list.copy() if psm.protein_list else [],
            rank=psm.rank,
            source=psm.source,
            provenance_data=psm.provenance_data.copy() if psm.provenance_data else {},
            # Can share as keys are read-only
            metadata=psm.metadata.copy() if psm.metadata else {},
            rescoring_features={},  # Fresh dict - this is what will be modified
        )
        copied_psms.append(new_psm)
    return PSMList(psm_list=copied_psms)


class FeatureAnnotator:
    """
    Annotator for peptide-spectrum matches (PSMs) using MS2PIP and DeepLC models.

    This class handles the annotation of PSMs with additional features generated
    from MS2PIP and DeepLC models to improve rescoring.
    """

    def __init__(self, feature_generators: str, only_features: Optional[str] = None, ms2_model: str = "HCD2021",
                 force_model: bool = False, ms2_model_path: str = "models", ms2_tolerance: float = 0.05,
                 ms2_tolerance_unit: str = "Da", calibration_set_size: float = 0.2,
                 valid_correlations_size: float = 0.7,
                 skip_deeplc_retrain: bool = False, processes: int = 2, log_level: str = "INFO",
                 spectrum_id_pattern: str = "(.*)", psm_id_pattern: str = "(.*)", remove_missing_spectra: bool = True,
                 ms2_only: bool = True, find_best_model: bool = False, consider_modloss: bool = False,
                 transfer_learning: bool = False, transfer_learning_test_ratio: float = 0.3,
                 save_retrain_model: bool = False, epoch_to_train_ms2: int = 20) -> None:
        """
        Initialize the Annotator with configuration parameters.

        Parameters
        ----------
        feature_generators : str
            Comma-separated list of feature generators (e.g., "ms2pip,deeplc").
        only_features : str, optional
            Comma-separated list of features to include in annotation.
        ms2_model : str, optional
            MS2 model name (default: "HCD2021").
        ms2_model_path : str, optional
            Path to MS2 model directory (default: "./").
        ms2_tolerance : float, optional
            MS2 tolerance for feature generation (default: 0.05).
        calibration_set_size : float, optional
            Percentage of PSMs to use for calibration (default: 0.2).
        skip_deeplc_retrain : bool, optional
            Skip retraining the deepLC model (default: False).
        processes : int, optional
            Number of parallel processes (default: 2).
        log_level : str, optional
            Logging level (default: "INFO").
        spectrum_id_pattern : str, optional
            Pattern for identifying spectrum IDs (default: "(.*)").
        psm_id_pattern : str, optional
            Pattern for identifying PSM IDs (default: "(.*)").
        remove_missing_spectra : bool, optional
            Remove PSMs with missing spectra (default: True).
        ms2_only : bool, optional
            Process only MS2-level PSMs (default: True).
        force_model : bool, optional
            Force the use of the provided MS2 model (default: False).
        find_best_model : bool, optional
            Force the use of the best MS2 model (default: False).
        consider_modloss: bool, optional
            If modloss ions are considered in the ms2 model. `modloss`
            ions are mostly useful for phospho MS2 prediction model.
            Defaults to True.
        transfer_learning : bool, required
            Whether to use MS2 transfer learning. Set to True to enable transfer learning for MS2 model.
        transfer_learning_test_ratio: float, optional
            The ratio of test data for MS2 transfer learning.
            Defaults to 0.3.
        save_retrain_model: bool, optional
            Save retrained MS2 model.
            Defaults to False.
        epoch_to_train_ms2: int, optional
            Epochs to train AlphaPeptDeep MS2 model.
            Defaults to 20.
        Raises
        ------
        ValueError
            If no feature generators are provided or if neither ms2pip nor deeplc is specified.
        """
        # Set up logging
        from quantmsrescore.logging_config import configure_logging

        configure_logging(log_level)

        # Validate inputs
        if not feature_generators:
            raise ValueError("feature_generators must be provided.")

        feature_annotators = feature_generators.split(",")
        if not any(annotator in feature_annotators for annotator in ["deeplc", "ms2pip", "alphapeptdeep"]):
            raise ValueError("At least one of deeplc or ms2pip or alphapeptdeep must be provided.")

        # Initialize state
        self._idparquet_reader = None
        self._idparquet_psm = None
        self._idparquet_search_param = None
        self._idparquet_proteins = None
        self._idparquet_protein_groups = None

        self._deepLC = "deeplc" in feature_annotators
        if "ms2pip" in feature_annotators and ms2_tolerance_unit == "Da":
            self._ms2pip = True
        elif "ms2pip" in feature_annotators and ms2_tolerance_unit == "ppm":
            raise ValueError(
                "MS2PIP only supports Da units. Please remove 'ms2pip' from feature_generators or set ms2_tolerance_unit to 'Da'.")
        else:
            self._ms2pip = False
        if "alphapeptdeep" in feature_annotators:
            self._alphapeptdeep = True
        else:
            self._alphapeptdeep = False
        self.ms2_generator = None

        # Parse and validate features
        self._only_features = []
        if only_features:
            self._only_features = OpenMSHelper.validate_features(only_features.split(","))

        # Store configuration
        self._ms2_model = ms2_model
        self._ms2_model_path = ms2_model_path
        self._ms2_tolerance = ms2_tolerance
        self._ms2_tolerance_unit = ms2_tolerance_unit
        self._calibration_set_size = calibration_set_size
        self._valid_correlations_size = valid_correlations_size
        self._processes = processes
        self._higher_score_better = None
        self._spectrum_id_pattern = spectrum_id_pattern
        self._psm_id_pattern = psm_id_pattern
        self._skip_deeplc_retrain = skip_deeplc_retrain
        self._remove_missing_spectra = remove_missing_spectra
        self._ms2_only = ms2_only
        self._force_model = force_model
        self._find_best_model = find_best_model
        self._consider_modloss = consider_modloss
        self._transfer_learning = transfer_learning
        self._transfer_learning_test_ratio = transfer_learning_test_ratio
        self._save_retrain_model = save_retrain_model
        self._epoch_to_train_ms2 = epoch_to_train_ms2

    def build_consensus_idparquet(self, parquet_files, spectrum_path):
        try:
            self._idparquet_reader = ParquetRescoringReader(parquet_files,
                                                            spectrum_path,
                                                            only_ms2=self._ms2_only,
                                                            remove_missing_spectrum=self._remove_missing_spectra,
                                                            )
            self._higher_score_better = self._idparquet_reader.high_score_better

            openms_helper = OpenMSHelper()
            decoys, targets = openms_helper.count_decoys_targets(self._idparquet_reader.psms_df)
            logger.info(
                f"Loaded {len(self._idparquet_reader.psms)} PSMs from {parquet_files}: {decoys} decoys and {targets} targets"
            )

        except Exception as e:
            logger.error(f"Failed to load input files: {str(e)}")
            raise

    def annotate(self) -> None:
        """
        Annotate PSMs with MS2PIP and/or DeepLC features.

        This method runs the selected feature generators to add annotations
        to the loaded PSMs.

        Raises
        ------
        ValueError
            If no idXML data is loaded.
        """
        if not self._idparquet_reader:
            raise ValueError("No idXML data loaded. Call build_consensus_idparquet first.")

        logger.debug(f"Running annotations with configuration: {self.__dict__}")

        # find_best_model between MS2PIP and AlphaPeptDeep
        if self._find_best_model:
            self._find_and_apply_ms2_model()
        elif self._ms2pip:
            self._run_ms2pip_annotation()
            self.ms2_generator = "MS2PIP"
        elif self._alphapeptdeep:
            if (2, 'HCD') not in self._idparquet_reader._stats.ms_level_dissociation_method:
                logger.error(
                    "Found not HCD dissociation methods"
                    "AlphaPeptdeep pretrained models are not trained for not HCD dissociation methods"
                )
            self._run_alphapeptdeep_annotation()
            self.ms2_generator = "AlphaPeptDeep"

        # Run DeepLC annotation if enabled
        if self._deepLC:
            self._run_deeplc_annotation()

        # Convert features to OpenMS format if any annotations were added
        if self._ms2pip or self._alphapeptdeep or self._find_best_model or self._deepLC:
            # self._convert_features_psms_to_oms_peptides()
            self._convert_features_psms_to_idparquet()
        # Clear spectrum cache to free memory after annotation is complete
        clear_spectrum_cache()
        gc.collect()

        logger.info("Annotation complete")

    def write_idparquet_file(self, filename: Union[str, Path]) -> None:
        """
        Write annotated data to idparquet file.

        Parameters
        ----------
        filename : Union[str, Path]
            Path where the annotated idparquet file will be written.

        Raises
        ------
        Exception
            If writing the file fails.
        """
        output_dir = Path(filename)
        output_dir.mkdir(parents=True, exist_ok=True)

        psm_file = output_dir / "psms.parquet"
        search_param_file = output_dir / "search_params.parquet"
        proteins_file = output_dir / "proteins.parquet"
        protein_groups_file = output_dir / "protein_groups.parquet"

        try:
            out_path = Path(filename)
            pq.write_table(self._idparquet_psm, psm_file)
            logger.info(f"psms.parquet file written to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write psms.parquet psm file: {str(e)}")
            raise

        # search_params.parquet
        try:
            pq.write_table(self._idparquet_search_param, search_param_file)
            logger.info(f"search_params.parquet written to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write search_params.parquet file: {str(e)}")
            raise

        # proteins.parquet
        try:
            pq.write_table(self._idparquet_proteins, proteins_file)
            logger.info(f"proteins.parquet written to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write proteins.parquet file: {str(e)}")
            raise

        # search_params.parquet
        try:
            pq.write_table(self._idparquet_protein_groups, protein_groups_file)
            logger.info(f"protein_groups.parquet written to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write protein_groups.parquet file: {str(e)}")
            raise

    def _run_ms2pip_annotation(self) -> None:
        """Run MS2PIP annotation on the loaded PSMs."""
        logger.info("Running MS2PIP annotation")

        # Initialize MS2PIP annotator
        try:
            ms2pip_generator = self._create_ms2pip_annotator()
        except Exception as e:
            logger.error(f"Failed to initialize MS2PIP: {e}")
            raise

        # Get PSM list
        psm_list = self._idparquet_reader.psms

        try:
            # Save original model for reference
            original_model = ms2pip_generator.model

            # Determine which model to use based on configuration and validation
            model_to_use = original_model

            # Case 1: Force specific model regardless of validation
            if self._force_model:
                model_to_use = original_model
                logger.info(f"Using forced model: {model_to_use}")

            # Case 2: Find best model if requested and not forcing original
            elif self._find_best_model:
                best_model, best_corr = ms2pip_generator._find_best_ms2pip_model(psm_list)
                if best_model and ms2pip_generator.validate_features(psm_list=psm_list, model=best_model):
                    model_to_use = best_model
                    logger.info(f"Using best model: {model_to_use} with correlation: {best_corr:.4f}")
                else:
                    # Fallback to original model if best model doesn't validate
                    if ms2pip_generator.validate_features(psm_list, model=original_model):
                        logger.warning("Best model validation failed, falling back to original model")
                    else:
                        logger.error("Both best model and original model validation failed")
                        return  # Exit early since no valid model is available

            # Case 3: Use original model but validate it first
            else:
                if not ms2pip_generator.validate_features(psm_list):
                    logger.error("Original model validation failed. No features added.")
                    return  # Exit early since validation failed
                logger.info(f"Using original model: {model_to_use}")

            # Apply the selected model
            ms2pip_generator.model = model_to_use
            ms2pip_generator.add_features(psm_list)
            logger.info(f"Successfully applied MS2PIP annotation using model: {model_to_use}")

        except Exception as e:
            logger.error(f"Failed to apply MS2PIP annotation: {e}", exc_info=True)
            return  # Indicate failure through early return

        return  # Successful completion

    def _run_alphapeptdeep_annotation(self) -> None:
        """Run Alphapeptdeep annotation on the loaded PSMs."""
        logger.info("Running Alphapeptdeep annotation")

        # Initialize Alphapeptdeep annotator
        try:
            alphapeptdeep_generator = self._create_alphapeptdeep_annotator()
        except Exception as e:
            logger.error(f"Failed to initialize alphapeptdeep: {e}")
            raise

        # Get PSM list
        psm_list = self._idparquet_reader.psms
        psms_df = self._idparquet_reader.psms_df

        try:
            # Save original model for reference
            original_model = alphapeptdeep_generator.model

            # Determine which model to use based on configuration and validation
            model_to_use = original_model

            # Case 1: Force specific model regardless of validation
            if self._force_model:
                model_to_use = original_model
                logger.info(f"Using forced model: {model_to_use}")

            else:
                if not alphapeptdeep_generator.validate_features(psm_list, psms_df):
                    logger.error("Original model validation failed. No features added.")
                    return  # Exit early since validation failed
                logger.info(f"Using original model: {model_to_use}")

            # Apply the selected model
            alphapeptdeep_generator.model = model_to_use
            alphapeptdeep_generator.add_features(psm_list, psms_df)
            logger.info(
                f"Successfully applied AlphaPeptDeep annotation using model: {str(alphapeptdeep_generator._peptdeep_model)}")

        except Exception as e:
            logger.error(f"Failed to apply AlphaPeptDeep annotation: {e}", exc_info=True)
            return  # Indicate failure through early return

        return  # Successful completion

    def _create_alphapeptdeep_annotator(self, model: Optional[str] = None, tolerance: Optional[float] = None,
                                        tolerance_unit: Optional[str] = None):
        """
        Create an AlphaPeptDeep annotator with the specified or default model.

        Parameters
        ----------
        model : str, optional
            AlphaPeptDeep model name to use, defaults to generic if None.

        Returns
        -------
        AlphaPeptDeep
            Configured AlphaPeptDeep annotator.
        """

        return AlphaPeptDeepAnnotator(
            ms2_tolerance=tolerance or self._ms2_tolerance,
            ms2_tolerance_unit=tolerance_unit or self._ms2_tolerance_unit,
            model=model or "generic",
            spectrum_path=self._idparquet_reader.spectrum_path,
            spectrum_id_pattern=self._spectrum_id_pattern,
            model_dir=self._ms2_model_path,
            calibration_set_size=self._calibration_set_size,
            valid_correlations_size=self._valid_correlations_size,
            correlation_threshold=0.7,  # Consider making this configurable
            higher_score_better=self._higher_score_better,
            processes=self._processes,
            force_model=self._force_model,
            consider_modloss=self._consider_modloss,
            transfer_learning=self._transfer_learning,
            transfer_learning_test_ratio=self._transfer_learning_test_ratio,
            save_retrain_model=self._save_retrain_model,
            epoch_to_train_ms2=self._epoch_to_train_ms2
        )

    def _create_ms2pip_annotator(
            self, model: Optional[str] = None, tolerance: Optional[float] = None
    ) -> MS2PIPAnnotator:
        """
        Create an MS2PIP annotator with the specified or default model.

        Parameters
        ----------
        model : str, optional
            MS2PIP model name to use, defaults to self._ms2pip_model if None.

        Returns
        -------
        MS2PIPAnnotator
            Configured MS2PIP annotator.
        """
        return MS2PIPAnnotator(
            ms2_tolerance=tolerance or self._ms2_tolerance,
            model=model or self._ms2_model,
            spectrum_path=self._idparquet_reader.spectrum_path,
            spectrum_id_pattern=self._spectrum_id_pattern,
            model_dir=self._ms2_model_path,
            calibration_set_size=self._calibration_set_size,
            valid_correlations_size=self._valid_correlations_size,
            correlation_threshold=0.7,  # Consider making this configurable
            higher_score_better=self._higher_score_better,
            processes=self._processes,
            force_model=self._force_model
        )

    def _validate_and_apply_alphapeptdeep_model(self, alphapeptdeep_generator, alphapeptdeep_best_model,
                                                alphapeptdeep_best_corr, psm_list, psms_df, original_model):
        """
        Validate and apply AlphaPeptDeep model to the PSM list, with fallback to original model if needed.

        Parameters
        ----------
        alphapeptdeep_generator : AlphaPeptDeepAnnotator
            The AlphaPeptDeep annotator instance.
        alphapeptdeep_best_model : str
            The best model to use.
        alphapeptdeep_best_corr : float
            The correlation of the best model.
        psm_list : PSMList
            List of PSMs to annotate.
        psms_df : pd.DataFrame
            DataFrame of PSMs.
        original_model : str
            The original/fallback model to use if best model validation fails.
        """
        model_to_use = alphapeptdeep_best_model  # AlphaPeptdeep only has generic model
        if alphapeptdeep_best_model and alphapeptdeep_generator.validate_features(psm_list=psm_list,
                                                                                  psms_df=psms_df,
                                                                                  model=alphapeptdeep_best_model):
            logger.info(
                f"Using best model: {alphapeptdeep_best_model} with correlation: {alphapeptdeep_best_corr:.4f}")
        else:
            # Fallback to original model if best model doesn't validate
            if alphapeptdeep_generator.validate_features(psm_list=psm_list, psms_df=psms_df,
                                                         model=original_model):
                logger.warning("Best model validation failed, falling back to original model")
                model_to_use = original_model
            else:
                logger.error("Both best model and original model validation failed")
                return False  # Indicate failure

        # Apply the selected model
        alphapeptdeep_generator.model = model_to_use
        alphapeptdeep_generator.add_features(psm_list, psms_df)
        logger.info(f"Successfully applied AlphaPeptDeep annotation using model: {model_to_use}")
        self.ms2_generator = "AlphaPeptDeep"
        return True  # Indicate success

    def _find_and_apply_ms2_model(self):
        """
        Find and apply the best MS2 model for the dataset.

        Parameters
        ----------
        psm_list : PSMList
            List of PSMs to annotate.
        """
        logger.info("Finding best MS2 model for the dataset")
        if (2, 'HCD') not in self._idparquet_reader._stats.ms_level_dissociation_method:
            is_HCD = False
        else:
            is_HCD = True
        if is_HCD:
            # Initialize AlphaPeptDeep annotator
            try:
                alphapeptdeep_generator = self._create_alphapeptdeep_annotator(model="generic")
            except Exception as e:
                logger.error(f"Failed to initialize AlphaPeptDeep: {e}")
                raise

        # Initialize MS2PIP annotator
        if self._ms2_tolerance_unit == "Da":
            try:
                ms2pip_generator = self._create_ms2pip_annotator()
                original_model = ms2pip_generator.model
            except Exception as e:
                logger.error(f"Failed to initialize MS2PIP: {e}")
                raise
        elif is_HCD:
            original_model = alphapeptdeep_generator.model
        else:
            logger.error("Failed to initialize all models")

        # Get PSM list
        psm_list = self._idparquet_reader.psms
        psms_df = self._idparquet_reader.psms_df

        try:
            batch_psms_copy = (
                psm_list.copy()
            )  # Copy ms2pip results to avoid modifying the original list

            # Select only PSMs that are target and not decoys
            calibration_set = [
                result
                for result in batch_psms_copy.psm_list
                if not result.is_decoy and result.rank == 1
            ]
            calibration_set = PSMList(psm_list=calibration_set)
            psms_df_without_decoy = psms_df[psms_df["is_decoy"] == 0]

            if is_HCD:
                logger.info("Running AlphaPeptDeep model")
                alphapeptdeep_best_model, alphapeptdeep_best_corr = alphapeptdeep_generator._find_best_ms2_model(
                    calibration_set, psms_df_without_decoy)
            else:
                alphapeptdeep_best_model, alphapeptdeep_best_corr = None, -1

            ms2pip_best_corr = -1  # Initial MS2PIP best correlation
            ms2pip_best_model = None

            # Determine which model to use based on configuration and validation
            if self._ms2_tolerance_unit == "Da":
                # Save original model for reference
                logger.info("Running MS2PIP model")
                ms2pip_best_model, ms2pip_best_corr = ms2pip_generator._find_best_ms2pip_model(calibration_set)
            else:
                logger.info("MS2PIP model doesn't support ppm tolerance unit. Only consider AlphaPeptDeep model")

            # When using ppm tolerance, only AlphaPeptDeep is supported
            if self._ms2_tolerance_unit != "Da":
                alphapeptdeep_original_model = alphapeptdeep_generator.model
                if not self._validate_and_apply_alphapeptdeep_model(alphapeptdeep_generator, alphapeptdeep_best_model,
                                                                    alphapeptdeep_best_corr, psm_list, psms_df,
                                                                    alphapeptdeep_original_model):
                    return  # Exit early since no valid model is available

            # When using Da tolerance, compare AlphaPeptDeep and MS2PIP
            elif is_HCD and alphapeptdeep_best_corr > ms2pip_best_corr:
                alphapeptdeep_original_model = alphapeptdeep_generator.model
                if not self._validate_and_apply_alphapeptdeep_model(alphapeptdeep_generator, alphapeptdeep_best_model,
                                                                    alphapeptdeep_best_corr, psm_list, psms_df,
                                                                    alphapeptdeep_original_model):
                    return  # Exit early since no valid model is available

            else:
                # Use MS2PIP when Da tolerance and ms2pip has better correlation
                if ms2pip_best_model and ms2pip_generator.validate_features(psm_list=psm_list, model=ms2pip_best_model):
                    model_to_use = ms2pip_best_model
                    logger.info(f"Using best model: {model_to_use} with correlation: {ms2pip_best_corr:.4f}")
                else:
                    # Fallback to original model if best model doesn't validate
                    if ms2pip_generator.validate_features(psm_list,
                                                          model=original_model if original_model != "generic" else "HCD2021"):
                        logger.warning("Best model validation failed, falling back to original model")
                        model_to_use = original_model if original_model != "generic" else "HCD2021"
                    else:
                        logger.error("Both best model and original model validation failed")
                        return  # Exit early since no valid model is available

                # Apply the selected model
                ms2pip_generator.model = model_to_use
                ms2pip_generator.add_features(psm_list)
                logger.info(f"Successfully applied MS2PIP annotation using model: {model_to_use}")
                self.ms2_generator = "MS2PIP"

        except Exception as e:
            logger.error(f"Failed to apply MS2 annotation: {e}")
            return  # Indicate failure through early return

        return  # Successful completion

    def _run_deeplc_annotation(self) -> None:
        """Run DeepLC annotation on the loaded PSMs."""
        logger.info("Running DeepLC annotation")

        try:
            if self._skip_deeplc_retrain:
                # Simple case - use pre-trained model
                deeplc_annotator = self._create_deeplc_annotator(retrain=False)
            else:
                # Compare retrained vs pretrained performance
                deeplc_annotator = self._determine_optimal_deeplc_model()

            # Apply annotation
            psm_list = self._idparquet_reader.psms
            deeplc_annotator.add_features(psm_list)
            self._idparquet_reader.psms = psm_list
            logger.info("DeepLC annotations added to PSMs")

        except Exception as e:
            logger.error(f"Failed to apply DeepLC annotation: {e}")
            raise

    def _create_deeplc_annotator(
            self, retrain: bool = False, calibration_set_size: float = None
    ) -> DeepLCAnnotator:
        """
        Create a DeepLC annotator with specified configuration.

        Parameters
        ----------
        retrain : bool
            Whether to retrain the DeepLC model.

        Returns
        -------
        DeepLCAnnotator
            Configured DeepLC annotator.
        """
        kwargs = {"deeplc_retrain": retrain}

        if calibration_set_size is None:
            calibration_set_size = self._calibration_set_size

        return DeepLCAnnotator(
            not self._higher_score_better,
            calibration_set_size=calibration_set_size,
            processes=self._processes,
            **kwargs,
        )

    def _determine_optimal_deeplc_model(self) -> DeepLCAnnotator:
        """
        Determine the optimal DeepLC model by comparing retrained vs. pretrained performance.

        This function evaluates both a retrained model and a pretrained model on the same dataset,
        calculates the Mean Absolute Error (MAE) for each, and selects the model with lower error.

        Returns
        -------
        DeepLCAnnotator
            The DeepLC annotator with the lowest MAE (best performance).

        Notes
        -----
        Uses shallow copies of PSM lists instead of deep copies for memory efficiency.
        The rescoring_features dict is the only mutable part that needs to be fresh.
        """
        # Evaluate retrained model using shallow copy (memory efficient)
        retrained_psms = _shallow_copy_psm_list(self._idparquet_reader.psms)
        retrained_model = self._create_deeplc_annotator(retrain=True, calibration_set_size=0.6)
        retrained_model.add_features(retrained_psms)
        mae_retrained = self._get_mae_from_psm_list(retrained_psms)

        # Clean up retrained PSMs if we don't need them
        del retrained_psms
        gc.collect()

        # Evaluate pretrained model using shallow copy (memory efficient)
        pretrained_psms = _shallow_copy_psm_list(self._idparquet_reader.psms)
        pretrained_model = self._create_deeplc_annotator(retrain=False, calibration_set_size=0.6)
        pretrained_model.add_features(pretrained_psms)
        mae_pretrained = self._get_mae_from_psm_list(pretrained_psms)

        # Clean up pretrained PSMs
        del pretrained_psms
        gc.collect()

        # Select model with lower MAE
        if mae_retrained < mae_pretrained:
            logger.info(
                f"Retrained DeepLC model has lower MAE ({mae_retrained:.4f} vs {mae_pretrained:.4f}), using it: {retrained_model.selected_model}"
            )
            return retrained_model
        else:
            logger.info(
                f"Pretrained DeepLC model has lower/equal MAE ({mae_pretrained:.4f} vs {mae_retrained:.4f}), using it: {pretrained_model.selected_model}"
            )
            return pretrained_model

    def _convert_features_psms_to_idparquet(self) -> None:
        """
        Transfer features from PSM objects to idparquet objects.
        """
        records = []
        psm_dict = {next(iter(psm.provenance_data)): psm for psm in self._idparquet_reader.psms}
        psms_df = self._idparquet_reader.psms_df.drop(columns=["mods", "mod_sites", "nce", "instrument"])
        added_features: Set[str] = set()

        for _, record in psms_df.iterrows():
            record = record.to_dict()
            psm_features = psm_dict.get(record["provenance_data"])
            psm_metavalues = record["psm_metavalues"]

            if self._idparquet_reader.search_params["search_engine"] == "quantms-rescoring":
                if not self._idparquet_reader.get_meta_features(psm_metavalues, "MS:1002049"):
                    psm_metavalues = self.add_search_scores(psm_metavalues, "MS:1002049",
                                                            str(self._idparquet_reader.min_msgf_RawScore),
                                                            "int")

                if not self._idparquet_reader.get_meta_features(psm_metavalues, "MS:1002052"):
                    psm_metavalues = self.add_search_scores(psm_metavalues, "MS:1002052",
                                                            str(self._idparquet_reader.max_msgf_EValue),
                                                            "double")
                if np.isinf(record["score"]):
                    record["score"] = self._idparquet_reader.max_comet_expectation_value
                    psm_metavalues = self.add_search_scores(psm_metavalues, "MS:1002257",
                                                            str(record["score"]),
                                                            "double")
                if not self._idparquet_reader.get_meta_features(psm_metavalues, "MS:1002252"):
                    psm_metavalues = self.add_search_scores(psm_metavalues, "MS:1002252",
                                                            str(self._idparquet_reader.min_comet_xcorr),
                                                            "double")

            psm_metavalues, added_features = self.add_rescoring_features(psm_metavalues, psm_features, added_features)
            record["psm_metavalues"] = psm_metavalues
            record.pop("provenance_data", None)
            records.append(record)

        if self._idparquet_reader.search_params["search_engine"] == "quantms-rescoring":
            main_scores_features = {"MS:1002049,MS:1002052,MS:1002252,MS:1002257"}
            all_features = main_scores_features.union(added_features)
        else:
            # Update search parameters with added features
            try:
                features_existing = self._idparquet_reader.get_meta_features(
                    self._idparquet_reader.search_params["sp_metavalues"],
                    "extra_features"
                )
                if features_existing:
                    existing_set = set(features_existing.split(","))
                else:
                    existing_set = set()
            except (KeyError, AttributeError, RuntimeError) as e:
                logger.debug(f"No existing extra_features found: {e}")
                existing_set = set()

            # Combine existing and new features
            all_features = existing_set.union(added_features)

        found = False
        for mv in self._idparquet_reader.search_params["sp_metavalues"]:
            if mv["name"] == "extra_features":
                mv["value"] = ",".join(sorted(all_features))
                found = True
                break

        if not found:
            self._idparquet_reader.search_params["sp_metavalues"].append({
                "name": "extra_features",
                "value": ",".join(sorted(all_features)),
                "value_type": "string"
            })

        self._idparquet_psm = pa.Table.from_pylist(records, schema=self._idparquet_reader.psm_schema)
        self._idparquet_search_param = pa.Table.from_pylist([self._idparquet_reader.search_params],
                                                            schema=self._idparquet_reader.search_params_schema)
        self._idparquet_proteins = pa.Table.from_pandas(
            self._idparquet_reader.proteins_df,
            schema=self._idparquet_reader.proteins_schema,
            preserve_index=False
        )
        self._idparquet_protein_groups = pa.Table.from_pylist(
            self._idparquet_reader.protein_groups,
            schema=self._idparquet_reader.protein_groups_schema
        )

    @staticmethod
    def add_search_scores(psm_metavalues, name, value, value_type):
        """Add key-value pairs to the metavalue."""
        psm_metavalues.append({
            "name": name,
            "value": value,
            "value_type": value_type
        })
        return psm_metavalues

    def add_rescoring_features(self, psm_metavalues, psm_features, added_features):
        """Add rescoring features to the PSM metavalue."""
        for feature, value in psm_features.rescoring_features.items():
            if isinstance(value, int):
                value_type = "int"
            elif isinstance(value, float):
                value_type = "double"
            else:
                value_type = "string"

            canonical_feature = OpenMSHelper.get_canonical_feature(feature)
            if canonical_feature is not None and self.ms2_generator == "AlphaPeptDeep":
                canonical_feature = canonical_feature.replace("MS2PIP", "AlphaPeptDeep")

            if canonical_feature is not None:
                if (
                        self._only_features
                        and canonical_feature not in self._only_features
                ):
                    continue
                added_features.add(canonical_feature)

            psm_metavalues.append({
                "name": canonical_feature,
                "value": str(value),
                "value_type": value_type
            })
        return psm_metavalues, added_features

    def _update_search_parameters(self, features: Set[str]) -> None:
        """
        Update search parameters with new features.

        Parameters
        ----------
        features : Set[str]
            Set of feature names to add to search parameters.
        """
        if not features:
            return

        logger.info(f"Adding features to search parameters: {', '.join(sorted(features))}")

        # Get search parameters
        search_parameters = self._idparquet_reader.oms_proteins[0].getSearchParameters()

        # Get existing features
        try:
            features_existing = search_parameters.getMetaValue("extra_features")
            if features_existing:
                existing_set = set(features_existing.split(","))
            else:
                existing_set = set()
        except (KeyError, AttributeError, RuntimeError) as e:
            logger.debug(f"No existing extra_features found: {e}")
            existing_set = set()

        # Combine existing and new features
        all_features = existing_set.union(features)

        # Update search parameters
        search_parameters.setMetaValue("extra_features", ",".join(sorted(all_features)))
        self._idparquet_reader.oms_proteins[0].setSearchParameters(search_parameters)

    def _get_top_batch_psms(self, psm_list: PSMList) -> PSMList:
        """
        Get top-scoring non-decoy PSMs for calibration.

        Parameters
        ----------
        psm_list : PSMList
            List of PSMs to filter.

        Returns
        -------
        PSMList
            Filtered list containing top-scoring PSMs.
        """
        logger.info("Selecting top PSMs for calibration")

        # Filter non-decoy PSMs
        non_decoy_psms = [result for result in psm_list.psm_list if not result.is_decoy]

        if not non_decoy_psms:
            logger.warning("No non-decoy PSMs found for calibration")
            return PSMList(psm_list=[])

        # Sort by score
        non_decoy_psms.sort(key=lambda x: x.score, reverse=self._higher_score_better)

        # Select top 60% for calibration
        calibration_size = max(1, int(len(non_decoy_psms) * 0.6))
        calibration_psms = non_decoy_psms[:calibration_size]

        return PSMList(psm_list=calibration_psms)

    def _get_highest_fragmentation(self) -> Optional[str]:
        """
        Determine the predominant fragmentation method in the dataset.

        Returns
        -------
        Optional[str]
            "HCD", "CID", or None if not determined.
        """
        stats = self._idparquet_reader.stats
        if not stats or not stats.ms_level_dissociation_method:
            logger.warning("No fragmentation method statistics available")
            return None

        # Find the most common fragmentation method
        most_common = max(
            stats.ms_level_dissociation_method, key=stats.ms_level_dissociation_method.get
        )

        # Return "HCD" or "CID" if applicable
        if most_common[1] in ["HCD", "CID"]:
            return most_common[1]

        return None

    def _get_mae_from_psm_list(self, psm_list: PSMList) -> float:
        """
        Calculate Mean Absolute Error of retention time prediction.

        Parameters
        ----------
        psm_list : PSMList
            List of PSMs with retention time predictions.

        Returns
        -------
        float
            Mean Absolute Error (MAE) value or infinity if calculation fails.
        """
        best_scored_psms = self._get_top_batch_psms(psm_list)

        if not best_scored_psms.psm_list:
            logger.warning("No PSMs available for MAE calculation")
            return float("inf")

        total_error = 0.0
        count = 0

        for psm in best_scored_psms.psm_list:
            if "rt_diff" in psm.rescoring_features:
                total_error += abs(psm.rescoring_features["rt_diff"])
                count += 1

        if count == 0:
            logger.warning("No valid retention time differences for MAE calculation")
            return float("inf")

        return total_error / count
