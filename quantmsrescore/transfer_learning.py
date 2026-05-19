import os.path

import click
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq
# Import thread configuration FIRST before other heavy imports
from quantmsrescore import configure_threading, configure_torch_threads
from quantmsrescore.idparquet_reader import ParquetRescoringReader
from quantmsrescore.logging_config import get_logger
from quantmsrescore.openms import get_compiled_regex
from quantmsrescore.alphapeptdeep import read_spectrum_file, _get_targets_df_for_psm
from alphabase.peptide.fragment import create_fragment_mz_dataframe
from quantmsrescore.ms2_model_manager import MS2ModelManager
import pandas as pd
import ms2pip.exceptions as exceptions

# Get logger for this module
logger = get_logger(__name__)


@click.command(
    "transfer_learning",
    short_help="Annotate PSMs in an idparquet file using ms2rescore features.",
)
@click.option(
    "-i",
    "--idparquet",
    help="Path to the idparquet containing the PSMs from OpenMS",
    required=True,
    type=click.Path(exists=True),
    multiple=True
)
@click.option(
    "-s",
    "--mzml",
    help="Path to the mzML file containing the spectra use for identification",
    required=True,
    type=click.Path(exists=True),
    multiple=True
)
@click.option(
    "-o",
    "--save_model_dir",
    required=True,
    help="Path for the retrained model",
)
@click.option(
    "--processes",
    help="Number of parallel processes (e.g., Nextflow's $task.cpus). "
         "Each process uses 1 internal thread to avoid HPC resource contention. "
         "Default: 4",
    type=int,
    default=4,
)
@click.option(
    "--ms2_model_dir",
    help="The path of AlphaPeptDeep model (default: `./`)",
    type=str,
    default="./",
)
@click.option(
    "--ms2_tolerance",
    help="Fragment mass tolerance (default: `0.05`)",
    type=float,
    default=0.05,
)
@click.option(
    "--ms2_tolerance_unit",
    help="Fragment mass tolerance unit (default: Da)",
    type=click.Choice(['Da', 'ppm'], case_sensitive=False),
    default="Da",
)
@click.option(
    "--calibration_set_size",
    help="Percentage of number of psms to use for calibration and retraining (default: `0.20)",
    default=0.20,
)
@click.option(
    "--spectrum_id_pattern",
    help="Pattern for spectrum identification",
    type=str,
    default="(.*)",
)
@click.option(
    "--consider_modloss",
    help="If modloss ions are considered in the ms2 model",
    is_flag=True,
)
@click.option("--transfer_learning_test_ratio",
              help="The ratio of test data for MS2 transfer learning",
              default=0.40)
@click.option("--epoch_to_train_ms2",
              help="Epochs to train AlphaPeptDeep MS2 model",
              type=int,
              default=20)
@click.option(
    "--force_transfer_learning",
    help="Forced save fine-tune model when it is not better than pretrained for test dataset",
    is_flag=True,
)
@click.option("--log_level", help="Logging level (default: `info`)", default="info")
@click.pass_context
def transfer_learning(
        ctx,
        idparquet,
        mzml,
        save_model_dir: str,
        processes,
        ms2_model_dir,
        ms2_tolerance,
        ms2_tolerance_unit,
        calibration_set_size,
        spectrum_id_pattern: str,
        consider_modloss,
        transfer_learning_test_ratio,
        epoch_to_train_ms2,
        force_transfer_learning,
        log_level,
):
    """
    Annotate PSMs in an idparquet file with additional features using specified models.

    This command-line interface (CLI) command processes a PSM file by adding
    annotations from the MS²PIP and DeepLC models, among others, while preserving
    existing information. It supports various options for specifying input and
    output paths, logging levels, and feature generation configurations.

    Parameters
    ----------

    ctx : click.Context
        The Click context object.
    idparquet : str
        Path to the idparquet file containing the PSMs.
    mzml : str
        Path to the mzML file containing the mass spectrometry deeplc_models.
    save_model_dir : str
        Path for the retrained model.
    processes : int
        The number of parallel processes available (e.g., Nextflow's $task.cpus).
        Each process uses 1 internal thread for HPC safety.
    ms2_tolerance : float
        The tolerance for MS²PIP annotation.
    ms2_tolerance_unit : str, optional
        Unit for the fragment mass tolerance, e.g. "Da" or "ppm".
    ms2_model_dir: str, optional
        Path for MS2 model
    calibration_set_size : float
        The percentage of PSMs to use for calibration and retraining.
    spectrum_id_pattern : str
        The regex pattern for spectrum IDs.
    consider_modloss: bool, optional
        If modloss ions are considered in the ms2 model. `modloss`
        ions are mostly useful for phospho MS2 prediction model.
        Defaults to True.
    transfer_learning_test_ratio: float, optional
        The ratio of test data for MS2 transfer learning.
        Defaults to 0.3.
    epoch_to_train_ms2: int, optional
        Number of epochs to train AlphaPeptDeep MS2 model. Defaults to 20.
    force_transfer_learning: bool, optional
        Forced save fine-tune model when it is not better than pretrained for test dataset.
        Defaults to False.
    log_level : str
        The logging level for the CLI command.
    """
    # Configure threading for HPC environments
    # Use 1 thread per process to avoid thread explosion with multiprocessing
    # This is critical for Nextflow/Slurm where $task.cpus defines total parallelism
    # disable_gpu=True prevents CUDA initialization errors on CPU-only nodes
    configure_threading(n_threads=1, verbose=True, disable_gpu=True)
    configure_torch_threads(n_threads=1)

    annotator = AlphaPeptdeepTrainer(
        ms2_model_path=ms2_model_dir,
        ms2_tolerance=ms2_tolerance,
        ms2_tolerance_unit=ms2_tolerance_unit,
        calibration_set_size=calibration_set_size,
        processes=processes,
        spectrum_id_pattern=spectrum_id_pattern,
        consider_modloss=consider_modloss,
        transfer_learning_test_ratio=transfer_learning_test_ratio,
        epoch_to_train_ms2=epoch_to_train_ms2,
        force_transfer_learning=force_transfer_learning,
        save_model_dir=save_model_dir,
        log_level=log_level.upper()
    )
    annotator.build_consensus_idparquet(idparquet, mzml)
    annotator.fine_tune()


class AlphaPeptdeepTrainer:
    def __init__(self,
                 ms2_model_path: str = "models", ms2_tolerance: float = 0.05,
                 ms2_tolerance_unit: str = "Da", calibration_set_size: float = 0.2,
                 processes: int = 2,
                 save_model_dir: str = None,
                 spectrum_id_pattern: str = "(.*)",
                 consider_modloss: bool = False,
                 transfer_learning_test_ratio: float = 0.3,
                 epoch_to_train_ms2: int = 20,
                 force_transfer_learning: bool = False,
                 log_level: str = "INFO"):
        self.spec_file = None
        self.psms_df = []
        self.high_score_better = None
        self._processes = processes
        self._spectrum_id_pattern = spectrum_id_pattern
        self._consider_modloss = consider_modloss
        self._calibration_set_size = calibration_set_size
        self._transfer_learning_test_ratio = transfer_learning_test_ratio
        self._epoch_to_train_ms2 = epoch_to_train_ms2
        self._force_transfer_learning = force_transfer_learning
        self._ms2_tolerance = ms2_tolerance
        self._ms2_tolerance_unit = ms2_tolerance_unit
        self._model_dir = ms2_model_path
        self._save_model_dir = save_model_dir

        # Set up logging
        from quantmsrescore.logging_config import configure_logging

        configure_logging(log_level)

    @staticmethod
    def _read_idparquet_file(parquet_dir, spectrum_paths):
        # Load the idparquet file and corresponding mzML file

        psms_file = Path(parquet_dir) / "psms.parquet"
        psms_df = pq.read_table(psms_file).to_pandas()
        runs = psms_df["reference_file_name"].unique()
        spectrum_path = None
        for mzml_file in spectrum_paths:
            if os.path.basename(mzml_file) == runs[0]:
                spectrum_path = mzml_file
                break

        if spectrum_path is None:
            raise FileNotFoundError(
                f"Missing mzML for idparquet: {parquet_dir}"
            )

        reader = ParquetRescoringReader(parquet_dir, spectrum_path)
        return reader.psms_df, reader.high_score_better

    def build_consensus_idparquet(self, idparquet_dir, mzml_files):
        logger.info(f"Loading Parquet data: {idparquet_dir}")

        try:
            with ProcessPoolExecutor(max_workers=self._processes) as executor:
                future_to_file = {executor.submit(self._read_idparquet_file, parquet_dir, mzml_files): parquet_dir for
                                  parquet_dir in idparquet_dir}
                for future in as_completed(future_to_file):
                    try:
                        content, high_score_better = future.result()
                        self.psms_df.append(content)

                        if self.high_score_better is None:
                            self.high_score_better = high_score_better
                        elif self.high_score_better != high_score_better:
                            logger.warning(
                                "Inconsistent high_score_better across files"
                            )
                    except Exception as e:
                        idparquet_file = future_to_file[future]
                        logger.error(f"Error processing file: {idparquet_file, e}")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

                self.psms_df = pd.concat(self.psms_df, ignore_index=True)
                self.psms_df.rename(columns={"precursor_charge": "charge"}, inplace=True)
                decoys = len(self.psms_df[self.psms_df["is_decoy"]])
                targets = len(self.psms_df) - decoys
                self.spec_file = mzml_files

            logger.info(
                f"Loaded {len(self.psms_df)} PSMs from {idparquet_dir}: {decoys} decoys and {targets} targets"
            )

        except Exception as e:
            logger.error(f"Failed to load input files: {str(e)}")
            raise

    def fine_tune(self):
        if self._consider_modloss:
            frag_types = ['b_z1', 'y_z1', 'b_z2', 'y_z2',
                          'b_modloss_z1', 'b_modloss_z2',
                          'y_modloss_z1', 'y_modloss_z2']
        else:
            frag_types = ['b_z1', 'y_z1', 'b_z2', 'y_z2']

        # Default hit rank=1
        calibration_set = self.psms_df[(~self.psms_df["is_decoy"])]
        calibration_set.sort_values(by="score", inplace=True, ascending=not self.high_score_better)
        precursor_df = calibration_set[:int(len(calibration_set) * self._calibration_set_size)]
        theoretical_mz_df = create_fragment_mz_dataframe(precursor_df, frag_types)
        precursor_df = precursor_df.set_index("provenance_data")

        # Get cached compiled regex for spectrum ID matching
        spectrum_id_regex = get_compiled_regex(self._spectrum_id_pattern)

        match_intensity_df = []
        current_index = 0
        # Process each spectrum
        for sf in self.spec_file:
            single_df = precursor_df[precursor_df["reference_file_name"] == os.path.basename(sf)]
            for spectrum in read_spectrum_file(Path(sf).as_posix()):

                # Match spectrum ID with provided regex
                match = spectrum_id_regex.search(spectrum.identifier)
                try:
                    spectrum_id = match[1]
                except (TypeError, IndexError):
                    raise exceptions.TitlePatternError(
                        f"Spectrum title pattern `{self._spectrum_id_pattern}` could not be matched to "
                        f"spectrum ID `{spectrum.identifier}`. "
                        " Are you sure that the regex contains a capturing group?"
                    )

                # print(spectrum_id)
                # Skip if no matching PSMs
                psm = single_df[single_df["spectrum_reference"] == spectrum_id]
                if psm.shape[0] <= 0:
                    continue
                # Process each PSM for this spectrum
                for row_index, row in psm.iterrows():
                    row = precursor_df.loc[row_index]
                    mz = theoretical_mz_df.iloc[row["frag_start_idx"]:row["frag_stop_idx"], ]
                    match_intensity = _get_targets_df_for_psm(
                        mz, spectrum, self._ms2_tolerance, self._ms2_tolerance_unit
                    )
                    fragment_len = match_intensity.shape[0]
                    precursor_df.loc[row_index, "match_start_idx"] = current_index
                    precursor_df.loc[row_index, "match_stop_idx"] = current_index + fragment_len
                    match_intensity_df.append(match_intensity)
                    current_index += fragment_len

        match_intensity_df = pd.concat(match_intensity_df, ignore_index=True)
        psm_num_to_train_ms2 = int(len(precursor_df) * (1 - self._transfer_learning_test_ratio))
        psm_num_to_test_ms2 = len(precursor_df) - psm_num_to_train_ms2

        precursor_df.drop(columns=["frag_start_idx", "frag_stop_idx"], inplace=True)
        precursor_df.rename(columns={
            "match_start_idx": "frag_start_idx",
            "match_stop_idx": "frag_stop_idx"
        }, inplace=True)
        precursor_df["frag_start_idx"] = precursor_df["frag_start_idx"].astype("int64")
        precursor_df["frag_stop_idx"] = precursor_df["frag_stop_idx"].astype("int64")

        model_mgr = MS2ModelManager(
            mask_modloss=not self._consider_modloss,
            device="cpu",
            model_dir=self._model_dir
        )

        model_mgr.ms2_fine_tuning(precursor_df,
                                  match_intensity_df,
                                  psm_num_to_train_ms2=psm_num_to_train_ms2,
                                  psm_num_to_test_ms2=psm_num_to_test_ms2,
                                  train_verbose=True,
                                  force_transfer_learning=self._force_transfer_learning,
                                  epoch_to_train_ms2=self._epoch_to_train_ms2)

        model_mgr.save_ms2_model(self._save_model_dir)
