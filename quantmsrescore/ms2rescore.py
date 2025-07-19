# Written by Jonas Scheid under the MIT license
# Contributions by Yasset Perez-Riverol and Dai Chengxin
# This script is part of the quantmsutils package

import click

from quantmsrescore.annotator import FeatureAnnotator
from quantmsrescore.logging_config import configure_logging

# Configure logging with default settings
configure_logging()


@click.command(
    "msrescore2feature",
    short_help="Annotate PSMs in an idXML file using ms2rescore features.",
)
@click.option(
    "-i",
    "--idxml",
    help="Path to the idxml containing the PSMs from OpenMS",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-s",
    "--mzml",
    help="Path to the mzML file containing the spectra use for identification",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output",
    help="Path the output idxml for the annotated PSMs",
)
@click.option("--log_level", help="Logging level (default: `info`)", default="info")
@click.option(
    "--processes",
    help="Number of parallel processes available to MS²Rescore (default: 4)",
    type=int,
    default=4,
)
@click.option(
    "--feature_generators",
    help="Comma-separated list of feature generators to use (default: `ms2pip,deeplc`). See rescoring doc for further information",
    default="ms2pip,deeplc",
)
@click.option(
    "--force_model",
    help="Force to run with provided MS2PIP model. Don't look for the best model and validation. Default False",
    is_flag=True,
)
@click.option(
    "--find_best_model",
    help="Find the best model with the best performance. Default True",
    is_flag=True,
)
@click.option(
    "--only_features",
    help="Comma-separated list of features to use for annotation (read docs for default)",
)
@click.option(
    "--ms2_model",
    help="MS²PIP model (default: `HCD2021`)",
    type=str,
    default="HCD2021",
)
@click.option(
    "--ms2_model_dir",
    help="The path of MS²PIP model (default: `./`)",
    type=str,
    default="./",
)
@click.option(
    "--ms2_tolerance",
    help="Fragment mass tolerance [Da](default: `0.05`)",
    type=float,
    default=0.05,
)
@click.option(
    "--calibration_set_size",
    help="Percentage of number of psms to use for calibration and retraining (default: `0.20)",
    default=0.20,
)
@click.option(
    "--valid_correlations_size",
    help="Percentage of number of psms with correlation above the threshold (default: `0.70)",
    default=0.70,
)
@click.option(
    "--skip_deeplc_retrain",
    help="Skip retraining of DeepLC model (default: `False`)",
    is_flag=True,
)
@click.option(
    "--spectrum_id_pattern",
    help="Pattern for spectrum identification",
    type=str,
    default="(.*)",
)
@click.option(
    "--psm_id_pattern",
    help="Pattern for PSM identification",
    type=str,
    default="(.*)",
)
@click.option(
    "--mask_modloss",
    help="If modloss ions are masked to zeros in the ms2 model",
    is_flag=True,
)
@click.pass_context
def msrescore2feature(
    ctx,
    idxml: str,
    mzml,
    output: str,
    log_level,
    processes,
    feature_generators,
    only_features,
    ms2_model_dir,
    ms2_model,
    force_model,
    find_best_model,
    ms2_tolerance,
    calibration_set_size,
    valid_correlations_size,
    skip_deeplc_retrain,
    spectrum_id_pattern: str,
    psm_id_pattern: str,
    mask_modloss
):
    """
    Annotate PSMs in an idXML file with additional features using specified models.

    This command-line interface (CLI) command processes a PSM file by adding
    annotations from the MS²PIP and DeepLC models, among others, while preserving
    existing information. It supports various options for specifying input and
    output paths, logging levels, and feature generation configurations.

    Parameters
    ----------

    ctx : click.Context
        The Click context object.
    idxml : str
        Path to the idXML file containing the PSMs.
    mzml : str
        Path to the mzML file containing the mass spectrometry deeplc_models.
    output : str
        Path to the output idXML file with annotated PSMs.
    log_level : str
        The logging level for the CLI command.
    processes : int
        The number of parallel processes available for MS²Rescore.
    feature_generators : str
        Comma-separated list of feature generators to use for annotation.
    only_features : str
        Comma-separated list of features to use for annotation.
    ms2pip_model_dir : str
        Path to the directory containing the MS²PIP models.
    ms2pip_model : str
        The MS²PIP model to use for annotation.
    force_model : bool
        Whether to force the use of the provided MS²PIP model.
    find_best_model : bool
        Whether to find the model with the best performance.
    ms2_tolerance : float
        The tolerance for MS²PIP annotation.
    calibration_set_size : float
        The percentage of PSMs to use for calibration and retraining.
    valid_correlations_size: float
        Fraction of the valid PSM.
    skip_deeplc_retrain : bool
        Whether to skip retraining the DeepLC model.
    spectrum_id_pattern : str
        The regex pattern for spectrum IDs.
    psm_id_pattern : str
        The regex pattern for PSM IDs.
    mask_modloss: bool, optional
        If modloss ions are masked to zeros in the ms2 model. `modloss`
        ions are mostly useful for phospho MS2 prediciton model.
        Defaults to True.
    """

    annotator = FeatureAnnotator(
        feature_generators=feature_generators,
        only_features=only_features,
        ms2_model=ms2_model,
        force_model=force_model,
        find_best_model=find_best_model,
        ms2_model_path=ms2_model_dir,
        ms2_tolerance=ms2_tolerance,
        calibration_set_size=calibration_set_size,
        valid_correlations_size=valid_correlations_size,
        skip_deeplc_retrain=skip_deeplc_retrain,
        processes=processes,
        log_level=log_level.upper(),
        spectrum_id_pattern=spectrum_id_pattern,
        psm_id_pattern=psm_id_pattern,
        mask_modloss=mask_modloss
    )
    annotator.build_idxml_data(idxml, mzml)
    annotator.annotate()

    if output:
        annotator.write_idxml_file(output)