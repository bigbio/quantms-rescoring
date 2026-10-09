"""
Model downloader for quantms-rescoring.

This module provides functionality to download all required models for MS2PIP
and AlphaPeptDeep ahead of time for offline use.
"""

import shutil
from pathlib import Path
from typing import Optional

import click
import ms2pip
from quantmsrescore.logging_config import configure_logging, get_logger
from quantmsrescore.ms2_model_manager import MS2ModelManager

# Get logger for this module
logger = get_logger(__name__)


def download_ms2pip_models(model_dir: Optional[Path] = None) -> None:
    """Download and validate the models declared by the installed MS2PIP version."""
    model_dir = Path(model_dir or Path.home() / ".ms2pip").expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    ms2pip.download_models(model_dir=model_dir)
    logger.info("MS2PIP models validated successfully.")


def download_alphapeptdeep_models(model_dir: Optional[Path] = None) -> None:
    """
    Download AlphaPeptDeep (peptdeep) models.

    This function downloads the pretrained models for AlphaPeptDeep/peptdeep
    for MS2 spectrum prediction, retention time prediction, and CCS prediction.

    Parameters
    ----------
    model_dir : Path, optional
        Target directory for models. If provided, models will be copied to
        this location after downloading.

    Raises
    ------
    ImportError
        If peptdeep package is not installed.
    Exception
        If model download fails.
    """
    try:
        logger.info("Downloading AlphaPeptDeep models...")

        # Download models to specified location or default
        target_dir = str(model_dir) if model_dir else "."
        MS2ModelManager(model_dir=target_dir)
        logger.info("AlphaPeptDeep models downloaded successfully.")

    except ImportError:
        logger.error("peptdeep package not found. Please install peptdeep")
        raise


@click.command(
    "download_models",
    short_help="Download all models for offline use (MS2PIP, AlphaPeptDeep).",
)
@click.option(
    "--model_dir",
    help="Directory to store downloaded models (optional, uses default cache if not specified)",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
)
@click.option(
    "--log_level",
    help="Logging level (default: `info`)",
    default="info",
)
@click.option(
    "--models",
    help="Comma-separated list of models to download: ms2pip, alphapeptdeep (default: ms2pip,alphapeptdeep)",
    default="ms2pip,alphapeptdeep",
)
def download_models(model_dir: Optional[str], log_level: str, models: str) -> None:
    """
    Download all required models for quantms-rescoring for offline use.

    This command downloads models for MS2PIP and AlphaPeptDeep
    to enable running quantms-rescoring in environments without internet access.

    Examples
    --------
    Download all models to default cache locations:

        $ rescoring download_models

    Download all models to a specific directory:

        $ rescoring download_models --model_dir /path/to/models

    Download only specific models:

        $ rescoring download_models --models alphapeptdeep

    Parameters
    ----------
    model_dir : str, optional
        Directory to store downloaded models. If not specified, models are
        downloaded to their default cache locations.
    log_level : str
        Logging level (default: "info").
    models : str
        Comma-separated list of models to download (default: "ms2pip,alphapeptdeep").
    """
    # Configure logging
    configure_logging(log_level.upper())

    # Validate model names
    VALID_MODELS = {"ms2pip", "alphapeptdeep"}

    # Convert model_dir to Path if provided
    target_dir = Path(model_dir) if model_dir else None

    # Parse and validate models list
    # Filter out empty strings from split result
    models_list = [m.strip().lower() for m in models.split(",") if m.strip()]
    invalid_models = [m for m in models_list if m not in VALID_MODELS]

    if invalid_models:
        error_msg = (
            f"Invalid model name(s): {', '.join(invalid_models)}. "
            f"Valid options are: {', '.join(sorted(VALID_MODELS))}"
        )
        logger.error(error_msg)
        raise click.BadParameter(error_msg)

    if not models_list:
        error_msg = "No models specified. Please provide at least one model to download."
        logger.error(error_msg)
        raise click.BadParameter(error_msg)

    logger.info("Starting model download process...")
    if target_dir:
        logger.info(f"Target directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Using default cache locations for each model type")

    # Download requested models
    success_count = 0
    failed_models = []

    if "ms2pip" in models_list:
        try:
            logger.info("\n=== Downloading MS2PIP models ===")
            download_ms2pip_models(target_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to download MS2PIP models: {e}")
            failed_models.append("ms2pip")

    if "alphapeptdeep" in models_list:
        try:
            logger.info("\n=== Downloading AlphaPeptDeep models ===")
            download_alphapeptdeep_models(target_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to download AlphaPeptDeep models: {e}")
            failed_models.append("alphapeptdeep")

    # Summary
    logger.info("\n=== Download Summary ===")
    logger.info(f"Successfully downloaded: {success_count}/{len(models_list)} model types")

    if failed_models:
        logger.error(f"Failed to download: {', '.join(failed_models)}")
        error_msg = (
            f"Failed to download some models: {', '.join(failed_models)}.\n"
            "Troubleshooting tips:\n"
            "  - Check your internet connection\n"
            "  - Ensure required packages are installed (ms2pip, deeplc, peptdeep)\n"
            "  - Check the log messages above for specific error details"
        )
        raise click.ClickException(error_msg)
    else:
        logger.info("All requested models downloaded successfully!")
        logger.info("\nYou can now use quantms-rescoring in offline environments.")
        if target_dir:
            logger.info(f"Models are available in: {target_dir}")
