"""
Model downloader for quantms-rescoring.

This module provides functionality to download all required models for MS2PIP
and AlphaPeptDeep ahead of time for offline use.
"""

import shutil
from pathlib import Path
from typing import Optional

import click
from ms2pip._utils.xgb_models import validate_requested_xgb_model
from quantmsrescore.logging_config import configure_logging, get_logger
from quantmsrescore import exceptions
from quantmsrescore.ms2_model_manager import MS2ModelManager

# Get logger for this module
logger = get_logger(__name__)

MODELS = {
    "CID": {
        "id": 0,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_CID_train_B.xgboost",
            "y": "model_20190107_CID_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_CID_train_B.xgboost": "4398c6ebe23e2f37c0aca42b095053ecea6fb427",
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f",
        },
    },
    "HCD2019": {
        "id": 1,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_HCD_train_B.xgboost",
            "y": "model_20190107_HCD_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_HCD_train_B.xgboost": "2503856c382806672e4b85f6b0ccc1f3093acc1b",
            "model_20190107_HCD_train_Y.xgboost": "867bbc9940f75845b3f4f845d429b3780c997a02",
        },
    },
    "TTOF5600": {
        "id": 2,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_TTOF5600_train_B.xgboost",
            "y": "model_20190107_TTOF5600_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_TTOF5600_train_B.xgboost": "ab2e28dfbc4ee60640253b0b4c127fc272c9d0ed",
            "model_20190107_TTOF5600_train_Y.xgboost": "f8e9ddd8ca78ace06f67460a2fea0d8fa2623452",
        },
    },
    "TMT": {
        "id": 3,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_TMT_train_B.xgboost",
            "y": "model_20190107_TMT_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_TMT_train_B.xgboost": "352073a591d45a2e3181818f5feef99c22755af7",
            "model_20190107_TMT_train_Y.xgboost": "d9a73bff21ab504bb91eb386f20cd8a86d60c95d",
        },
    },
    "iTRAQ": {
        "id": 4,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_iTRAQ_train_B.xgboost",
            "y": "model_20190107_iTRAQ_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_iTRAQ_train_B.xgboost": "b8d94ad329a245210c652a5b35d724d2c74d0d50",
            "model_20190107_iTRAQ_train_Y.xgboost": "56ae87d56fd434b53fcc1d291745cabb7baf463a",
        },
    },
    "iTRAQphospho": {
        "id": 5,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_iTRAQphospho_train_B.xgboost",
            "y": "model_20190107_iTRAQphospho_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_iTRAQphospho_train_B.xgboost": "e283b158cc50e219f42f93be624d0d0ac01d6b49",
            "model_20190107_iTRAQphospho_train_Y.xgboost": "261b2e1810a299ed7ebf193ce1fb81a608c07d3b",
        },
    },
    # ETD': {'id': 6, 'ion_types': ['B', 'Y', 'C', 'Z'], 'peaks_version': 'etd', 'features_version': 'normal'},
    "HCDch2": {
        "id": 7,
        "ion_types": ["B", "Y", "B2", "Y2"],
        "peaks_version": "ch2",
        "features_version": "normal",
    },
    "CIDch2": {
        "id": 8,
        "ion_types": ["B", "Y", "B2", "Y2"],
        "peaks_version": "ch2",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_CID_train_B.xgboost",
            "y": "model_20190107_CID_train_Y.xgboost",
            "b2": "model_20190107_CID_train_B2.xgboost",
            "y2": "model_20190107_CID_train_Y2.xgboost",
        },
        "model_hash": {
            "model_20190107_CID_train_B.xgboost": "4398c6ebe23e2f37c0aca42b095053ecea6fb427",
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f",
            "model_20190107_CID_train_B2.xgboost": "602f2fc648890aebbbe2646252ade658af3221a3",
            "model_20190107_CID_train_Y2.xgboost": "4e4ad0f1d4606c17015aae0f74edba69f684d399",
        },
    },
    "HCD2021": {
        "id": 9,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20210416_HCD2021_B.xgboost",
            "y": "model_20210416_HCD2021_Y.xgboost",
        },
        "model_hash": {
            "model_20210416_HCD2021_B.xgboost": "c086c599f618b199bbb36e2411701fb2866b24c8",
            "model_20210416_HCD2021_Y.xgboost": "22a5a137e29e69fa6d4320ed7d701b61cbdc4fcf",
        },
    },
    "Immuno-HCD": {
        "id": 10,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20210316_Immuno_HCD_B.xgboost",
            "y": "model_20210316_Immuno_HCD_Y.xgboost",
        },
        "model_hash": {
            "model_20210316_Immuno_HCD_B.xgboost": "977466d378de2e89c6ae15b4de8f07800d17a7b7",
            "model_20210316_Immuno_HCD_Y.xgboost": "71948e1b9d6c69cb69b9baf84d361a9f80986fea",
        },
    },
    "CID-TMT": {
        "id": 11,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20220104_CID_TMT_B.xgboost",
            "y": "model_20220104_CID_TMT_Y.xgboost",
        },
        "model_hash": {
            "model_20220104_CID_TMT_B.xgboost": "fa834162761a6ae444bb6523c9c1a174b9738389",
            "model_20220104_CID_TMT_Y.xgboost": "299539179ca55d4ac82e9aed6a4e0bd134a9a41e",
        },
    },
    "timsTOF2023": {
        "id": 12,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20230912_timsTOF_B.xgboost",
            "y": "model_20230912_timsTOF_Y.xgboost",
        },
        "model_hash": {
            "model_20230912_timsTOF_B.xgboost": "6beb557052455310d8c66311c86866dda8291f4b",
            "model_20230912_timsTOF_Y.xgboost": "8edd87e0fba5f338d0a0881b5afbcf2f48ec5268",
        },
    },
    "timsTOF2024": {
        "id": 13,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20240105_timsTOF_B.xgboost",
            "y": "model_20240105_timsTOF_Y.xgboost",
        },
        "model_hash": {
            "model_20240105_timsTOF_B.xgboost": "d70e145c15cf2bfa30968077a68409699b2fa541",
            "model_20240105_timsTOF_Y.xgboost": "3f0414ee1ad7cff739e0d6242e25bfc22b6ebfe5",
        },
    },
}


MODELS["HCD"] = MODELS["HCD2021"]
MODELS["timsTOF"] = MODELS["timsTOF2024"]


def download_ms2pip_models(model_dir: Optional[Path] = None) -> None:
    """
    Download MS2PIP models.

    MS2PIP models are bundled with the ms2pip package and don't require
    separate downloading. This function validates that the ms2pip package
    is properly installed.

    Parameters
    ----------
    model_dir : Path, optional
        Target directory for models (not used for MS2PIP as models are bundled).

    Raises
    ------
    ImportError
        If ms2pip package is not installed.
    """
    try:
        model_dir = Path(model_dir or Path.home() / ".ms2pip").expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Available MS2PIP models: {list(MODELS.keys())}")
        models = list(MODELS.keys())
        for model in models:
            if "xgb_model_files" not in MODELS[model]:
                raise exceptions.UnknownModelError(model)
            continue
            logger.debug("Downloading %s model files", model)
            validate_requested_xgb_model(
                MODELS[model]["xgboost_model_files"],
                MODELS[model]["model_hash"],
                model_dir,
            )
        logger.info("MS2PIP models validated successfully.")
    except ImportError:
        logger.error("MS2PIP package not found. Please install ms2pip>=4.0")
        raise


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
