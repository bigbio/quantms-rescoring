# Get logger for this module
from quantmsrescore.logging_config import get_logger

logger = get_logger(__name__)

from warnings import filterwarnings

filterwarnings(
    "ignore",
    message="OPENMS_DATA_PATH environment variable already exists",
    category=UserWarning,
    module="pyopenms",
)

import click

from quantmsrescore.logging_config import configure_logging
from quantmsrescore.idparquet_reader import ParquetReader

# logging（必须保留）
configure_logging()


@click.command(
    "psm_feature_clean",
    short_help="Clean PSMs in parquet using spectrum-based filtering.",
)
@click.option(
    "-i",
    "--idparquet",
    help="Path to parquet directory containing PSMs",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-s",
    "--mzml",
    help="Path to mzML file",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output",
    help="Output parquet file",
    required=True,
    type=click.Path(),
)
def psm_feature_clean(
    idparquet: str,
    mzml: str,
    output: str,
):
    """
    Clean PSMs from parquet input using:
    - spectrum existence check
    - MS2 filtering
    - invalid score removal
    - duplicate removal

    Also rebuild:
    - protein table
    - protein group table
    """

    logger.info("[START] PSM feature clean (parquet mode)")
    logger.info(f"Input: {idparquet}")
    logger.info(f"mzML: {mzml}")

    # =========================
    # 1. Load parquet reader
    # =========================
    reader = ParquetReader(idparquet)

    reader.build_spectrum_lookup(
        mzml,
        check_unix_compatibility=True
    )

    # =========================
    # 2. Load PSM table
    # =========================
    psms_file = reader.filename / "psms.parquet"

    reader._psms_df = reader._load_parquet(psms_file)

    if reader._psms_df.empty:
        logger.error("Empty PSM table")
        raise ValueError("No PSMs found")

    logger.info(f"Loaded PSMs: {len(reader._psms_df)}")

    # =========================
    # 3. Clean PSMs
    # =========================
    stats = reader.psm_clean(
        remove_missing_spectrum=True,
        only_ms2=True
    )

    # =========================
    # 4. Logging stats
    # =========================
    logger.info(
        f"Clean summary: "
        f"missing={stats.missing_spectra}, "
        f"empty={stats.empty_spectra}, "
        f"invalid={stats.invalid_score}, "
        f"duplicates={stats.duplicates_psm}"
    )

    logger.info(f"MS levels: {dict(stats.ms_level_counts)}")

    # =========================
    # 5. Save parquet output
    # =========================
    out_path = (
        output if output.endswith(".parquet")
        else f"{output}/psm_clean.parquet"
    )

    reader._psms_df.to_parquet(out_path, index=False)

    logger.info(f"[DONE] saved: {out_path}")