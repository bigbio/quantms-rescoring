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
from quantmsrescore.idparquet_reader import ParquetRescoringReader

configure_logging()


@click.command(
    "psm_feature_clean",
    short_help="Clean PSMs in parquet using spectrum-based filtering.",
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
    idparquet_reader = ParquetRescoringReader(idparquet,
                                              mzml,
                                              only_ms2=True,
                                              remove_missing_spectrum=True,
                                              )
    # =========================
    # 5. Save parquet output
    # =========================
    # =========================
    # 5. output
    # =========================
    psms_df = idparquet_reader.psms_df.drop(columns=["mods", "mod_sites", "nce", "instrument"])

    idparquet_psm = pa.Table.from_pandas(psms_df, schema=idparquet_reader.psm_schema)
    idparquet_search_param = pa.Table.from_pylist([idparquet_reader.search_params],
                                                  schema=idparquet_reader.search_params_schema)
    idparquet_proteins = pa.Table.from_pandas(
        idparquet_reader.proteins_df,
        schema=idparquet_reader.proteins_schema,
        preserve_index=False
    )
    idparquet_protein_groups = pa.Table.from_pylist(
        idparquet_reader.protein_groups,
        schema=idparquet_reader.protein_groups_schema
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    psm_file = output_dir / "psms.parquet"
    search_param_file = output_dir / "search_params.parquet"
    proteins_file = output_dir / "proteins.parquet"
    protein_groups_file = output_dir / "protein_groups.parquet"

    try:
        out_path = Path(output)
        pq.write_table(idparquet_psm, psm_file)
        logger.info(f"psms.parquet file written to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write psms.parquet psm file: {str(e)}")
        raise

    # search_params.parquet
    try:
        pq.write_table(idparquet_search_param, search_param_file)
        logger.info(f"search_params.parquet written to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write search_params.parquet file: {str(e)}")
        raise

    # proteins.parquet
    try:
        pq.write_table(idparquet_proteins, proteins_file)
        logger.info(f"proteins.parquet written to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write proteins.parquet file: {str(e)}")
        raise

    # search_params.parquet
    try:
        pq.write_table(idparquet_protein_groups, protein_groups_file)
        logger.info(f"protein_groups.parquet written to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write protein_groups.parquet file: {str(e)}")
        raise