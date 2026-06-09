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
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from typing import Set
from pathlib import Path
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

    if len(idparquet) > 1:
        # merge score
        main_scores_features: Set[str] = set()
        records = []

        for _, record in psms_df.iterrows():
            record = record.to_dict()
            psm_metavalues = record["psm_metavalues"]
            record, psm_metavalues, main_scores_features = fill_search_scores(idparquet_reader,
                                                                              record,
                                                                              psm_metavalues)
            record["psm_metavalues"] = psm_metavalues
            record.pop("provenance_data", None)
            records.append(record)

        found = False
        for mv in idparquet_reader.search_params["sp_metavalues"]:
            if mv["name"] == "extra_features":
                mv["value"] = ",".join(sorted(main_scores_features))
                found = True
                break

        if not found:
            idparquet_reader.search_params["sp_metavalues"].append({
                "name": "extra_features",
                "value": ",".join(sorted(main_scores_features)),
                "value_type": "string"
            })
        idparquet_psm = pa.Table.from_pylist(records, schema=idparquet_reader.psm_schema)
    else:
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


def fill_search_scores(idparquet_reader, record, psm_metavalues):
    main_scores_features = set()
    if len(set(idparquet_reader.merge_search_engines)) > 1:
        if "Comet" in idparquet_reader.merge_search_engines:
            main_scores_features = main_scores_features.union({"MS:1002252", "MS:1002257"})
            if "MS-GF+" in idparquet_reader.merge_search_engines:
                main_scores_features = main_scores_features.union({"MS:1002049", "MS:1002052"})
                if not idparquet_reader.get_meta_features(psm_metavalues, "MS:1002049"):
                    psm_metavalues = add_search_scores(psm_metavalues, "MS:1002049",
                                                            str(idparquet_reader.min_msgf_RawScore),
                                                            "int")

                if not idparquet_reader.get_meta_features(psm_metavalues, "MS:1002052"):
                    psm_metavalues = add_search_scores(psm_metavalues, "MS:1002052",
                                                            str(idparquet_reader.max_msgf_EValue),
                                                            "double")
            if "Sage" in idparquet_reader.merge_search_engines:
                main_scores_features.add("ln(hyperscore)")
                if not idparquet_reader.get_meta_features(psm_metavalues, "ln(hyperscore)"):
                    psm_metavalues = add_search_scores(psm_metavalues, "ln(hyperscore)",
                                                            str(idparquet_reader.min_sage_hyperscore),
                                                            "double")
            if np.isinf(record["score"]):
                record["score"] =idparquet_reader.max_comet_expectation_value
                psm_metavalues = add_search_scores(psm_metavalues, "MS:1002257",
                                                        str(record["score"]),
                                                        "double")
            if not idparquet_reader.get_meta_features(psm_metavalues, "MS:1002252"):
                psm_metavalues = add_search_scores(psm_metavalues, "MS:1002252",
                                                        str(idparquet_reader.min_comet_xcorr),
                                                        "double")
        else:
            main_scores_features = {"MS:1002049", "MS:1002052", "ln(hyperscore)"}
            if np.isinf(record["score"]):
                record["score"] = idparquet_reader.max_msgf_EValue
                psm_metavalues = add_search_scores(psm_metavalues, "MS:1002052",
                                                        str(record["score"]),
                                                        "double")
            if not idparquet_reader.get_meta_features(psm_metavalues, "MS:1002049"):
                psm_metavalues = add_search_scores(psm_metavalues, "MS:1002049",
                                                        str(idparquet_reader.min_msgf_RawScore),
                                                        "double")
            if not idparquet_reader.get_meta_features(psm_metavalues, "ln(hyperscore)"):
                psm_metavalues = add_search_scores(psm_metavalues, "ln(hyperscore)",
                                                        str(idparquet_reader.min_sage_hyperscore),
                                                        "double")
    return record, psm_metavalues, main_scores_features


def add_search_scores(psm_metavalues, name, value, value_type):
    """Add key-value pairs to the metavalue."""
    psm_metavalues.append({
        "name": name,
        "value": value,
        "value_type": value_type
    })
    return psm_metavalues