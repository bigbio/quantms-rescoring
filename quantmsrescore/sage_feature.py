import click
import pandas as pd
from pathlib import Path
import shutil
from quantmsrescore.logging_config import get_logger, configure_logging
from quantmsrescore.utils import ParquetReader
import pyarrow as pa
import pyarrow.parquet as pq

configure_logging()
logger = get_logger(__name__)


@click.command("sage2feature")
@click.option("--idparquet", "-i", required=True, help="Input idparquet folder")
@click.option("--output_dir", "-o", required=True, help="Output idparquet folder")
@click.option("--feat_file", "-f", required=True, help="Feature file from Sage")
def add_sage_feature(idparquet: str, output_dir: str, feat_file: str):
    """
    Add extra features into Parquet search_params (Sage compatible).
    """

    idparquet = Path(idparquet)
    output_dir = Path(output_dir)

    logger.info(f"Reading feature file: {feat_file}")

    feat = pd.read_csv(feat_file, sep="\t")

    extra_feat = [
        row["feature_name"]
        for _, row in feat.iterrows()
        if row["feature_generator"] != "psm_file"
    ]

    logger.info(f"Extracted {len(extra_feat)} extra features")

    search_params_file = idparquet / "search_params.parquet"

    if not search_params_file.exists():
        raise click.ClickException("search_params.parquet not found")

    search_df = pd.read_parquet(search_params_file)

    if search_df.empty:
        raise click.ClickException("search_params.parquet is empty")

    # assume single-row search params
    search_params = search_df.iloc[0].to_dict()

    # Update search parameters with added features
    try:
        features_existing = search_params["sp_metavalues"]["extra_features"]
        if features_existing:
            existing_set = set(features_existing.split(","))
        else:
            existing_set = set()
    except (KeyError, AttributeError, RuntimeError) as e:
        logger.debug(f"No existing extra_features found: {e}")
        existing_set = set()

    # Combine existing and new features
    all_features = existing_set.union(set(extra_feat))
    found = False
    for mv in search_params["sp_metavalues"]:
        if mv["name"] == "extra_features":
            mv["value"] = ",".join(sorted(all_features))
            found = True
            break

    if not found:
        search_params["sp_metavalues"].append({
            "name": "extra_features",
            "value": ",".join(sorted(all_features)),
            "value_type": "string"
        })

    logger.info(f"Updated extra_features: {len(extra_feat)} features")

    # write back
    output_dir.mkdir(parents=True, exist_ok=True)

    # copy original idparquet
    if output_dir.exists():
        shutil.rmtree(output_dir)

    shutil.copytree(idparquet, output_dir)

    # overwrite updated search_params
    idparquet_reader = ParquetReader(output_dir)
    idparquet_search_param = pa.Table.from_pylist(
        [search_params],
        schema=idparquet_reader.search_params_schema
    )

    pq.write_table(
        idparquet_search_param,
        output_dir / "search_params.parquet"
    )

    logger.info(f"Saved updated search_params to {output_dir}")
    logger.info("Done")