import click
import pandas as pd
from pathlib import Path

from quantmsrescore.logging_config import get_logger, configure_logging

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

    old_features = search_params.get("extra_features", "")

    if isinstance(old_features, str) and old_features:
        new_features = old_features.split(",") + extra_feat
    else:
        new_features = extra_feat

    # deduplicate
    new_features = list(dict.fromkeys(new_features))

    search_params["extra_features"] = ",".join(new_features)

    logger.info(f"Updated extra_features: {len(new_features)} features")

    # write back
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([search_params]).to_parquet(
        output_dir / "search_params.parquet",
        index=False
    )

    logger.info(f"Saved updated search_params to {output_dir}")
    logger.info("Done")