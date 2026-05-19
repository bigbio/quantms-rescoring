import re
import click
import numpy as np
from scipy.stats import entropy
from typing import Set
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from quantmsrescore.logging_config import get_logger, configure_logging
from quantmsrescore.openms import OpenMSHelper
from quantmsrescore.idparquet_reader import ParquetRescoringReader

# init logging
configure_logging()
logger = get_logger(__name__)


# =========================
# Spectrum feature container
# =========================
class SpectrumMetrics:
    """Store computed spectrum-level features for one MS/MS spectrum."""

    def __init__(self, snr, spectral_entropy, fraction_tic_top_10, weighted_std_mz):
        """
        Initialize spectrum-level metrics.

        Parameters
        ----------
        snr : float
            Signal-to-noise ratio.
        spectral_entropy : float
            Spectral entropy value.
        fraction_tic_top_10 : float
            Fraction of TIC explained by top 10 peaks.
        weighted_std_mz : float
            Weighted standard deviation of m/z values.
        """
        self.snr = snr
        self.spectral_entropy = spectral_entropy
        self.fraction_tic_top_10 = fraction_tic_top_10
        self.weighted_std_mz = weighted_std_mz

    def as_dict(self):
        """Convert metrics into OpenMS MetaValue format."""
        return {
            "Quantms:Snr": OpenMSHelper.get_str_metavalue_round(self.snr),
            "Quantms:SpectralEntropy": OpenMSHelper.get_str_metavalue_round(self.spectral_entropy),
            "Quantms:FracTICinTop10Peaks": OpenMSHelper.get_str_metavalue_round(
                self.fraction_tic_top_10
            ),
            "Quantms:WeightedStdMz": OpenMSHelper.get_str_metavalue_round(
                self.weighted_std_mz
            ),
        }


# =========================
# Spectrum analysis logic
# =========================
class SpectrumAnalyzer:

    @staticmethod
    def compute_signal_to_noise(intensities: np.ndarray) -> float:
        """
        Signal-to-noise ratio = max intensity / RMSD
        """
        if len(intensities) == 0:
            return 0.0

        rmsd = np.sqrt(np.mean(intensities ** 2))
        return 0.0 if rmsd == 0 else np.max(intensities) / rmsd

    @staticmethod
    def compute_spectrum_metrics(mz_array, intensity_array) -> SpectrumMetrics:
        """
        Compute full spectrum-level descriptors.
        """

        if len(mz_array) == 0 or len(intensity_array) == 0:
            raise ValueError("Empty spectrum")

        if len(mz_array) != len(intensity_array):
            raise ValueError("mz/intensity mismatch")

        tic = np.sum(intensity_array)
        if tic == 0:
            raise ValueError("TIC = 0")

        # normalize intensity
        norm = intensity_array / tic

        # 1. SNR
        snr = SpectrumAnalyzer.compute_signal_to_noise(intensity_array)

        # 2. entropy
        spectral_entropy = entropy(norm)

        # 3. top-10 TIC fraction
        top10 = np.sort(intensity_array)[-10:]
        frac_top10 = np.sum(top10) / tic

        # 4. weighted m/z variance
        wmz = np.sum(mz_array * norm)
        wstd = np.sqrt(np.sum(norm * (mz_array - wmz) ** 2))

        return SpectrumMetrics(snr, spectral_entropy, frac_top10, wstd)


def write_idparquet_file(idparquet_psm, idparquet_search_param, idparquet_proteins, idparquet_protein_groups, output):
    """Write annotated data to idparquet file."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    psm_file = output_dir / "psms.parquet"
    search_param_file = output_dir / "search_params.parquet"
    proteins_file = output_dir / "proteins.parquet"
    protein_groups_file = output_dir / "protein_groups.parquet"

    try:
        pq.write_table(idparquet_psm, psm_file)
        logger.info(f"psms.parquet file written to {psm_file}")
    except Exception as e:
        logger.error(f"Failed to write psms.parquet psm file: {str(e)}")
        raise

    # search_params.parquet
    try:
        pq.write_table(idparquet_search_param, search_param_file)
        logger.info(f"search_params.parquet written to {search_param_file}")
    except Exception as e:
        logger.error(f"Failed to write search_params.parquet file: {str(e)}")
        raise

    # proteins.parquet
    try:
        pq.write_table(idparquet_proteins, proteins_file)
        logger.info(f"proteins.parquet written to {proteins_file}")
    except Exception as e:
        logger.error(f"Failed to write proteins.parquet file: {str(e)}")
        raise

    # search_params.parquet
    try:
        pq.write_table(idparquet_protein_groups, protein_groups_file)
        logger.info(f"protein_groups.parquet written to {protein_groups_file}")
    except Exception as e:
        logger.error(f"Failed to write protein_groups.parquet file: {str(e)}")
        raise


def update_search_parameter(idparquet_reader, added_features):
    # Update search parameters with added features
    try:
        features_existing = idparquet_reader.get_meta_features(
            idparquet_reader.search_params["sp_metavalues"],
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
    for mv in idparquet_reader.search_params["sp_metavalues"]:
        if mv["name"] == "extra_features":
            mv["value"] = ",".join(sorted(all_features))
            found = True
            break

    if not found:
        idparquet_reader.search_params["sp_metavalues"].append({
            "name": "extra_features",
            "value": ",".join(sorted(all_features)),
            "value_type": "string"
        })
    return idparquet_reader

# =========================
# CLI entry
# =========================
@click.command("spectrum2feature")
@click.option(
    "-i",
    "--idparquet",
    help="Path to the idparquet containing the PSMs from OpenMS",
    required=True,
    type=click.Path(exists=True),
    multiple=True
)
@click.option(
    "--mzml",
    type=click.Path(exists=True),
    required=True,
    help="mzML file with spectra",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output parquet file",
)
def spectrum2feature(idparquet, mzml, output):
    logger.info("[START] spectrum2feature")
    logger.info(f"Input parquet: {idparquet}")
    logger.info(f"mzML file: {mzml}")

    idparquet_reader = ParquetRescoringReader(idparquet,
                                              mzml,
                                              only_ms2=True,
                                              remove_missing_spectrum=True,
                                              )
    psms_df = idparquet_reader.psms_df.drop(columns=["mods", "mod_sites", "nce", "instrument"])

    result_rows = []
    added_features: Set[str] = set()

    for idx, row in psms_df.iterrows():

        spectrum_reference = row.get("spectrum_reference", None)

        if spectrum_reference is None:
            logger.warning(f"Missing spectrum_reference at row {idx}")
            continue

        # parse scan id
        scan_match = re.findall(r"(spectrum|scan)=(\d+)", str(spectrum_reference))

        if not scan_match:
            logger.warning(f"Cannot parse scan: {spectrum_reference}")
            continue

        scan = int(scan_match[0][1])
        spectrum_data = OpenMSHelper.get_peaks_by_scan(
            scan,
            idparquet_reader.exp,
            idparquet_reader.spec_lookup,
        )

        if spectrum_data is None:
            logger.debug(f"No spectrum found for scan {scan}")
            continue

        mz_array, intensity_array = spectrum_data
        try:
            metrics = SpectrumAnalyzer.compute_spectrum_metrics(
                np.array(mz_array),
                np.array(intensity_array),
            )

            record = row.to_dict()
            psm_metavalues = record["psm_metavalues"]

            # attach snr features
            for feature, value in metrics.as_dict().items():
                psm_metavalues.append({
                    "name": feature,
                    "value": str(value),
                    "value_type": "string"
                })
                added_features.add(feature)

            record["psm_metavalues"] = psm_metavalues
            result_rows.append(record)

        except Exception as e:
            logger.error(f"Failed spectrum {scan}: {e}")

    idparquet_reader = update_search_parameter(idparquet_reader, added_features)

    idparquet_psm = pa.Table.from_pylist(result_rows, schema=idparquet_reader.psm_schema)
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
    write_idparquet_file(idparquet_psm, idparquet_search_param, idparquet_proteins, idparquet_protein_groups, output)
