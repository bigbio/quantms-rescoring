import re
import click
import numpy as np
import pandas as pd
from scipy.stats import entropy

from quantmsrescore.logging_config import get_logger, configure_logging
from quantmsrescore.openms import OpenMSHelper
from quantmsrescore.utils import ParquetReader

# init logging
configure_logging()
logger = get_logger(__name__)


# =========================
# Spectrum feature container
# =========================
class SpectrumMetrics:
    """
    Store computed spectrum-level features for one MS/MS spectrum.
    """

    def __init__(self, snr, spectral_entropy, fraction_tic_top_10, weighted_std_mz):
        self.snr = snr
        self.spectral_entropy = spectral_entropy
        self.fraction_tic_top_10 = fraction_tic_top_10
        self.weighted_std_mz = weighted_std_mz

    def as_dict(self):
        """
        Convert metrics into OpenMS MetaValue format.
        """
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


# =========================
# CLI entry
# =========================
@click.command("spectrum2feature")
@click.option(
    "--parquet",
    type=click.Path(exists=True),
    required=True,
    help="Input parquet directory containing PSMs",
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
def spectrum2feature(parquet, mzml, output):

    logger.info(f"[START] spectrum2feature")
    logger.info(f"Input parquet: {parquet}")
    logger.info(f"mzML file: {mzml}")

    # =========================
    # 1. Load Parquet pipeline
    # =========================
    reader = ParquetReader(parquet)

    # build spectrum lookup index (OpenMS wrapper)
    reader.build_spectrum_lookup(mzml)

    # load PSM table
    psms_df = reader._load_parquet(reader.filename / "psms.parquet")

    if psms_df.empty:
        logger.error("No PSMs found in parquet")
        raise ValueError("Empty PSM table")

    logger.info(f"Loaded {len(psms_df)} PSMs")

    result_rows = []

    # =========================
    # 2. iterate PSMs
    # =========================
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

        # =========================
        # 3. fetch spectrum
        # =========================
        spectrum_data = OpenMSHelper.get_peaks_by_scan(
            scan,
            reader.exp,
            reader.spec_lookup,
        )

        if spectrum_data is None:
            logger.debug(f"No spectrum found for scan {scan}")
            continue

        mz_array, intensity_array = spectrum_data

        # =========================
        # 4. compute features
        # =========================
        try:
            metrics = SpectrumAnalyzer.compute_spectrum_metrics(
                np.array(mz_array),
                np.array(intensity_array),
            )

            record = row.to_dict()
            record.update(metrics.as_dict())

            result_rows.append(record)

        except Exception as e:
            logger.error(f"Failed spectrum {scan}: {e}")
            continue

    # =========================
    # 5. output
    # =========================
    result_df = pd.DataFrame(result_rows)

    logger.info(f"Final valid spectra: {len(result_df)}")

    # save parquet
    out_file = (
        output if output.endswith(".parquet")
        else f"{output}/psms.parquet"
    )

    result_df.to_parquet(out_file, index=False)

    logger.info(f"[DONE] saved to {out_file}")