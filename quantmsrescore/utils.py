# Get logger for this module
from quantmsrescore.logging_config import get_logger
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple, DefaultDict
from quantmsrescore.exceptions import MS3NotSupportedException
import pyopenms as oms
import pandas as pd
import pyarrow.parquet as pq
from quantmsrescore.openms import OpenMSHelper

logger = get_logger(__name__)


class SpectrumStats:
    """Statistics about spectrum analysis."""

    def __init__(self):
        self.missing_spectra: int = 0
        self.empty_spectra: int = 0
        self.invalid_score: int = 0
        self.duplicates_psm: int = 0
        self.ms_level_counts: DefaultDict[int, int] = defaultdict(int)
        self.ms_level_dissociation_method: Dict[Tuple[int, str], int] = {}


class ParquetReader:
    """
    A class to read and parse Parquet files for protein and peptide identifications.

    Attributes
    ----------
    filename : Path
        The path to the idXML file.
    oms_proteins : List[oms.ProteinIdentification]
        List of protein identifications parsed from the idXML file.
    oms_peptides : List[oms.PeptideIdentification]
        List of peptide identifications parsed from the idXML file.
    """

    def __init__(self, idparquet: Union[Path, str]) -> None:
        """
        Initialize IdXMLReader with the specified idXML file.

        Parameters
        ----------
        idxml_filename : Union[Path, str]
            Path to the idXML file to be read and parsed.
        """
        self.filename = Path(idparquet)
        self.spec_lookup = None
        self.exp = None

        # Private properties for spectrum lookup
        self._mzml_path = None
        self._stats = None  # IdXML stats

    def _load_parquet(self, parquet_file: Path) -> pd.DataFrame:
        """
        Load parquet file into pandas DataFrame.
        """
        if not parquet_file.exists():
            logger.warning(f"{parquet_file} not found")
            return pd.DataFrame()

        return pq.read_table(parquet_file).to_pandas()

    def _load_search_params(self, parquet_dir: Path) -> Dict:
        """
        Load search parameters.
        """
        search_params_file = parquet_dir / "search_params.parquet"
        if not search_params_file:
            return {}

        df = self._load_parquet(search_params_file)

        if df.empty:
            return {}

        if len(df) == 1:
            return df.iloc[0].to_dict()

        return df.to_dict(orient="records")

    @property
    def stats(self) -> Optional[SpectrumStats]:
        """Get spectrum statistics."""
        return self._stats

    @property
    def spectrum_path(self) -> Optional[Union[str, Path]]:
        """Get the path to the mzML file."""
        return self._mzml_path

    def build_spectrum_lookup(
            self, mzml_file: Union[str, Path], check_unix_compatibility: bool = False
    ) -> None:
        """
        Build a SpectrumLookup indexer from an mzML file.

        Parameters
        ----------
        mzml_file : Union[str, Path]
            The path to the mzML file to be processed.
        check_unix_compatibility : bool, optional
            Flag to check for Unix compatibility in the mzML file, by default, False.
        """
        self._mzml_path = str(mzml_file) if isinstance(mzml_file, Path) else mzml_file
        if check_unix_compatibility:
            OpenMSHelper.check_unix_compatibility(self._mzml_path)
        self.exp, self.spec_lookup = OpenMSHelper.get_spectrum_lookup_indexer(self._mzml_path)
        logger.info(f"Built SpectrumLookup from {self._mzml_path}")

    def psm_clean(
            self,
            remove_missing_spectrum: bool = True,
            only_ms2: bool = True
    ) -> SpectrumStats:

        if self.spec_lookup is None or self.exp is None:
            raise ValueError("Spectrum lookup not initialized")

        self._stats = SpectrumStats()

        valid_rows = []
        rebuilt_psms = []
        unique_spectrum_reference = set()

        search_engine = self.search_params.get("search_engine", "")

        for _, row in self._psms_df.iterrows():

            spectrum_reference = (
                    row.get("spectrum_ref")
                    or row.get("spectrum_reference")
                    or row.get("spectrum_id")
                    or row.get("scan")
            )

            if spectrum_reference is None:
                continue

            # duplicate check
            if spectrum_reference in unique_spectrum_reference:
                self._stats.duplicates_psm += 1
                continue

            unique_spectrum_reference.add(spectrum_reference)

            spectrum = OpenMSHelper.get_spectrum_for_psm(
                row, self.exp, self.spec_lookup
            )

            missing = False
            empty = False
            ms_level = 2

            if spectrum is None:
                self._stats.missing_spectra += 1
                missing = True
            else:
                peaks = spectrum.get_peaks()[0]

                if peaks is None or len(peaks) == 0:
                    self._stats.empty_spectra += 1
                    empty = True

                ms_level = spectrum.getMSLevel()
                self._stats.ms_level_counts[ms_level] += 1

            score = row.get("score")

            if score is None or pd.isna(score) or np.isinf(score):
                self._stats.invalid_score += 1
                invalid = True
            else:
                invalid = False

            # filtering
            if remove_missing_spectrum and (missing or empty or invalid):
                continue

            if only_ms2 and ms_level != 2:
                continue

            valid_rows.append(row)

        # update psms_df
        self._psms_df = pd.DataFrame(valid_rows)

        # rebuild PSMList
        for _, row in self._psms_df.iterrows():
            psm = self._parse_psm(row, self.search_params)
            if psm:
                rebuilt_psms.append(psm)

        self._psms = PSMList(rebuilt_psms)

        # update protein / protein group
        self.rebuild_proteins()
        self.rebuild_protein_groups()

        self._log_spectrum_statistics()

        return self._stats

    def _log_spectrum_statistics(self):
        """Log statistics about spectrum validation."""
        if self._stats.missing_spectra or self._stats.empty_spectra:
            logger.error(
                f"Found {self._stats.missing_spectra} PSMs with missing spectra and "
                f"{self._stats.empty_spectra} PSMs with empty spectra"
            )

        if len({k[1] for k in self._stats.ms_level_dissociation_method}) > 1:
            logger.error(
                "Found multiple dissociation methods in the same MS level. "
                "MS2pip models are not trained for multiple dissociation methods"
            )

        logger.info(f"MS level distribution: {dict(self._stats.ms_level_counts)}")
        logger.info(
            f"Dissociation Method Distribution: {self._stats.ms_level_dissociation_method}"
        )

    def _process_dissociation_methods(self, spectrum, ms_level):
        """Process dissociation methods from spectrum precursors."""
        oms_dissociation_matrix = OpenMSHelper.get_pyopenms_dissociation_matrix()
        for precursor in spectrum.getPrecursors():
            for method_index in precursor.getActivationMethods():
                if (oms_dissociation_matrix is not None) and (
                        0 <= method_index < len(oms_dissociation_matrix)
                ):
                    method = (
                        ms_level,
                        OpenMSHelper.get_dissociation_method(
                            method_index, oms_dissociation_matrix
                        ),
                    )
                    self._stats.ms_level_dissociation_method[method] = (
                            self._stats.ms_level_dissociation_method.get(method, 0) + 1
                    )
                else:
                    logger.warning(f"Unknown dissociation method index {method_index}")

    def rebuild_proteins(self):

        protein_hits = defaultdict(set)

        for _, row in self._psms_df.iterrows():

            proteins = row.get("proteins", [])
            peptide = row.get("peptidoform")

            if proteins is None:
                continue

            if isinstance(proteins, str):
                proteins = [proteins]

            for p in proteins:
                protein_hits[p].add(peptide)

        self._proteins_df = pd.DataFrame([
            {
                "accession": prot,
                "n_peptides": len(peps),
                "peptides": list(peps)
            }
            for prot, peps in protein_hits.items()
        ])

    def rebuild_protein_groups(self):

        groups = []
        group_index = 0

        for _, row in self._proteins_df.iterrows():
            groups.append({
                "group_index": group_index,
                "accessions": [row["accession"]],
                "n_proteins": 1,
                "n_peptides": row.get("n_peptides", 0)
            })

            group_index += 1

        self._protein_groups_df = pd.DataFrame(groups)