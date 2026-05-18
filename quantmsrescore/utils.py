# Get logger for this module
from quantmsrescore.logging_config import get_logger
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple, DefaultDict
import pyopenms as oms
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from quantmsrescore.openms import OpenMSHelper
from datetime import datetime, timezone
import uuid
import numpy as np
from psm_utils import PSM, PSMList

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

    def __init__(self, idparquet: Union[str, Path, List[Union[str, Path]]]) -> None:
        """
        Initialize IdXMLReader with the specified idXML file.

        Parameters
        ----------
        idxml_filename : Union[Path, str]
            Path to the idXML file to be read and parsed.
        """

        if isinstance(idparquet, (str, Path)):
            self.parquet_dirs = [Path(idparquet)]
        else:
            self.parquet_dirs = [Path(p) for p in idparquet]

        for p in self.parquet_dirs:
            if not p.exists():
                raise FileNotFoundError(f"{p} does not exist")

        self.spec_lookup = None
        self.exp = None
        self.psm_schema = None
        self.search_params_schema = None
        self.proteins_schema = None
        self.protein_groups_schema = None

        # Private properties for spectrum lookup
        self._mzml_path = None
        self._stats = None  # IdXML stats
        self.creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.meta_struct = pa.struct([
            pa.field("name", pa.string()),
            pa.field("value", pa.string()),
            pa.field("value_type", pa.string()),
        ])
        self._init_psm_schema()
        self._init_search_params_schema()
        self._init_proteins_schema()
        self._init_protein_groups_schema()

    def _init_psm_schema(self):
        # =========================================================
        # modification struct
        # =========================================================

        modification_struct = pa.struct([
            ("name", pa.string()),
            ("accession", pa.string()),
            ("positions", pa.list_(pa.struct([
                ("position", pa.string()),
                ("scores", pa.float64()),
            ]))),
        ])

        # =========================================================
        # additional score struct
        # =========================================================

        additional_score_struct = pa.struct([
            ("score_name", pa.string()),
            ("score_value", pa.float64()),
            ("higher_better", pa.bool_()),
        ])

        # =========================================================
        # protein accession struct
        # =========================================================

        protein_accession_struct = pa.struct([
            pa.field(
                "accession",
                pa.string(),
                nullable=False
            ),
            pa.field(
                "aa_before",
                pa.string()
            ),
            pa.field(
                "aa_after",
                pa.string()
            ),
            pa.field(
                "start",
                pa.int32()
            ),
            pa.field(
                "end",
                pa.int32()
            ),
        ])

        # =========================================================
        # PSM schema
        # =========================================================

        self.psm_schema = pa.schema([
            pa.field("sequence", pa.string()),
            pa.field("peptidoform", pa.string()),
            pa.field(
                "modifications",
                pa.list_(modification_struct)
            ),
            pa.field(
                "precursor_charge",
                pa.int32()
            ),
            pa.field(
                "posterior_error_probability",
                pa.float64()
            ),
            pa.field(
                "is_decoy",
                pa.bool_()
            ),
            pa.field(
                "calculated_mz",
                pa.float64()
            ),
            pa.field(
                "observed_mz",
                pa.float64()
            ),
            pa.field(
                "additional_scores",
                pa.list_(additional_score_struct)
            ),
            pa.field(
                "protein_accessions",
                pa.list_(protein_accession_struct)
            ),
            pa.field(
                "predicted_rt",
                pa.float64()
            ),
            pa.field(
                "reference_file_name",
                pa.string()
            ),
            pa.field(
                "cv_params",
                pa.string()
            ),
            pa.field(
                "scan",
                pa.int32()
            ),
            pa.field(
                "rt",
                pa.float64()
            ),
            pa.field(
                "ion_mobility",
                pa.float64()
            ),
            pa.field(
                "spectrum_reference",
                pa.string()
            ),
            pa.field(
                "score",
                pa.float64()
            ),
            pa.field(
                "score_type",
                pa.string()
            ),
            pa.field(
                "higher_score_better",
                pa.bool_()
            ),
            pa.field(
                "hit_index",
                pa.int32()
            ),
            pa.field(
                "peptide_identification_index",
                pa.int32()
            ),
            pa.field(
                "psm_metavalues",
                pa.list_(self.meta_struct)
            ),
            pa.field(
                "spectrum_metavalues",
                pa.list_(self.meta_struct)
            ),
            pa.field(
                "run_identifier",
                pa.string()
            ),
            pa.field(
                "mz_array",
                pa.list_(pa.float32())
            ),
            pa.field(
                "intensity_array",
                pa.list_(pa.float32())
            ),
            pa.field(
                "charge_array",
                pa.list_(pa.int32())
            ),
            pa.field(
                "ion_type_array",
                pa.list_(pa.string())
            ),

        ], metadata={
            b"software_provider": b"OpenMS",
            b"creation_date": self.creation_date.encode(),
            b"uuid": str(uuid.uuid4()).encode(),
            b"file_type": b"psms",
            b"creator": b"OpenMS",
            b"qpx_version": b"1.0",
        })

    def _init_search_params_schema(self):
        self.search_params_schema = pa.schema([
            pa.field("run_identifier", pa.string(), nullable=False),
            pa.field("search_engine", pa.string(), nullable=False),
            pa.field("search_engine_version", pa.string()),
            pa.field("inference_engine", pa.string()),
            pa.field("inference_engine_version", pa.string()),
            pa.field("date", pa.timestamp("ms")),
            pa.field("score_type", pa.string(), nullable=False),
            pa.field("higher_score_better", pa.bool_(), nullable=False),
            pa.field("significance_threshold", pa.float64()),
            pa.field("db", pa.string()),
            pa.field("db_version", pa.string()),
            pa.field("taxonomy", pa.string()),
            pa.field("charges", pa.string()),
            pa.field("mass_type", pa.string(), nullable=False),
            pa.field("precursor_mass_tolerance", pa.float64(), nullable=False),
            pa.field("precursor_mass_tolerance_ppm", pa.bool_(), nullable=False),
            pa.field("fragment_mass_tolerance", pa.float64(), nullable=False),
            pa.field("fragment_mass_tolerance_ppm", pa.bool_(), nullable=False),
            pa.field("digestion_enzyme", pa.string()),
            pa.field("enzyme_term_specificity", pa.string()),
            pa.field("missed_cleavages", pa.int32(), nullable=False),
            pa.field(
                "fixed_modifications",
                pa.list_(pa.string()),
                nullable=False
            ),
            pa.field(
                "variable_modifications",
                pa.list_(pa.string()),
                nullable=False
            ),
            pa.field(
                "primary_ms_run_paths",
                pa.list_(pa.string()),
                nullable=False
            ),
            pa.field(
                "metavalues",
                pa.list_(self.meta_struct),
                nullable=False
            ),
            pa.field(
                "sp_metavalues",
                pa.list_(self.meta_struct),
                nullable=False
            ),
        ], metadata={
            b"software_provider": b"OpenMS",
            b"creation_date": self.creation_date.encode("utf-8"),
            b"uuid": str(uuid.uuid4()).encode(),
            b"file_type": b"search_params",
            b"creator": b"OpenMS",
            b"qpx_version": b"1.0",
        })

    def _init_proteins_schema(self):
        # =========================================================
        # proteins schema
        # =========================================================

        modification_struct = pa.struct([
            ("position", pa.int32()),
            ("modification", pa.string()),
        ])

        self.proteins_schema = pa.schema([

            pa.field(
                "accession",
                pa.string(),
                nullable=False
            ),

            pa.field(
                "score",
                pa.float64(),
                nullable=False
            ),

            pa.field(
                "rank",
                pa.int32(),
                nullable=False
            ),

            pa.field(
                "coverage",
                pa.float64()
            ),

            pa.field(
                "sequence",
                pa.string()
            ),

            pa.field(
                "description",
                pa.string()
            ),

            pa.field(
                "is_decoy",
                pa.bool_()
            ),

            pa.field(
                "run_identifier",
                pa.string(),
                nullable=False
            ),

            pa.field(
                "modifications",
                pa.list_(modification_struct)
            ),

            pa.field(
                "metavalues",
                pa.list_(self.meta_struct),
                nullable=False
            ),

        ], metadata={
            b"software_provider": b"OpenMS",
            b"creation_date": self.creation_date.encode(),
            b"uuid": str(uuid.uuid4()).encode(),
            b"file_type": b"proteins",
            b"creator": b"OpenMS",
            b"qpx_version": b"1.0",
        })

    def _init_protein_groups_schema(self):
        float_value_struct = pa.struct([
            ("name", pa.string()),
            ("values", pa.list_(pa.float64())),
        ])

        string_value_struct = pa.struct([
            ("name", pa.string()),
            ("values", pa.list_(pa.string())),
        ])

        integer_value_struct = pa.struct([
            ("name", pa.string()),
            ("values", pa.list_(pa.int64())),
        ])

        # =========================================================
        # protein groups schema
        # =========================================================

        self.protein_groups_schema = pa.schema([
            pa.field("group_type", pa.string(), nullable=False),
            pa.field("probability", pa.float64(), nullable=False),
            pa.field("accessions", pa.list_(pa.string()), nullable=False),
            pa.field("run_identifier", pa.string(), nullable=False),
            pa.field("group_index", pa.int32(), nullable=False),
            pa.field(
                "float_data",
                pa.list_(float_value_struct)
            ),
            pa.field(
                "string_data",
                pa.list_(string_value_struct)
            ),
            pa.field(
                "integer_data",
                pa.list_(integer_value_struct)
            ),
        ], metadata={
            b"software_provider": b"OpenMS",
            b"creation_date": self.creation_date.encode(),
            b"uuid": str(uuid.uuid4()).encode(),
            b"file_type": b"protein_groups",
            b"creator": b"OpenMS",
            b"qpx_version": b"1.0",
        })

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

    def validate_psm(self, row, only_ms2, remove_missing_spectrum):
        spectrum_reference = row.get("spectrum_reference")
        if spectrum_reference is None:
            return False

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
            self._process_dissociation_methods(spectrum, ms_level)

        score = row.get("score")

        if score is None or pd.isna(score) or np.isinf(score):
            self._stats.invalid_score += 1
            invalid = True
        else:
            invalid = False

        # filtering
        if remove_missing_spectrum and (missing or empty or invalid):
            logger.debug(f"Removing Missing PSM {spectrum_reference}")
            return False

        if only_ms2 and ms_level != 2:
            logger.debug(f"Removing PSM {spectrum_reference}")
            return False
        return True

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

    @staticmethod
    def get_meta_features(metavalues, key):
        for metavalue in metavalues:
            if metavalue["name"] == key:
                return metavalue["value"]
        return None