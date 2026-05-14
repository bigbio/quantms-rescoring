# Get logger for this module
import numpy as np
from quantmsrescore.logging_config import get_logger
from quantmsrescore import __version__

logger = get_logger(__name__)

from collections import defaultdict
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple, DefaultDict
from warnings import filterwarnings
import pandas as pd
import re
import pyarrow.parquet as pq
import copy
from datetime import datetime, timezone

filterwarnings(
    "ignore",
    message="OPENMS_DATA_PATH environment variable already exists",
    category=UserWarning,
    module="pyopenms",
)

import psm_utils
import pyopenms as oms
from psm_utils import PSM, PSMList

from quantmsrescore.openms import OpenMSHelper
from quantmsrescore.utils import ParquetReader

# Patterns to match open and closed round/square brackets
MOD_PATTERN = re.compile(r"\(((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*)\)")
MOD_PATTERN_NTERM = re.compile(r"^\.\[((?:[^][]+|\[(?:[^][]+|\[[^][]*\])*\])*)\]")
MOD_PATTERN_CTERM = re.compile(r"\.\[((?:[^][]+|\[(?:[^][]+|\[[^][]*\])*\])*)\]$")

# 当前 UTC 时间
now = datetime.now(timezone.utc)

# 生成 run identifier
run_identifier = f"quantms-rescoring_{now.strftime('%Y-%m-%d_%H:%M:%S')}"


class ScoreStats:
    """Statistics about score occurrence in peptide hits."""

    def __init__(self):
        self.total_hits: int = 0
        self.missing_count: int = 0

    @property
    def missing_percentage(self) -> float:
        """Calculate percentage of missing scores."""
        return (self.missing_count / self.total_hits * 100) if self.total_hits else 0


class ParquetRescoringReader(ParquetReader):
    """
    Reader class for parsing Comet/OpenMS parquet identification folders.

    Example folder structure
    ------------------------
    UPS1_12500amol_R1_comet.idparquet/
    ├── protein_groups.parquet
    ├── proteins.parquet
    ├── psms.parquet
    └── search_params.parquet
    """

    def __init__(
            self,
            parquet_dir: Union[str, Path, List[Union[str, Path]]],
            mzml_file: Union[str, Path],
            only_ms2: bool = True,
            remove_missing_spectrum: bool = True,
    ) -> None:

        super().__init__(parquet_dir)

        self._mzml_path = str(mzml_file) if isinstance(mzml_file, Path) else mzml_file
        self.exp, self.spec_lookup = OpenMSHelper.get_spectrum_lookup_indexer(self._mzml_path)
        logger.info(f"Built SpectrumLookup from {self._mzml_path}")

        self.high_score_better: Optional[bool] = None
        self.search_params: Optional[Dict] = None

        self._psms: Optional[PSMList] = None
        self._psms_df: Optional[pd.DataFrame] = None
        self._proteins_df: Optional[pd.DataFrame] = None
        self._protein_groups_df: Optional[pd.DataFrame] = None

        self._build_psm_index()
        self._build_protein_index()
        self._build_protein_groups_index()

    @property
    def psms(self) -> Optional[PSMList]:
        return self._psms

    @property
    def psms_df(self) -> Optional[pd.DataFrame]:
        return self._psms_df

    @psms.setter
    def psms(self, psm_list: PSMList) -> None:
        """Set the list of PSMs."""
        if not isinstance(psm_list, PSMList):
            raise TypeError("psm_list must be an instance of PSMList")
        self._psms = psm_list

    @property
    def proteins_df(self) -> Optional[pd.DataFrame]:
        return self._proteins_df

    @proteins_df.setter
    def proteins_df(self, proteins_df: pd.DataFrame) -> None:
        """Get proteins DataFrame."""
        if not isinstance(proteins_df, pd.DataFrame):
            raise TypeError("proteins_df must be an instance of DataFrame")
        self._proteins_df = proteins_df

    @property
    def protein_groups_df(self) -> Optional[pd.DataFrame]:
        return self._protein_groups_df

    @protein_groups_df.setter
    def protein_groups_df(self, protein_groups_df: pd.DataFrame) -> None:
        """Get protein groups DataFrame."""
        if not isinstance(protein_groups_df, pd.DataFrame):
            raise TypeError("protein_groups_df must be an instance of DataFrame")
        self._protein_groups_df = protein_groups_df

    @property
    def spectrum_path(self) -> Optional[Union[str, Path]]:
        """Get the path to the mzML file."""
        return self._mzml_path

    @staticmethod
    def _safe_get(row, keys, default=None):
        """
        Safely get value from row using candidate column names.
        """
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return default

    @staticmethod
    def _extract_sequence(peptide: str) -> str:
        """
        Extract unmodified peptide sequence.
        """
        if peptide is None:
            return None

        sequence = re.sub(r"\[.*?\]", "", peptide)
        sequence = re.sub(r"\(.*?\)", "", sequence)

        return sequence

    @staticmethod
    def _extract_modifications(peptidoform: str) -> Tuple[str, str]:
        """
        Extract modifications and modification sites.

        Example
        -------
        ACDEK[+15.9949]R

        Returns
        -------
        mods: Oxidation@M
        mod_sites: 5
        """
        if peptidoform is None:
            return "", ""

        mods = []
        mod_sites = []

        # [] modification
        pattern = re.compile(r"([A-Z])(\[.*?\]|\(.*?\))")

        for match in pattern.finditer(peptidoform):
            aa = match.group(1)
            mod = match.group(2)

            position = match.start(1) + 1

            mods.append(f"{mod[1:-1]}@{aa}")
            mod_sites.append(str(position))

        return ";".join(mods), ";".join(mod_sites)

    def _parse_psm(self, row: pd.Series, search_params: Dict) -> Optional[PSM]:
        """
        Convert parquet row to psm_utils.PSM.
        """

        peptide = self._safe_get(
            row,
            [
                "peptide",
                "sequence",
                "peptidoform",
                "PeptideSequence"
            ]
        )

        if peptide is None:
            return None

        charge = self._safe_get(
            row,
            [
                "charge",
                "precursor_charge",
                "Charge"
            ],
            0
        )

        spectrum_id = self._safe_get(
            row,
            [
                "spectrum_ref",
                "spectrum_reference",
                "spectrum_id",
                "scan"
            ]
        )

        score = self._safe_get(
            row,
            [
                "score",
                "hyperscore",
                "Score"
            ],
            0.0
        )

        is_decoy = self._safe_get(
            row,
            [
                "is_decoy",
                "decoy"
            ],
            False
        )

        rank = self._safe_get(
            row,
            [
                "rank"
            ],
            1
        )

        rt = self._safe_get(
            row,
            [
                "rt",
                "retention_time"
            ]
        )

        precursor_mz = self._safe_get(
            row,
            [
                "precursor_mz",
                "observed_mz"
            ]
        )
        run_file_name = self._safe_get(
            row,
            [
                "run_file_name"
            ]
        )

        try:
            peptidoform = self._parse_peptidoform(peptide, charge)

            provenance_key = f"{spectrum_id}_{peptide}_{rt}_{charge}_{rank}"

            psm = PSM(
                peptidoform=peptidoform,
                spectrum_id=str(spectrum_id),
                run=run_file_name,
                is_decoy=bool(is_decoy),
                score=float(score),
                precursor_mz=precursor_mz,
                retention_time=rt,
                rank=int(rank),
                source="parquet",
                metadata={
                    "search_engine_name": search_params["search_engine"]
                },
                provenance_data={provenance_key: ""},  # We use only the key for provenance
            )

            return psm

        except Exception as e:
            logger.error(f"Failed to parse PSM: {e}")
            return None

    def _build_psm_index(self):
        """
        Build PSMList and DataFrame.
        """

        merged_psms = {}
        merged_records = {}

        for parquet_dir in self.parquet_dirs:
            search_params = self._load_search_params(parquet_dir)
            if self.search_params is None:
                self.search_params = search_params
            else:
                search_params["run_identifier"] = run_identifier
                search_params["search_engine"] = "quantms-rescoring"
                search_params["search_engine_version"] = __version__
                self.search_params.update(search_params)

            psms_file = parquet_dir / "psms.parquet"

            psms_df = self._load_parquet(psms_file)

            if psms_df.empty:
                continue

            for _, row in psms_df.iterrows():

                psm = self._parse_psm(row, search_params)
                high_score_better = self._safe_get(
                    row,
                    [
                        "higher_score_better"
                    ]
                )

                if self.high_score_better is None:
                    self.high_score_better = high_score_better
                elif self.high_score_better != high_score_better:
                    logger.warning("Inconsistent score direction found in parquet file")

                if psm is None:
                    continue

                peptide = self._safe_get(
                    row,
                    [
                        "peptidoform"
                    ]
                )
                mods, mod_sites = self._extract_modifications(peptide)

                # Start with all original columns
                record = row.to_dict()
                # Overwrite/add columns we want to update
                record.update({
                    "mods": mods,
                    "mod_sites": mod_sites,
                    "unique_hash": next(iter(psm.provenance_data.keys()))
                })

                record["charge"] = record.pop("precursor_charge")
                prov_key = "_".join(next(iter(psm.provenance_data.keys())).split("_")[:2])
                psm_metavalues = row["psm_metavalues"].tolist()

                if prov_key not in merged_psms:
                    if "Comet" not in search_params["search_engine"]:
                        psm.score = np.inf
                        record["score"] = np.inf
                        record["score_type"] = row["score_type"]
                    merged_psms[prov_key] = copy.copy(psm)
                    record["psm_metavalues"] = psm_metavalues
                    merged_records[prov_key] = copy.copy(record)
                else:
                    if search_params["search_engine"] == "Comet":
                        merged_psms[prov_key].score = psm.score
                        merged_records[prov_key]["score"] = psm.score
                        merged_records[prov_key]["score_type"] = row["score_type"]

                    merged_records[prov_key]["psm_metavalues"] = self.merge_dedup_metavalues(
                        merged_records[prov_key]["psm_metavalues"],
                        psm_metavalues
                    )

            logger.info(
                f"Loaded PSMs from {parquet_dir}"
            )

        self._psms = PSMList(psm_list=list(merged_psms.values()))
        self._psms_df = pd.DataFrame(merged_records.values())

    @staticmethod
    def _parse_peptidoform(sequence: str, charge: int) -> str:
        """
        Parse idXML peptide to :py:class:`~psm_utils.peptidoform.Peptidoform`.

        Notes
        -----
        Implemented according to the documentation on
        `github.com/OpenMS/OpenMS <https://github.com/OpenMS/OpenMS/blob/8cb90/src/openms/include/OpenMS/CHEMISTRY/AASequence.h>`_
        . The differentiation between square- and round bracket notation is removed after parsing.
        """
        sequence = MOD_PATTERN.sub(r"[\1]", sequence)
        if sequence[:2] == ".[":
            sequence = MOD_PATTERN_NTERM.sub(r"[\1]-", sequence)
        if sequence[-1] == "]":
            sequence = MOD_PATTERN_CTERM.sub(r"-[\1]", sequence)
        sequence = sequence.strip(".")
        sequence += f"/{charge}"

        return sequence

    @staticmethod
    def merge_dedup_metavalues(existing, new):
        if existing is None:
            existing = []
        if new is None:
            new = []

        if isinstance(existing, np.ndarray):
            existing = existing.tolist()
        if isinstance(new, np.ndarray):
            new = new.tolist()

        merged = {}

        for x in existing + new:
            if not x or "name" not in x:
                continue
            merged[x["name"]] = x

        return list(merged.values())

    def _build_protein_index(self):
        """
        Build merged protein DataFrame from multiple parquet directories.
        """

        merged_proteins = {}

        for parquet_dir in self.parquet_dirs:

            proteins_file = parquet_dir / "proteins.parquet"

            if not proteins_file.exists():
                logger.warning(f"{proteins_file} not found")
                continue

            proteins_df = self._load_parquet(proteins_file)

            if proteins_df.empty:
                continue

            for _, row in proteins_df.iterrows():

                record = row.to_dict()

                accession = record["accession"]

                # normalize ndarray -> list
                metavalues = record.get("metavalues", [])
                if isinstance(metavalues, np.ndarray):
                    metavalues = metavalues.tolist()

                record["metavalues"] = metavalues

                # update run_identifier
                record["run_identifier"] = run_identifier

                if accession not in merged_proteins:
                    merged_proteins[accession] = copy.deepcopy(record)
                else:
                    existing = merged_proteins[accession]
                    # merge metavalues
                    existing["metavalues"] = self.merge_dedup_metavalues(
                        existing.get("metavalues", []),
                        metavalues
                    )

        self._proteins_df = pd.DataFrame(merged_proteins.values())

    def _build_protein_groups_index(self):
        """
        Build merged protein groups DataFrame.
        """

        merged_groups = []

        group_index = 0

        for parquet_dir in self.parquet_dirs:

            protein_groups_file = parquet_dir / "protein_groups.parquet"

            if not protein_groups_file.exists():
                logger.warning(f"{protein_groups_file} not found")
                continue

            protein_groups_df = self._load_parquet(protein_groups_file)

            if protein_groups_df.empty:
                continue

            for _, row in protein_groups_df.iterrows():

                record = row.to_dict()

                # normalize ndarray -> list
                for key in [
                    "accessions",
                    "float_data",
                    "string_data",
                    "integer_data"
                ]:
                    value = record.get(key)

                    if isinstance(value, np.ndarray):
                        record[key] = value.tolist()

                # overwrite run identifier
                record["run_identifier"] = run_identifier

                # reassign unique group index
                record["group_index"] = group_index

                merged_groups.append(record)

                group_index += 1

            logger.info(
                f"Loaded protein groups from {parquet_dir}"
            )

        self._protein_groups_df = pd.DataFrame(merged_groups)

    def analyze_score_coverage(self) -> Dict[str, ScoreStats]:

        score_stats: Dict[str, ScoreStats] = defaultdict(ScoreStats)

        total_hits = len(self._psms_df)

        for psm_metavalues in self._psms_df["psm_metavalues"]:

            if psm_metavalues is None:
                continue

            if isinstance(psm_metavalues, np.ndarray):
                psm_metavalues = psm_metavalues.tolist()

            seen_scores = {
                x["name"]
                for x in psm_metavalues
                if isinstance(x, dict) and "name" in x
            }

            for score_name in seen_scores:
                score_stats[score_name].total_hits += 1

        for stats in score_stats.values():
            stats.missing_count = total_hits - stats.total_hits

        return score_stats

    @staticmethod
    def log_score_coverage(score_stats: Dict[str, ScoreStats]) -> None:
        """
        Log feature coverage statistics.
        """

        for score, stats in score_stats.items():

            if stats.missing_count > 0:

                percentage = stats.missing_percentage

                logger.warning(
                    f"Feature {score} is missing in "
                    f"{stats.missing_count} PSMs "
                    f"({percentage:.1f}% of total)"
                )

                if percentage > 10:
                    logger.error(
                        f"Feature {score} is missing "
                        f"in more than 10% of PSMs"
                    )