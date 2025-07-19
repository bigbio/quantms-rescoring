# Get logger for this module
from quantmsrescore.logging_config import get_logger

logger = get_logger(__name__)

from collections import defaultdict
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple, DefaultDict
from warnings import filterwarnings
import pandas as pd
import re

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
from quantmsrescore.utils import IdXMLReader


class ScoreStats:
    """Statistics about score occurrence in peptide hits."""

    def __init__(self):
        self.total_hits: int = 0
        self.missing_count: int = 0

    @property
    def missing_percentage(self) -> float:
        """Calculate percentage of missing scores."""
        return (self.missing_count / self.total_hits * 100) if self.total_hits else 0


class IdXMLRescoringReader(IdXMLReader):
    """
    Reader class for processing and rescoring idXML files containing peptide identifications.

    This class handles reading and parsing idXML files, managing PSMs (Peptide-Spectrum Matches),
    and provides functionality for spectrum validation and scoring analysis.

    Attributes
    ----------
    filename : Path
        Path to the idXML file.
    high_score_better : Optional[bool]
        Indicates if higher scores are better.
    """

    def __init__(
            self,
            idxml_filename: Union[Path, str],
            mzml_file: Union[str, Path],
            only_ms2: bool = True,
            remove_missing_spectrum: bool = True,
    ) -> None:
        """
        Initialize the IdXMLRescoringReader with the specified files.

        Parameters
        ----------
        idexml_filename : Union[Path, str]
            Path to the idXML file to be read and parsed.
        mzml_file : Union[str, Path]
            Path to the mzML file for spectrum lookup.
        only_ms2 : bool, optional
            Flag to filter for MS2 spectra only, by default True.
        remove_missing_spectrum : bool, optional
            Flag to remove PSMs with missing spectra, by default True.
        """
        super().__init__(idxml_filename)
        self.build_spectrum_lookup(mzml_file)
        self.high_score_better: Optional[bool] = None

        # Private attributes
        self._psms: Optional[PSMList] = None
        self._psms_df: Optional[pd.DataFrame] = None
        self.psm_clean(
            only_ms2=only_ms2, remove_missing_spectrum=remove_missing_spectrum
        )
        self._build_psm_index(only_ms2=only_ms2)

    @property
    def psms(self) -> Optional[PSMList]:
        """Get the list of PSMs."""
        return self._psms

    @psms.setter
    def psms(self, psm_list: PSMList) -> None:
        """Set the list of PSMs."""
        if not isinstance(psm_list, PSMList):
            raise TypeError("psm_list must be an instance of PSMList")
        self._psms = psm_list

    @property
    def psms_df(self) -> Optional[pd.DataFrame]:
        """Get the list of PSMs."""
        return self._psms_df

    @psms_df.setter
    def psms_df(self, psms: pd.DataFrame) -> None:
        """Set the list of PSMs."""
        if not isinstance(psms, pd.DataFrame):
            raise TypeError("psms must be an instance of DataFrame")
        self._psms_df = psms

    def analyze_score_coverage(self) -> Dict[str, ScoreStats]:
        """
        Analyze the coverage of scores across peptide hits.

        Returns
        -------
        Dict[str, ScoreStats]
            A dictionary mapping score names to their respective statistics.
        """
        scores_stats: Dict[str, ScoreStats] = defaultdict(ScoreStats)
        total_hits = sum(len(peptide_id.getHits()) for peptide_id in self.oms_peptides)

        for peptide_id in self.oms_peptides:
            for hit in peptide_id.getHits():
                meta_values = []
                hit.getKeys(meta_values)
                for score in meta_values:
                    scores_stats[score].total_hits += 1

        for stats in scores_stats.values():
            stats.missing_count = total_hits - stats.total_hits

        return scores_stats

    @staticmethod
    def log_score_coverage(score_stats: Dict[str, ScoreStats]) -> None:
        """
        Log information about score coverage.

        Parameters
        ----------
        score_stats : Dict[str, ScoreStats]
            Dictionary mapping score names to their statistics.
        """
        for score, stats in score_stats.items():
            if stats.missing_count > 0:
                percentage = stats.missing_percentage
                logger.warning(
                    f"Score {score} is missing in {stats.missing_count} PSMs "
                    f"({percentage:.1f}% of total)"
                )
                if percentage > 10:
                    logger.error(f"Score {score} is missing in more than 10% of PSMs")

    @staticmethod
    def _parse_psm(
            protein_ids: Union[oms.ProteinIdentification, List[oms.ProteinIdentification]],
            peptide_id: oms.PeptideIdentification,
            peptide_hit: oms.PeptideHit,
            is_decoy: bool = False,
    ) -> Optional[PSM]:
        """
        Parse a peptide-spectrum match (PSM) from given protein and peptide models.

        Parameters
        ----------
        protein_ids : Union[oms.ProteinIdentification, List[oms.ProteinIdentification]]
            Protein identification(s) associated with the PSM.
        peptide_id : oms.PeptideIdentification
            Peptide identification containing the peptide hit.
        peptide_hit : oms.PeptideHit
            Peptide hit to be parsed into a PSM.
        is_decoy : bool, optional
            Indicates if the PSM is a decoy, by default False.

        Returns
        -------
        Optional[PSM]
            A PSM object if parsing is successful, otherwise None.
        """
        try:
            peptidoform = psm_utils.io.idxml.IdXMLReader._parse_peptidoform(
                peptide_hit.getSequence().toString(), peptide_hit.getCharge()
            )

            spectrum_ref = peptide_id.getMetaValue("spectrum_reference")
            rt = peptide_id.getRT()

            # Create provenance tracking models
            provenance_key = OpenMSHelper.get_psm_hash_unique_id(
                peptide_hit=peptide_id, psm_hit=peptide_hit
            )

            return PSM(
                peptidoform=peptidoform,
                spectrum_id=spectrum_ref,
                run=psm_utils.io.idxml.IdXMLReader._get_run(protein_ids, peptide_id),
                is_decoy=is_decoy,
                score=peptide_hit.getScore(),
                precursor_mz=peptide_id.getMZ(),
                retention_time=rt,
                rank=peptide_hit.getRank() + 1,  # Ranks in idXML start at 0
                source="idXML",
                provenance_data={provenance_key: ""},  # We use only the key for provenance
            )
        except Exception as e:
            logger.error(f"Failed to parse PSM: {e}")
            return None

    def _build_psm_index(self, only_ms2: bool = True) -> Tuple[PSMList, pd.DataFrame]:
        """
        Read and parse the idXML file to extract PSMs.

        Parameters
        ----------
        only_ms2 : bool, optional
            Flag to filter for MS2 spectra only, by default True.

        Returns
        -------
        PSMList
            A list of parsed PSM objects.
        """
        psm_list = []
        psm_df = pd.DataFrame()
        fixed_mods = self.oms_proteins[0].getSearchParameters().fixed_modifications
        var_mods = self.oms_proteins[0].getSearchParameters().variable_modifications
        mods_name_dict = {}
        for m in fixed_mods + var_mods:
            mods_name_dict[m.decode('utf-8').split(" ")[0]] = " ".join(m.decode('utf-8').split(" ")[1:]).replace("(", "").replace(")", "")

        instrument = OpenMSHelper.get_instrument(self.exp)
        if only_ms2 and self.spec_lookup is None:
            logger.warning("Spectrum lookup not initialized, cannot filter for MS2 spectra")
            only_ms2 = False

        for peptide_id in self.oms_peptides:
            if self.high_score_better is None:
                self.high_score_better = peptide_id.isHigherScoreBetter()
            elif self.high_score_better != peptide_id.isHigherScoreBetter():
                logger.warning("Inconsistent score direction found in idXML file")

            for psm_hit in peptide_id.getHits():
                if (
                        only_ms2
                        and self.spec_lookup is not None
                        and OpenMSHelper.get_ms_level(peptide_id, self.spec_lookup, self.exp) != 2
                ):
                    continue
                psm = self._parse_psm(
                    protein_ids=self.oms_proteins,
                    peptide_id=peptide_id,
                    peptide_hit=psm_hit,
                    is_decoy=OpenMSHelper.is_decoy_peptide_hit(psm_hit),
                )

                sequence = psm_hit.getSequence().toUnmodifiedString()
                peptidoform = psm_hit.getSequence().toString()
                mods, mod_sites = extract_modifications(peptidoform, mods_name_dict)
                nce = OpenMSHelper.get_nce_psm(peptide_id, self.spec_lookup, self.exp)
                if psm is not None:
                    psm_list.append(psm)
                    psm_df = psm_df._append({"sequence": sequence,
                                             "charge": psm_hit.getCharge(),
                                             "mods": mods,
                                             "mod_sites": mod_sites,
                                             "nce": nce,
                                             "provenance_data": next(iter(psm.provenance_data.keys())),
                                             "instrument": instrument}, ignore_index=True)

        self._psms = PSMList(psm_list=psm_list)
        self._psms_df = psm_df
        logger.info(f"Loaded {len(self._psms)} PSMs from {self.filename}")

        return self._psms, self._psms_df


def extract_modifications(peptidoform, mods_name_dict):
    pattern = re.compile(r"(\(.*?\))")
    mods = pattern.findall(peptidoform)
    mods_res = []
    mod_sites = []
    pre_len = 0
    for i, v in enumerate(list(pattern.finditer(peptidoform))):
        if peptidoform.startswith(".") and i == 0:
            mods_res.append(v.group(0)[1:-1] + "@" + mods_name_dict[v.group(0)[1:-1]].replace(" ", "_"))
            mod_sites.append("0")
            pre_len += 1 + len(mods[0])
        else:
            position = v.start() - pre_len
            pre_len += len(v.group(0))
            mods_res.append(v.group(0)[1:-1] + "@" + peptidoform[v.start()-1])
            mod_sites.append(str(position))
    return ";".join(mods_res), ";".join(mod_sites)