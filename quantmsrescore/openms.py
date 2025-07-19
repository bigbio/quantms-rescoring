import re
from pathlib import Path
from typing import List, Union, Optional, Tuple
from warnings import filterwarnings

filterwarnings(
    "ignore",
    message="OPENMS_DATA_PATH environment variable already exists",
    category=UserWarning,
    module="pyopenms",
)

import numpy as np
import pyopenms as oms
from packaging import version
from psm_utils import PSM
from pyopenms import (
    PeptideIdentification,
    ProteinIdentification,
    SpectrumLookup,
    PeptideHit,
    TheoreticalSpectrumGenerator,
)

from quantmsrescore.constants import (
    DEEPLC_FEATURES,
    MS2PIP_FEATURES,
    OPENMS_DISSOCIATION_METHODS_PATCH_3_3_0,
    OPENMS_DISSOCIATION_METHODS_PATCH_3_1_0,
)
from quantmsrescore.exceptions import MzMLNotUnixException
from quantmsrescore.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

OPENMS_DECOY_FIELD = "target_decoy"
SPECTRUM_PATTERN = r"(spectrum|scan)=(\d+)"


class OpenMSHelper:
    """
    This class should contain methods to help with OpenMS operations on PSM lists
    Features, parameters.
    """

    @staticmethod
    def count_decoys_targets(
        peptide_list: Union[List[PeptideIdentification], List[PeptideHit]],
    ) -> (int, int):
        """
        Count the number of decoy and target PSMs in the given list.

        This method iterates over a list of PSM objects, counting how many
        are labeled as 'target' or 'decoy' based on their rescoring features
        and the `is_decoy` attribute. It ensures that the counts from the
        rescoring features match the counts from the `is_decoy` attribute.

        Parameters
        ----------
        peptide_list (List[PSM]): A list of PeptideIdentification objects to be analyzed.

        Returns
        -------
        tuple: A tuple containing the count of decoy PSMs and the count of
        target PSMs.

        Raises
        -------
        ValueError: If the counts from the rescoring features do not match
        the counts from the `is_decoy` attribute.
        """

        openms_count_target = 0
        openms_count_decoy = 0

        for pep in peptide_list:
            if isinstance(pep, PeptideHit):
                if OpenMSHelper.is_decoy_peptide_hit(pep):
                    openms_count_decoy += 1
                else:
                    openms_count_target += 1
            else:
                for psm in pep.getHits():
                    if psm.metaValueExists(OPENMS_DECOY_FIELD):
                        if psm.getMetaValue(OPENMS_DECOY_FIELD) == "decoy":
                            openms_count_decoy += 1
                        else:
                            openms_count_target += 1

        if openms_count_decoy + openms_count_target == 0:
            logger.warning("No PSMs found; decoy percentage cannot be computed.")
            return 0, 0
        percentage_decoy = (openms_count_decoy / (openms_count_decoy + openms_count_target)) * 100
        logger.info(
            "Decoy percentage: %s, targets %s and decoys %s",
            percentage_decoy,
            openms_count_target,
            openms_count_decoy,
        )
        return openms_count_decoy, openms_count_target

    @staticmethod
    def get_psm_count(peptide_list: Union[List[PeptideIdentification], List[PeptideHit]]) -> int:
        """
        Count the number of PSMs in the given list.

        This method iterates over a list of PSM objects, counting the total
        number of PSMs.

        Parameters
        ----------
        peptide_list (List[PSM]): A list of PeptideIdentification objects to be analyzed.

        Returns
        -------
        int: The total number of PSMs in the list.
        """

        openms_count = 0

        for pep in peptide_list:
            if isinstance(pep, PeptideHit):
                openms_count += 1
            else:
                openms_count += len(pep.getHits())
        logger.info("Total PSMs: %s", openms_count)
        return openms_count

    @staticmethod
    def is_decoy_peptide_hit(peptide_hit: PeptideHit) -> bool:
        """
        Check if a PeptideHit is a decoy.

        This method checks if a PeptideHit is a decoy based on the
        'target_decoy' field in the PeptideHit.

        Parameters
        ----------
        peptide_hit (PeptideIdentification): A PeptideIdentification object to be checked.

        Returns
        -------
        bool: True if the PeptideHit is a decoy, False otherwise.
        """

        if peptide_hit.metaValueExists(OPENMS_DECOY_FIELD):
            return peptide_hit.getMetaValue(OPENMS_DECOY_FIELD) == "decoy"
        return False

    @staticmethod
    def get_spectrum_lookup_indexer(
        mzml_file: Union[str, Path]
    ) -> tuple[oms.MSExperiment, SpectrumLookup]:
        """
        Create a SpectrumLookup indexer from an mzML file.

        This method loads an mzML file into an MSExperiment object and
        initializes a SpectrumLookup object to read spectra using a
        specified regular expression pattern for scan numbers.

        Parameters
        ----------
        mzml_file : str
        The path to the mzML file to be loaded.

        Returns
        -------
        tuple: A tuple containing the MSExperiment object with the loaded
        """

        if isinstance(mzml_file, Path):
            mzml_file = str(mzml_file)

        exp = oms.MSExperiment()
        oms.MzMLFile().load(mzml_file, exp)

        lookup = SpectrumLookup()
        if "spectrum=" in exp.getSpectrum(0).getNativeID():
            lookup.readSpectra(exp, "spectrum=(?<SCAN>\\d+)")
        else:
            lookup.readSpectra(exp, "scan=(?<SCAN>\\d+)")

        return exp, lookup

    @staticmethod
    def get_spectrum_reference(
        identification: Union[PSM, PeptideIdentification]
    ) -> Union[str, None]:
        """
        Get the spectrum reference for a PSM.

        This method retrieves the spectrum reference from a PSM object,
        which can be either a PSM or a PeptideIdentification object.

        Parameters
        ----------
        identification : Union[PSM, PeptideIdentification]
            The PSM object containing the spectrum reference.

        Returns
        -------
        str
            The spectrum reference for the PSM.
        """
        if isinstance(identification, PSM):
            return identification.spectrum_id
        elif isinstance(identification, PeptideIdentification):
            return identification.getMetaValue("spectrum_reference")
        return None

    @staticmethod
    def get_spectrum_for_psm(
        psm: Union[PSM, PeptideIdentification], exp: oms.MSExperiment, lookup: SpectrumLookup
    ) -> Union[None, oms.MSSpectrum]:

        spectrum_reference = OpenMSHelper.get_spectrum_reference(psm)
        if spectrum_reference is None:
            psm_info = psm.provenance_data if hasattr(psm, "provenance_data") else "N/A"
            logger.warning(
                f"Missing spectrum reference for PSM {psm_info}, skipping spectrum retrieval."
            )
            return None

        matches = re.findall(r"(spectrum|scan)=(\d+)", spectrum_reference)
        if not matches:
            psm_info = psm.provenance_data if hasattr(psm, "provenance_data") else "N/A"
            logger.warning(
                f"Missing or invalid spectrum reference for PSM {psm_info}, "
                f"skipping spectrum retrieval."
            )
            return None
        scan_number = int(matches[0][1])

        try:
            index = lookup.findByScanNumber(scan_number)
            spectrum = exp.getSpectrum(index)
            return spectrum

        except Exception as e:
            psm_info = psm.provenance_data if hasattr(psm, "provenance_data") else "N/A"
            logger.error(
                "Error while retrieving spectrum for PSM %s spectrum_ref %s: %s",
                psm_info,
                spectrum_reference,
                e,
            )
        return None

    @staticmethod
    def write_idxml_file(
        filename: Union[str, Path],
        peptide_ids: List[PeptideIdentification],
        protein_ids: List[ProteinIdentification],
    ) -> None:
        """
        Write protein and peptide identifications to an idXML file.

        Parameters
        ----------
        filename : Union[str, Path]
            The path to the idXML file to be written.
        peptide_ids : List[PeptideIdentification]
            A list of PeptideIdentification objects to be written to the file.
        protein_ids : List[ProteinIdentification]
            A list of ProteinIdentification objects to be written to the file.

        """

        if isinstance(filename, Path):
            filename = str(filename)

        id_data = oms.IdXMLFile()
        id_data.store(filename, protein_ids, peptide_ids)

    @staticmethod
    def get_peaks_by_scan(
        scan_number: int, exp: oms.MSExperiment, lookup: SpectrumLookup
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get spectrum deeplc_models for a given scan number

        Parameters
        ----------
            scan_number: The scan number to look up
            exp: The MSExperiment object containing the spectra
            lookup: The SpectrumLookup object to find the spectrum

        Returns
        -------
            Tuple of (mz_array, intensity_array) if found, None if not found
        """
        try:
            index = lookup.findByScanNumber(scan_number)
            spectrum = exp.getSpectrum(index)
            return spectrum.get_peaks()
        except IndexError:
            logger.warning(f"Scan number {scan_number} not found")
            return None

    @staticmethod
    def get_ms_level(
        psm_hit: PeptideIdentification, spec_lookup: oms.SpectrumLookup, exp: oms.MSExperiment
    ) -> int:
        spectrum = OpenMSHelper.get_spectrum_for_psm(psm_hit, exp, spec_lookup)
        if spectrum is None:
            return -1
        return spectrum.getMSLevel()

    @staticmethod
    def get_psm_hash_unique_id(peptide_hit: PeptideIdentification, psm_hit: PeptideHit) -> str:
        """
        Generate a unique hash identifier for a PSM.

        This method constructs a unique hash string for a given PSM by
        combining the peptide sequence, charge, retention time, and
        spectrum reference.

        Parameters
        ----------
        peptide_hit : PeptideIdentification
            The PeptideIdentification object containing metadata for the PSM.
        psm_hit : PeptideHit
            The PeptideHit object representing the PSM.

        Returns
        -------
        str
            A unique hash string for the PSM.
        """

        spectrum_ref = peptide_hit.getMetaValue("spectrum_reference")
        rank = psm_hit.getRank()
        rt = peptide_hit.getRT()
        sequence = psm_hit.getSequence().toString()
        charge = psm_hit.getCharge()
        unique_hash = f"{spectrum_ref}_{sequence}_{rt}_{charge}_{rank}"
        return unique_hash

    @staticmethod
    def get_str_metavalue_round(metavalue: float):
        """
        Get a string representation of a metadata value, rounded to 4 decimal places.

        Parameters
        ----------
        metavalue : float
            The metadata value to be converted to a string.

        Returns
        -------
        str
            A string representation of the metadata value, rounded to 4 decimal places.
        """
        if np.isnan(metavalue) or np.isinf(metavalue):
            return "0.0"
        return "{:.4f}".format(metavalue)

    @staticmethod
    def get_canonical_feature(feature: str) -> Union[str, None]:
        """
        Retrieve the canonical feature name for a given feature.

        This method searches through predefined feature mappings to find
        the canonical name corresponding to the provided feature string.
        It first checks the DEEPLC_FEATURES mapping, followed by the
        MS2PIP_FEATURES mapping.

        Parameters
        ----------
        feature : str
            The feature name to be converted to its canonical form.

        Returns
        -------
        str
            The canonical feature name if found, otherwise None.
        """
        if feature is None:
            return None

        canonical_feature = next((k for k, v in DEEPLC_FEATURES.items() if v == feature), None)
        if canonical_feature is not None:
            return canonical_feature

        canonical_feature = next((k for k, v in MS2PIP_FEATURES.items() if v == feature), None)
        return canonical_feature

    @staticmethod
    def validate_features(original_features: List[str]) -> List[str]:
        """
        This function make sures that the given features are supported by the tool
        in DEEPLC_FEATURES and MS2PIP_FEATURES.

        Parameters
        ----------

        original_features: List[str]
            The list of features to validate.

        Returns
        -------

        List[str]
            The list of validated features.
        """

        validated_features = []
        for feature in original_features:
            feature = feature.strip()  # Remove leading and trailing whitespaces
            if feature in DEEPLC_FEATURES.values() or feature in MS2PIP_FEATURES.values():
                validated_features.append(feature)
            else:
                logger.warning(f"Feature {feature} not supported by quantms rescoring")
        return validated_features

    @staticmethod
    def get_pyopenms_version():
        """
        Get the version of the pyopenms library.

        Returns
        -------
        str
            The version of the pyopenms library.
        """
        oms_version = oms.VersionInfo.getVersionStruct()
        return (
            f"{oms_version.version_major}.{oms_version.version_minor}.{oms_version.version_patch}"
        )

    @staticmethod
    def get_pyopenms_dissociation_matrix() -> Union[List[dict], None]:
        """
        Retrieve the dissociation methods matrix based on the pyopenms version.

        This method checks the current version of pyopenms and returns the
        appropriate dissociation methods matrix. If the version is not supported,
        a warning is logged and None is returned.

        Returns
        -------
        Union[List[dict], None]
            A list of dissociation methods if the version is supported, otherwise None.
        """
        oms_version = version.parse(OpenMSHelper.get_pyopenms_version())
        dissociation_methods = []
        if oms_version < version.parse("3.2.0"):
            dissociation_methods = OPENMS_DISSOCIATION_METHODS_PATCH_3_1_0
        if oms_version >= version.parse("3.2.0"):
            dissociation_methods = OPENMS_DISSOCIATION_METHODS_PATCH_3_3_0
        if not dissociation_methods:
            logger.warning("OpenMS version not supported, can't find the dissociation method.")
            return None
        return dissociation_methods

    @staticmethod
    def get_dissociation_method(method_index: int, matrix: List[dict] = None) -> Union[str, None]:
        """
        Retrieve the dissociation method name based on the method index and pyopenms version.

        This method determines the appropriate dissociation method list to use
        based on the current pyopenms version and retrieves the method name
        corresponding to the provided index.

        Parameters
        ----------
        method_index : int
            The index of the dissociation method to retrieve.
        matrix : List[dict], optional
            The dissociation methods matrix to use. If not provided, the default
            matrix for the current pyopenms version is used.

        Returns
        -------
        Union[str, None]
            The name of the dissociation method if found, otherwise None.
        """
        if matrix is None:
            dissociation_methods = OpenMSHelper.get_pyopenms_dissociation_matrix()
        else:
            dissociation_methods = matrix
        if dissociation_methods is None:
            return None
        if method_index < 0 or method_index >= len(dissociation_methods):
            logger.warning("Invalid dissociation method index.")
            return None
        return list(dissociation_methods[method_index].keys())[0]

    @classmethod
    def check_unix_compatibility(cls, mzml_path: Union[str, Path]):
        """
        Check if an mzML file has Unix-style line endings.

        This method verifies whether the specified mzML file uses Unix-style
        LF line endings. If Windows-style CRLF line endings are detected,
        an MzMLNotUnixException is raised.

        This is necessary because ms2rescore-rs is using a rust dependency to read mzML files, that if the file
        fails with Windows-style CRLF line endings, do not give an error message, just fail silently.

        Parameters
        ----------
        mzml_path : Union[str, Path]
            The path to the mzML file to be checked.

        Raises
        ------
        MzMLNotUnixException
            If the file contains Windows-style CRLF line endings.
        """

        if isinstance(mzml_path, Path):
            mzml_path = str(mzml_path)

        with open(mzml_path, "rb") as f:
            content = f.read()
            if b"\r\n" in content:
                raise MzMLNotUnixException(
                    f"File {mzml_path} has Windows-style CRLF line endings. Please convert to LF (Unix-style) using `dos2unix` or similar."
                )
            else:
                logger.info(f"File {mzml_path} has the correct Unix-style LF line endings.")

    @staticmethod
    def get_ms_tolerance(
        oms_proteins: List[ProteinIdentification],
    ) -> Union[Tuple[float, str], Tuple[float, None]]:
        """
        Get the mass tolerance and unit from the search parameters.

        Parameters
        ----------
        oms_proteins : List[ProteinIdentification]
            The list of ProteinIdentification objects to be analyzed.

        Returns
        -------
        Tuple[float, str]
            A tuple containing the mass tolerance and the unit.

        """

        if oms_proteins is None:
            return 0.0, None
        search_parameters = oms_proteins[0].getSearchParameters()
        if search_parameters.fragment_mass_tolerance_ppm:
            return search_parameters.fragment_mass_tolerance, "ppm"
        else:
            return search_parameters.fragment_mass_tolerance, "Da"

    @staticmethod
    def generate_theoretical_spectrum(peptide_sequence: str, charge: int):
        """
        Generate a theoretical spectrum for a given peptide sequence.

        Parameters:
        - peptide (str): Peptide sequence (e.g., "PEPTIDE").
        - charge (int): Charge state of the fragments.

        Returns:
        - theoretical_mzs (list): List of theoretical fragment m/z values.
        - ion_labels (list): Corresponding fragment ion labels.
        """

        tsg = TheoreticalSpectrumGenerator()
        spec = oms.MSSpectrum()
        peptide = oms.AASequence.fromString(peptide_sequence)

        # Generate b- and y-ions (can be extended for other ion types)
        tsg.getSpectrum(spec, peptide, 1, charge)

        theoretical_mzs = [peak.getMZ() for peak in spec]
        return theoretical_mzs

    @staticmethod
    def get_predicted_ms_tolerance(
        exp: oms.MSExperiment, ppm_tolerance: float
    ) -> Tuple[float, str]:
        """
        Calculate the predicted mass tolerance in Daltons for an MS experiment.

        This method computes the maximum fragment mass from the spectra in the
        given MSExperiment and calculates the mass tolerance in Daltons based
        on the provided parts-per-million (ppm) tolerance.

        Parameters
        ----------
        exp : oms.MSExperiment
            The MSExperiment object containing the spectra.
        ppm_tolerance : float
            The mass tolerance in parts-per-million.

        Returns
        -------
        Tuple[float, str]
            A tuple containing the calculated mass tolerance in Daltons and
            the unit "Da".
        """
        max_frag_mass = 0
        for spec in exp:
            if spec.getMSLevel() == 2:
                spec.updateRanges()
                if spec.getMaxMZ() > max_frag_mass:
                    max_frag_mass = spec.getMaxMZ()

        tol_da = max_frag_mass * ppm_tolerance / 1e6
        tol_da = round(tol_da, 4)
        return tol_da, "Da"

    @staticmethod
    def get_mslevel_spectra(file_name, ms_level):
        """
        Get the mslevel spectra from an mzML file.
        """
        exp = OpenMSHelper.get_spectrum_lookup_indexer(file_name)[0]
        spectra = []
        for spec in exp:
            if spec.getMSLevel() == ms_level:
                spectra.append(spec)
        return spectra

    @staticmethod
    def get_instrument(exp: oms.MSExperiment):
        instrument = exp.getInstrument().getName()
        return instrument

    @staticmethod
    def get_nce_psm(psm_hit: PeptideIdentification, spec_lookup: oms.SpectrumLookup, exp: oms.MSExperiment):
        spectrum = OpenMSHelper.get_spectrum_for_psm(psm_hit, exp, spec_lookup)
        collision_energy = float(re.findall(r"@[a-zA-Z]+(\d+\.\d+)\s", spectrum.getMetaValue("filter string"))[0])
        return collision_energy