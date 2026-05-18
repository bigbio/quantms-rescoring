import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from quantmsrescore.annotator import FeatureAnnotator
from quantmsrescore.idparquet_reader import ParquetRescoringReader
from quantmsrescore.openms import OpenMSHelper
from quantmsrescore.rescoring import cli

TESTS_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@pytest.mark.skip(reason="This is for local test in big datasets, skipping for now")
def test_ms2rescore():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "msrescore2feature",
            "--idparquet",
            "{}/test_data/TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet.idparquet".format(
                TESTS_DIR
            ),
            "--mzml",
            "{}/test_data/TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzML".format(
                TESTS_DIR
            ),
            "--processes",
            "2",
            "--feature_generators",
            "'ms2pip,deeplc'",
        ],
    )
    assert result.exit_code == 0


def test_idxmlreader():

    id_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet.idparquet"
    )

    mzml_file = (
        TESTS_DIR / "test_data" / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzML"
    )

    idxml_reader = ParquetRescoringReader(id_file, mzml_file)
    logging.info("Loaded %s PSMs from %s", len(idxml_reader.psms), id_file)
    assert len(idxml_reader.psms) == 5346

    openms_helper = OpenMSHelper()
    decoys, targets = openms_helper.count_decoys_targets(idxml_reader.psms_df)
    logging.info(
        "Loaded %s PSMs from %s, %s decoys and %s targets",
        len(idxml_reader.psms),
        id_file,
        decoys,
        targets,
    )
    assert decoys == 1923
    assert targets == 3423
    stats = idxml_reader.stats
    assert stats.missing_spectra == 4


def test_annotator_train_rt():

    id_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet.idparquet"
    )

    mzml_file = (
        TESTS_DIR / "test_data" / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzML"
    )

    annotator = FeatureAnnotator(
        feature_generators="ms2pip,deeplc",
        ms2_tolerance=0.05,
        calibration_set_size=0.15,
        skip_deeplc_retrain=False,
        processes=2,
        log_level="INFO",
        spectrum_id_pattern="(.*)",
        psm_id_pattern="(.*)",
    )
    annotator.build_consensus_idparquet(id_file, mzml_file)
    annotator.annotate()

    output_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet_rescored.idparquet"
    )

    annotator.write_idparquet_file(output_file)


def test_idxmlreader_filtering():

    id_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet.idparquet"
    )

    mzml_file = (
        TESTS_DIR / "test_data" / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzML"
    )

    annotator = FeatureAnnotator(
        feature_generators="ms2pip,deeplc",
                                 only_features="DeepLC:RtDiff,DeepLC:PredictedRetentionTimeBest,Ms2pip:DotProd",
        ms2_tolerance=0.2,
        calibration_set_size=0.15,
        skip_deeplc_retrain=True,
        processes=2,
        log_level="INFO",
        spectrum_id_pattern="(.*)",
        psm_id_pattern="(.*)",
    )
    annotator.build_consensus_idparquet(id_file, mzml_file)
    annotator.annotate()

    output_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet_rescored.idparquet"
    )

    annotator.write_idparquet_file(output_file)


def test_idxmlreader_wrong_model():

    id_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet.idparquet"
    )

    mzml_file = (
        TESTS_DIR / "test_data" / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzML"
    )

    annotator = FeatureAnnotator(
        feature_generators="ms2pip,deeplc",
                                 only_features="DeepLC:RtDiff,DeepLC:PredictedRetentionTimeBest,Ms2pip:DotProd",
        ms2_tolerance=0.2,
        calibration_set_size=0.15,
        skip_deeplc_retrain=True,
        processes=2,
        log_level="INFO",
        spectrum_id_pattern="(.*)",
        psm_id_pattern="(.*)",
    )
    annotator.build_consensus_idparquet(id_file, mzml_file)
    annotator.annotate()

    output_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet_rescored.idparquet"
    )

    annotator.write_idparquet_file(output_file)


def test_spectrum2feature_file():
    idxml_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet.idparquet"
    )
    mzml_file = (
        TESTS_DIR / "test_data" / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzML"
    )
    output_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet_snr.idparquet"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "spectrum2feature",
            "--mzml",
            mzml_file,
            "--idparquet",
            idxml_file,
            "--output",
            output_file,
        ],
    )

    assert result.exit_code == 0


# test for the add_sage_feature command in cli
def test_add_sage_feature_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["sage2feature", "--help"])

    assert result.exit_code == 0


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0


@pytest.mark.skip(reason="This is for local test in big datasets, skipping for now")
def test_local_file():
    runner = CliRunner()
    local_folder = TESTS_DIR / "test_data" / "dae1cb16fb57893b94bfcb731b2bf7"
    result = runner.invoke(
        cli,
        [
            "msrescore2feature",
            "--idparquet",
            "{}/UPS1_50amol_R1_comet.idparquet".format(local_folder),
            "--mzml",
            "{}/UPS1_50amol_R1_converted.mzML".format(local_folder),
            "--processes",
            "2",
            "--ms2pip_model",
            "CID",
            "--feature_generators",
            "ms2pip,deeplc",
            "--output",
            "{}/UPS1_50amol_R1_rescored.idparquet".format(local_folder),
            "--ms2_tolerance",
            "0.05",
        ],
    )
    assert result.exit_code == 0

def test_psm_clean():
    output_file = (
        TESTS_DIR
        / "test_data"
        / "TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet_clean.idparquet"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "psm_feature_clean",
            "--idparquet",
            "{}/test_data/TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01_comet.idparquet".format(
                TESTS_DIR
            ),
            "--mzml",
            "{}/test_data/TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzML".format(
                TESTS_DIR
            ),
            "--output",
            output_file,
        ],
    )
    assert result.exit_code == 0


def test_download_models_help():
    """Test that download_models command is accessible and shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["download_models", "--help"])
    
    assert result.exit_code == 0
    assert "Download all required models" in result.output or "download_models" in result.output


@pytest.mark.skip(reason="Requires internet connection and model downloads, skip for CI")
def test_download_models_ms2pip_only():
    """Test downloading only MS2PIP models."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "download_models",
            "--models",
            "ms2pip",
            "--log_level",
            "info",
        ],
    )
    assert result.exit_code == 0