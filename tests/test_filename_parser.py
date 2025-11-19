import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = REPO_ROOT / "anytimes" / "gui" / "filename_parser.py"


@pytest.fixture(scope="module")
def filename_parser():
    spec = importlib.util.spec_from_file_location("filename_parser", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def test_parse_dir_with_decimal_without_extension(filename_parser):
    data = filename_parser.parse_general_filename(
        "FSRU_Dir202.5_Hs2_3_Tp7_5_Uw12_Uc0_15_prob0_0118_fatigue"
    )
    assert pytest.approx(data["Dir"]) == 202.5
    assert pytest.approx(data["Hs"]) == 2.3
    assert pytest.approx(data["Tp"]) == 7.5
    assert pytest.approx(data["Uc"]) == 0.15
    assert pytest.approx(data["prob"]) == 0.0118


def test_parse_dir_with_decimal_and_extension(filename_parser):
    data = filename_parser.parse_general_filename(
        "FSRU_Dir202.5_Hs0_3_Tp6_Uw0_5_Uc0_1_prob0_0047_fatigue.ts"
    )
    assert pytest.approx(data["Dir"]) == 202.5
    assert pytest.approx(data["Hs"]) == 0.3
    assert pytest.approx(data["Tp"]) == 6


def test_choose_parse_target_prefers_file_name(filename_parser):
    pick = filename_parser.choose_parse_target("file_a.ts", "uniq-only")
    assert pick == "file_a.ts"

    pick = filename_parser.choose_parse_target("", "uniq-only")
    assert pick == "uniq-only"

    pick = filename_parser.choose_parse_target(None, "", None)
    assert pick == ""


def test_exposure_hours_from_probability(filename_parser):
    hours = filename_parser.exposure_hours_from_name(
        "FSRU_Dir0_Hs1_8_Tp8_Uw10_5_Uc0_15_prob0_0006_QTF_loaded_fatigue",
        design_life_years=25,
    )
    assert pytest.approx(hours) == 25 * 365 * 24 * 0.0006


def test_exposure_hours_from_direct_identifier(filename_parser):
    hours = filename_parser.exposure_hours_from_name(
        "case_exposure_time12_5_channel", design_life_years=1
    )
    assert pytest.approx(hours) == 12.5
