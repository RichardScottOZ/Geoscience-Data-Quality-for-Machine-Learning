"""Tests for quality scoring model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoscience_data_quality.quality_model import (
    add_quality_scores,
    compute_final_score,
    compute_resolution_score,
    get_quality_by_domain,
    get_quality_by_subdomain,
    load_quality_model,
)


@pytest.fixture
def sample_quality_df():
    """Create a sample quality model DataFrame."""
    return pd.DataFrame(
        {
            "Dataset": ["gravity-bouger", "magmap-tmi", "FeOH_Group_Content", "lith"],
            "Scr": [3, 3, 2, 1],
            "Pres": [1.0, 1.0, 1.0, 1.0],
            "Res": [800.0, 90.0, 90.0, 1000.0],
            "Res Scr": [0.00125, 0.0111, 0.0111, 0.001],
            "Final": [1.13, 10.0, 6.67, 0.3],
            "Domain": ["Geophysics", "Geophysics", "Remote Sensing", "Geology"],
            "Sub Domain": ["Gravity", "Magnetic", "ASTER", "Lithology"],
        }
    )


class TestLoadQualityModel:
    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(csv_path, index=False)
        result = load_quality_model(csv_path)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2"]

    @pytest.mark.skipif(
        not pd.io.common.import_optional_dependency("openpyxl", errors="ignore"),
        reason="openpyxl not installed",
    )
    def test_load_xlsx(self, tmp_path):
        xlsx_path = tmp_path / "test.xlsx"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_excel(xlsx_path, index=False)
        result = load_quality_model(xlsx_path)
        assert len(result) == 2

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_quality_model(path)


class TestComputeResolutionScore:
    def test_scalar(self):
        assert compute_resolution_score(100.0) == pytest.approx(0.01)

    def test_array(self):
        result = compute_resolution_score(np.array([100.0, 200.0]))
        np.testing.assert_allclose(result, [0.01, 0.005])

    def test_series(self):
        result = compute_resolution_score(pd.Series([90.0, 800.0]))
        assert result.iloc[0] == pytest.approx(1.0 / 90.0)
        assert result.iloc[1] == pytest.approx(1.0 / 800.0)

    def test_no_invert(self):
        assert compute_resolution_score(100.0, invert=False) == 100.0


class TestComputeFinalScore:
    def test_scalar(self):
        # score=3, presence=1.0, res_score=0.01 -> 3*1*0.01*100/3 = 1.0
        result = compute_final_score(3, 1.0, 0.01)
        assert result == pytest.approx(1.0)

    def test_max_score(self):
        # score=3, presence=1.0, res_score=1.0 -> 3*1*1*100/3 = 100
        result = compute_final_score(3, 1.0, 1.0)
        assert result == pytest.approx(100.0)

    def test_partial_presence(self):
        result = compute_final_score(3, 0.5, 1.0)
        assert result == pytest.approx(50.0)

    def test_array(self):
        scores = np.array([3, 2, 1])
        presence = np.array([1.0, 1.0, 0.5])
        res_score = np.array([0.01, 0.01, 0.01])
        result = compute_final_score(scores, presence, res_score)
        assert len(result) == 3


class TestGetQualityByDomain:
    def test_filter_geophysics(self, sample_quality_df):
        result = get_quality_by_domain(sample_quality_df, "Geophysics")
        assert len(result) == 2

    def test_filter_geology(self, sample_quality_df):
        result = get_quality_by_domain(sample_quality_df, "Geology")
        assert len(result) == 1

    def test_empty_result(self, sample_quality_df):
        result = get_quality_by_domain(sample_quality_df, "Topography")
        assert len(result) == 0


class TestGetQualityBySubdomain:
    def test_filter_gravity(self, sample_quality_df):
        result = get_quality_by_subdomain(sample_quality_df, "Gravity")
        assert len(result) == 1

    def test_filter_aster(self, sample_quality_df):
        result = get_quality_by_subdomain(sample_quality_df, "ASTER")
        assert len(result) == 1


class TestAddQualityScores:
    def test_recomputes_scores(self):
        df = pd.DataFrame(
            {
                "Scr": [3, 2],
                "Pres": [1.0, 0.5],
                "Res": [100.0, 200.0],
            }
        )
        result = add_quality_scores(df)
        assert "Res Scr" in result.columns
        assert "Final" in result.columns
        assert result["Res Scr"].iloc[0] == pytest.approx(0.01)

    def test_does_not_modify_original(self):
        df = pd.DataFrame(
            {
                "Scr": [3],
                "Pres": [1.0],
                "Res": [100.0],
            }
        )
        _ = add_quality_scores(df)
        assert "Res Scr" not in df.columns
