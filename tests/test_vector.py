"""Tests for vector data quality analysis."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from geoscience_data_quality.vector import (
    DEFAULT_QUALITY_FIELDS,
    analyze_quality_fields,
    get_quality_summary,
)


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame with quality fields."""
    return gpd.GeoDataFrame(
        {
            "confidence": ["high", "low", "high", "medium", "high"],
            "obsmethod": ["mapped", "mapped", "inferred", "mapped", "compiled"],
            "metadata": ["source_a", "source_a", "source_b", "source_a", None],
            "posacc_m": [100.0, 250.0, 500.0, 100.0, 1000.0],
            "geometry": [Point(0, 0)] * 5,
        }
    )


class TestAnalyzeQualityFields:
    def test_returns_dict(self, sample_gdf):
        result = analyze_quality_fields(sample_gdf)
        assert isinstance(result, dict)

    def test_default_fields(self, sample_gdf):
        result = analyze_quality_fields(sample_gdf)
        for field in DEFAULT_QUALITY_FIELDS:
            assert field in result

    def test_custom_fields(self, sample_gdf):
        result = analyze_quality_fields(sample_gdf, fields=["confidence"])
        assert "confidence" in result
        assert "obsmethod" not in result

    def test_value_counts_correct(self, sample_gdf):
        result = analyze_quality_fields(sample_gdf, fields=["confidence"])
        assert result["confidence"]["high"] == 3
        assert result["confidence"]["low"] == 1
        assert result["confidence"]["medium"] == 1

    def test_missing_field_skipped(self, sample_gdf):
        result = analyze_quality_fields(sample_gdf, fields=["nonexistent"])
        assert "nonexistent" not in result

    def test_empty_gdf(self):
        gdf = gpd.GeoDataFrame({"confidence": pd.Series(dtype="str")})
        result = analyze_quality_fields(gdf, fields=["confidence"])
        assert len(result["confidence"]) == 0


class TestGetQualitySummary:
    def test_returns_dataframe(self, sample_gdf):
        result = get_quality_summary(sample_gdf)
        assert isinstance(result, pd.DataFrame)

    def test_summary_columns(self, sample_gdf):
        result = get_quality_summary(sample_gdf)
        expected_cols = {"count", "unique", "top", "freq", "completeness"}
        assert set(result.columns) == expected_cols

    def test_completeness(self, sample_gdf):
        result = get_quality_summary(sample_gdf, fields=["metadata"])
        # 4 out of 5 non-null
        assert result.loc["metadata", "completeness"] == pytest.approx(0.8)

    def test_top_value(self, sample_gdf):
        result = get_quality_summary(sample_gdf, fields=["confidence"])
        assert result.loc["confidence", "top"] == "high"
        assert result.loc["confidence", "freq"] == 3

    def test_custom_fields(self, sample_gdf):
        result = get_quality_summary(sample_gdf, fields=["posacc_m"])
        assert "posacc_m" in result.index

    def test_empty_gdf(self):
        gdf = gpd.GeoDataFrame({"confidence": pd.Series(dtype="str")})
        result = get_quality_summary(gdf, fields=["confidence"])
        assert len(result) == 0 or result.loc["confidence", "count"] == 0
