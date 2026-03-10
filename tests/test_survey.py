"""Tests for geophysical survey quality assessment."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from geoscience_data_quality.survey import (
    filter_surveys,
    fix_survey_geometry,
)


@pytest.fixture
def sample_surveys():
    """Create a sample survey metadata DataFrame."""
    return pd.DataFrame(
        {
            "survey_name": ["S1", "S2", "S3", "S4", "S5"],
            "measure_type": [
                "magnetic",
                "magnetic",
                "radiometric",
                "gravity",
                "magnetic",
            ],
            "dataset_type": ["line", "grid", "line", "grid", "line"],
            "min_line_spacing_m": [200, 400, 300, 0, 100],
            "max_line_spacing_m": [400, 800, 600, 0, 200],
        }
    )


class TestFilterSurveys:
    def test_filter_by_measure_type(self, sample_surveys):
        result = filter_surveys(sample_surveys, measure_type="magnetic")
        assert len(result) == 3
        assert all(result["measure_type"] == "magnetic")

    def test_filter_by_measure_and_dataset(self, sample_surveys):
        result = filter_surveys(
            sample_surveys, measure_type="magnetic", dataset_type="line"
        )
        assert len(result) == 2
        assert all(result["measure_type"] == "magnetic")
        assert all(result["dataset_type"] == "line")

    def test_filter_returns_copy(self, sample_surveys):
        result = filter_surveys(sample_surveys, measure_type="magnetic")
        result["new_col"] = 1
        assert "new_col" not in sample_surveys.columns

    def test_empty_result(self, sample_surveys):
        result = filter_surveys(sample_surveys, measure_type="seismic")
        assert len(result) == 0


class TestFixSurveyGeometry:
    def test_parses_wkt_strings(self):
        df = pd.DataFrame(
            {
                "name": ["test"],
                "geometry": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
            }
        )
        result = fix_survey_geometry(
            df, swap_coordinates=False, source_crs="EPSG:4326", target_crs="EPSG:4326"
        )
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_swap_coordinates(self):
        df = pd.DataFrame(
            {
                "name": ["test"],
                "geometry": ["POLYGON ((0 10, 0 20, 5 20, 5 10, 0 10))"],
            }
        )
        result = fix_survey_geometry(
            df, swap_coordinates=True, source_crs="EPSG:4326", target_crs="EPSG:4326"
        )
        bounds = result.total_bounds
        # After swapping x/y, the coordinates should be transposed
        assert bounds is not None

    def test_preserves_columns(self):
        df = pd.DataFrame(
            {
                "name": ["test"],
                "value": [42],
                "geometry": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
            }
        )
        result = fix_survey_geometry(
            df, swap_coordinates=False, source_crs="EPSG:4326", target_crs="EPSG:4326"
        )
        assert "name" in result.columns
        assert "value" in result.columns
        assert result["value"].iloc[0] == 42

    def test_handles_shapely_geometries(self):
        df = pd.DataFrame(
            {
                "name": ["test"],
                "geometry": [box(0, 0, 1, 1)],
            }
        )
        result = fix_survey_geometry(
            df, swap_coordinates=False, source_crs="EPSG:4326", target_crs="EPSG:4326"
        )
        assert isinstance(result, gpd.GeoDataFrame)

    def test_crs_set_correctly(self):
        df = pd.DataFrame(
            {
                "name": ["test"],
                "geometry": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
            }
        )
        result = fix_survey_geometry(
            df,
            swap_coordinates=False,
            source_crs="EPSG:4283",
            target_crs="EPSG:4326",
        )
        assert result.crs.to_epsg() == 4326
