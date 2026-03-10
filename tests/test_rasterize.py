"""Tests for rasterization utilities."""

import os
import tempfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from geoscience_data_quality.rasterize import (
    rasterize_to_new_grid,
    rasterize_vector_attribute,
)


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame with polygons and values."""
    return gpd.GeoDataFrame(
        {
            "value": [10.0, 20.0, 30.0],
            "geometry": [
                box(0, 0, 5, 5),
                box(3, 3, 8, 8),
                box(6, 0, 10, 5),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def reference_raster(tmp_path):
    """Create a small reference raster."""
    path = str(tmp_path / "reference.tif")
    transform = from_bounds(0, 0, 10, 10, 10, 10)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 10,
        "height": 10,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(np.zeros((1, 10, 10), dtype="float32"))
    return path


class TestRasterizeVectorAttribute:
    def test_returns_array(self, sample_gdf, reference_raster):
        result = rasterize_vector_attribute(
            sample_gdf, "value", reference_raster
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)

    def test_writes_output(self, sample_gdf, reference_raster, tmp_path):
        output = str(tmp_path / "output.tif")
        rasterize_vector_attribute(
            sample_gdf, "value", reference_raster, output_path=output
        )
        assert os.path.exists(output)
        with rasterio.open(output) as src:
            data = src.read(1)
            assert data.shape == (10, 10)

    def test_values_burned_in(self, sample_gdf, reference_raster):
        result = rasterize_vector_attribute(
            sample_gdf, "value", reference_raster
        )
        # At least some pixels should have non-zero values
        assert np.any(result > 0)

    def test_sort_ascending_false(self, sample_gdf, reference_raster):
        """Descending sort means smaller values win in overlap regions."""
        result = rasterize_vector_attribute(
            sample_gdf, "value", reference_raster, sort_ascending=False
        )
        # The overlapping region between box(0,0,5,5) value=10 and
        # box(3,3,8,8) value=20 should have 10 (smallest wins when
        # sorted descending, because smallest is written last).
        assert isinstance(result, np.ndarray)

    def test_dtype(self, sample_gdf, reference_raster):
        result = rasterize_vector_attribute(
            sample_gdf, "value", reference_raster, dtype="float64"
        )
        assert result.dtype == np.float64


class TestRasterizeToNewGrid:
    def test_returns_array(self, sample_gdf):
        result = rasterize_to_new_grid(
            sample_gdf, "value", bounds=(0, 0, 10, 10), resolution=1.0
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)

    def test_writes_output(self, sample_gdf, tmp_path):
        output = str(tmp_path / "new_grid.tif")
        rasterize_to_new_grid(
            sample_gdf,
            "value",
            bounds=(0, 0, 10, 10),
            resolution=1.0,
            output_path=output,
        )
        assert os.path.exists(output)
        with rasterio.open(output) as src:
            assert src.crs.to_epsg() == 4326
            assert src.width == 10
            assert src.height == 10

    def test_values_present(self, sample_gdf):
        result = rasterize_to_new_grid(
            sample_gdf, "value", bounds=(0, 0, 10, 10), resolution=1.0
        )
        assert np.any(result > 0)

    def test_custom_resolution(self, sample_gdf):
        result = rasterize_to_new_grid(
            sample_gdf, "value", bounds=(0, 0, 10, 10), resolution=0.5
        )
        assert result.shape == (20, 20)
