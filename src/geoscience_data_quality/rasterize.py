"""Rasterization utilities for geoscience data quality.

Functions for converting vector quality attributes to raster grids,
a common operation across geological and geophysical quality assessments.

Based on the rasterization patterns used across the Geology-Quality-1M,
Magnetic-Survey-Quality, and Radiometric-Survey-Quality notebooks.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds


def rasterize_vector_attribute(
    gdf: gpd.GeoDataFrame,
    column: str,
    reference_raster: str,
    output_path: Optional[str] = None,
    all_touched: bool = True,
    dtype: str = "float32",
    sort_ascending: Optional[bool] = None,
) -> np.ndarray:
    """Rasterize a vector attribute using a reference raster grid.

    Converts a column of a GeoDataFrame to a raster array aligned to a
    reference raster, optionally writing the result to a GeoTIFF file.

    When *sort_ascending* is set, the GeoDataFrame is sorted by *column*
    before rasterization.  Using ``sort_ascending=False`` (descending) causes
    the **smallest** value to be written last and therefore retained where
    geometries overlap, which is a common pattern for "best quality wins"
    rasterization.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Vector data containing the attribute to rasterize.
    column : str
        Column name to rasterize.
    reference_raster : str
        Path to a reference raster whose grid (shape, transform, CRS) will
        be used for the output.
    output_path : str, optional
        If given, the rasterized array is written to this GeoTIFF path.
    all_touched : bool
        If ``True``, all pixels touched by a geometry are burned in,
        not just those whose centre falls inside the geometry.
    dtype : str
        NumPy dtype string for the output array.
    sort_ascending : bool, optional
        If not ``None``, sort *gdf* by *column* in the specified order
        before rasterization.  ``False`` (descending) means the smallest
        value wins in overlapping areas.

    Returns
    -------
    numpy.ndarray
        Rasterized 2-D array.
    """
    if sort_ascending is not None:
        gdf = gdf.sort_values(by=column, ascending=sort_ascending)

    with rasterio.open(reference_raster) as src:
        shapes = (
            (geom, value)
            for geom, value in zip(gdf.geometry, gdf[column])
        )

        result = rio_rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            all_touched=all_touched,
            dtype=dtype,
        )

        if output_path is not None:
            meta = src.meta.copy()
            meta.update(dtype=dtype)
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(result, indexes=1)

    return result


def rasterize_to_new_grid(
    gdf: gpd.GeoDataFrame,
    column: str,
    bounds: tuple[float, float, float, float],
    resolution: float,
    output_path: Optional[str] = None,
    all_touched: bool = True,
    dtype: str = "float32",
    crs: str = "EPSG:4326",
    sort_ascending: Optional[bool] = None,
) -> np.ndarray:
    """Rasterize a vector attribute to a new raster grid defined by bounds.

    Creates a new raster grid from the specified bounds and resolution,
    then rasterizes the vector attribute onto it.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Vector data containing the attribute to rasterize.
    column : str
        Column name to rasterize.
    bounds : tuple of float
        Bounding box as ``(minx, miny, maxx, maxy)``.
    resolution : float
        Pixel size in the units of *crs*.
    output_path : str, optional
        If given, the result is written to this GeoTIFF path.
    all_touched : bool
        Burn all touched pixels.
    dtype : str
        Output dtype.
    crs : str
        Coordinate reference system for the output raster.
    sort_ascending : bool, optional
        Sort order for the attribute column before rasterization.

    Returns
    -------
    numpy.ndarray
        Rasterized 2-D array.
    """
    minx, miny, maxx, maxy = bounds
    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    if sort_ascending is not None:
        gdf = gdf.sort_values(by=column, ascending=sort_ascending)

    shapes = (
        (geom, value)
        for geom, value in zip(gdf.geometry, gdf[column])
    )

    result = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        all_touched=all_touched,
        dtype=dtype,
    )

    if output_path is not None:
        meta = {
            "driver": "GTiff",
            "dtype": dtype,
            "width": width,
            "height": height,
            "count": 1,
            "crs": crs,
            "transform": transform,
        }
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(result, indexes=1)

    return result
