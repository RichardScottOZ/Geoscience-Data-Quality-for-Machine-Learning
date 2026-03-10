"""Geophysical survey quality assessment.

Functions for fetching, filtering, and preparing geophysical survey metadata
from services such as the Geoscience Australia GADDS WFS, and for assessing
data quality based on survey parameters like line spacing.

Based on analysis patterns from the Magnetic-Survey-Quality and
Radiometric-Survey-Quality notebooks.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
import shapely
import shapely.ops
from shapely import wkt

#: Default Geoscience Australia GADDS WFS URL for survey metadata.
GA_SURVEY_WFS_URL = (
    "https://services.ga.gov.au/gis/geophysical-surveys/wms"
    "?request=GetFeature&service=WFS&version=1.1.0"
    "&outputFormat=csv&typeName=gadds:geophysical_survey_datasets"
)


def fetch_ga_survey_metadata(url: Optional[str] = None) -> pd.DataFrame:
    """Fetch geophysical survey metadata from Geoscience Australia WFS.

    Parameters
    ----------
    url : str, optional
        WFS URL to fetch survey data from.  Defaults to
        :data:`GA_SURVEY_WFS_URL`.

    Returns
    -------
    pandas.DataFrame
        Survey metadata table.
    """
    if url is None:
        url = GA_SURVEY_WFS_URL
    return pd.read_csv(url)


def filter_surveys(
    surveys: pd.DataFrame,
    measure_type: str,
    dataset_type: Optional[str] = None,
) -> pd.DataFrame:
    """Filter surveys by measure type and optionally dataset type.

    Parameters
    ----------
    surveys : pandas.DataFrame
        Survey metadata table, e.g. from :func:`fetch_ga_survey_metadata`.
    measure_type : str
        Measurement type to filter on (e.g. ``"magnetic"``,
        ``"radiometric"``, ``"gravity"``).
    dataset_type : str, optional
        Dataset type to further filter on (e.g. ``"line"``, ``"grid"``).

    Returns
    -------
    pandas.DataFrame
        Filtered subset of surveys.
    """
    filtered = surveys[surveys["measure_type"] == measure_type].copy()
    if dataset_type is not None:
        filtered = filtered[filtered["dataset_type"] == dataset_type].copy()
    return filtered


def fix_survey_geometry(
    df: pd.DataFrame,
    swap_coordinates: bool = True,
    source_crs: str = "EPSG:4283",
    target_crs: str = "EPSG:4326",
    geometry_column: str = "geometry",
) -> gpd.GeoDataFrame:
    """Fix survey geometry and create a properly projected GeoDataFrame.

    Handles common issues with survey geometries from WFS services:

    * Parses WKT geometry strings into Shapely geometry objects.
    * Swaps x/y coordinates when they are transposed (common in WFS
      responses from some services).
    * Sets the source CRS and reprojects to the target CRS.

    Parameters
    ----------
    df : pandas.DataFrame
        Survey data with a geometry column containing WKT strings or
        Shapely geometry objects.
    swap_coordinates : bool
        Whether to swap x and y coordinates. Default ``True``.
    source_crs : str
        Source coordinate reference system. Default ``"EPSG:4283"``
        (GDA94).
    target_crs : str
        Target coordinate reference system. Default ``"EPSG:4326"``
        (WGS84).
    geometry_column : str
        Name of the geometry column. Default ``"geometry"``.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with corrected geometry and CRS.
    """
    df = df.copy()

    # Parse WKT strings if needed
    if len(df) > 0 and isinstance(df[geometry_column].iloc[0], str):
        df[geometry_column] = df[geometry_column].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, crs=source_crs, geometry=geometry_column)

    if swap_coordinates:
        gdf.geometry = gdf.geometry.map(
            lambda geom: shapely.ops.transform(lambda x, y: (y, x), geom)
        )
        # Re-set CRS after coordinate swap (doesn't change values)
        gdf = gdf.set_crs(source_crs, allow_override=True)

    gdf = gdf.to_crs(target_crs)
    return gdf
