"""Vector data quality analysis for geological datasets.

Provides functions to analyze quality-related fields in geospatial vector
data, such as confidence, observation method, positional accuracy, and
metadata fields commonly found in geological map datasets.

Based on analysis patterns from the Geology-Quality-1M notebook.
"""

from __future__ import annotations

from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd


#: Default quality fields commonly found in geological vector datasets.
DEFAULT_QUALITY_FIELDS = [
    "confidence",
    "obsmethod",
    "metadata",
    "posacc_m",
]


def analyze_quality_fields(
    gdf: gpd.GeoDataFrame,
    fields: Optional[Sequence[str]] = None,
) -> dict[str, pd.Series]:
    """Analyze quality-related fields in a GeoDataFrame.

    Returns value counts for each specified field, providing insight into
    the distribution of quality indicators across the dataset.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geodataframe with quality fields.
    fields : sequence of str, optional
        Column names to analyze. If ``None``, all columns from
        :data:`DEFAULT_QUALITY_FIELDS` that exist in *gdf* are used.

    Returns
    -------
    dict[str, pandas.Series]
        Mapping of field name to value counts for that field.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame({"confidence": ["high", "low", "high"]})
    >>> result = analyze_quality_fields(gdf, fields=["confidence"])
    >>> result["confidence"]["high"]
    2
    """
    if fields is None:
        fields = [f for f in DEFAULT_QUALITY_FIELDS if f in gdf.columns]

    results = {}
    for field in fields:
        if field in gdf.columns:
            results[field] = gdf[field].value_counts()

    return results


def get_quality_summary(
    gdf: gpd.GeoDataFrame,
    fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Get a summary of quality metrics for a geodataframe.

    Produces a summary table with feature count, number of unique values,
    the most common value, and the fraction of non-null entries for each
    quality field.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geodataframe with quality fields.
    fields : sequence of str, optional
        Column names to summarize. If ``None``, all columns from
        :data:`DEFAULT_QUALITY_FIELDS` that exist in *gdf* are used.

    Returns
    -------
    pandas.DataFrame
        Summary table indexed by field name with columns:
        ``count``, ``unique``, ``top``, ``freq``, ``completeness``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame({
    ...     "confidence": ["high", "low", "high", None],
    ...     "obsmethod": ["mapped", "mapped", "inferred", "mapped"],
    ... })
    >>> summary = get_quality_summary(gdf)
    >>> summary.loc["confidence", "completeness"]
    0.75
    """
    if fields is None:
        fields = [f for f in DEFAULT_QUALITY_FIELDS if f in gdf.columns]

    rows = []
    for field in fields:
        if field not in gdf.columns:
            continue
        col = gdf[field]
        vc = col.value_counts()
        top_value = vc.index[0] if len(vc) > 0 else None
        top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
        rows.append(
            {
                "field": field,
                "count": int(col.count()),
                "unique": int(col.nunique()),
                "top": top_value,
                "freq": top_freq,
                "completeness": col.count() / len(gdf) if len(gdf) > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows).set_index("field")
