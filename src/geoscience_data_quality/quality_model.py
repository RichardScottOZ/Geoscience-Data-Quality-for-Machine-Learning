"""Quality scoring model for geoscience datasets.

Functions for loading, computing, and managing data quality scores for
geoscience datasets used in machine learning.  The scoring framework
considers factors such as resolution, data presence, and expert ratings.

Based on the DataQuality_Models scoring system and the Model-Quality-Map
notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def load_quality_model(path: Union[str, Path]) -> pd.DataFrame:
    """Load a data quality model from a CSV or Excel file.

    Parameters
    ----------
    path : str or Path
        Path to a CSV (``.csv``) or Excel (``.xlsx`` / ``.xls``) file
        containing the quality model.

    Returns
    -------
    pandas.DataFrame
        Quality model table.

    Raises
    ------
    ValueError
        If the file extension is not recognized.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. Use .csv, .xlsx, or .xls."
        )


def compute_resolution_score(
    resolution: Union[float, np.ndarray, pd.Series],
    invert: bool = True,
) -> Union[float, np.ndarray, pd.Series]:
    """Compute a resolution score from raw resolution values.

    Smaller resolution values (finer resolution) yield higher scores.
    The score is calculated as ``1 / resolution`` when *invert* is
    ``True``.

    Parameters
    ----------
    resolution : float, array-like, or Series
        Raw resolution value(s), e.g. pixel size in metres.
    invert : bool
        If ``True`` (default), compute score as ``1 / resolution``.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Resolution score(s).
    """
    if invert:
        return 1.0 / resolution
    return resolution


def compute_final_score(
    score: Union[float, np.ndarray, pd.Series],
    presence: Union[float, np.ndarray, pd.Series],
    resolution_score: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Compute the final quality score.

    The final score combines an expert/category score, a data presence
    fraction, and a resolution score as::

        final = score * presence * resolution_score * 100 / 3

    This normalises the result so that a dataset with the maximum score
    of 3, full presence (1.0), and a resolution score of 1.0 would
    receive a final score of 100.

    Parameters
    ----------
    score : float, array-like, or Series
        Expert or category score (typically 1–3).
    presence : float, array-like, or Series
        Fraction of data that is present (0–1).
    resolution_score : float, array-like, or Series
        Resolution score (e.g. from :func:`compute_resolution_score`).

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Final quality score(s).
    """
    return score * presence * resolution_score * 100.0 / 3.0


def get_quality_by_domain(
    quality_df: pd.DataFrame,
    domain: str,
    domain_column: str = "Domain",
) -> pd.DataFrame:
    """Filter quality model entries by domain.

    Parameters
    ----------
    quality_df : pandas.DataFrame
        Quality model table.
    domain : str
        Domain name to filter by (e.g. ``"Geophysics"``,
        ``"Remote Sensing"``, ``"Geology"``).
    domain_column : str
        Name of the column containing domain labels.

    Returns
    -------
    pandas.DataFrame
        Filtered subset of the quality model.
    """
    return quality_df[quality_df[domain_column] == domain].copy()


def get_quality_by_subdomain(
    quality_df: pd.DataFrame,
    subdomain: str,
    subdomain_column: str = "Sub Domain",
) -> pd.DataFrame:
    """Filter quality model entries by sub-domain.

    Parameters
    ----------
    quality_df : pandas.DataFrame
        Quality model table.
    subdomain : str
        Sub-domain name to filter by (e.g. ``"Gravity"``,
        ``"Magnetic"``, ``"ASTER"``).
    subdomain_column : str
        Name of the column containing sub-domain labels.

    Returns
    -------
    pandas.DataFrame
        Filtered subset of the quality model.
    """
    return quality_df[quality_df[subdomain_column] == subdomain].copy()


def add_quality_scores(
    df: pd.DataFrame,
    score_column: str = "Scr",
    presence_column: str = "Pres",
    resolution_column: str = "Res",
    resolution_score_column: str = "Res Scr",
    final_column: str = "Final",
) -> pd.DataFrame:
    """Recompute resolution and final scores for a quality model table.

    Adds or overwrites the resolution score and final score columns by
    applying :func:`compute_resolution_score` and
    :func:`compute_final_score`.

    Parameters
    ----------
    df : pandas.DataFrame
        Quality model table with at least the score, presence, and
        resolution columns.
    score_column : str
        Column with expert/category scores.
    presence_column : str
        Column with data presence fractions.
    resolution_column : str
        Column with raw resolution values.
    resolution_score_column : str
        Column name for computed resolution scores.
    final_column : str
        Column name for computed final scores.

    Returns
    -------
    pandas.DataFrame
        Updated table with recomputed scores.
    """
    df = df.copy()
    df[resolution_score_column] = compute_resolution_score(df[resolution_column])
    df[final_column] = compute_final_score(
        df[score_column], df[presence_column], df[resolution_score_column]
    )
    return df
