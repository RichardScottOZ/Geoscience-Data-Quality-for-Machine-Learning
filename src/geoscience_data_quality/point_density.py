"""Point density analysis for geophysical observations.

Functions for computing observation density from point data such as
gravity station locations, providing a spatial measure of data quality.

Based on the Gravity-Survey-Quality notebook which uses verde for block
reduction to compute points-per-pixel.

These functions require the optional ``verde`` dependency.  Install it
with::

    pip install geoscience-data-quality[gravity]
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_point_density(
    coordinates: tuple[np.ndarray, np.ndarray],
    spacing: float = 0.1,
    center_coordinates: bool = True,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Compute point density using block reduction.

    Divides the area into blocks of the given *spacing* and counts the
    number of points in each block.

    Parameters
    ----------
    coordinates : tuple of ndarray
        ``(longitude, latitude)`` arrays of observation locations.
    spacing : float
        Block size in degrees. Default ``0.1``.
    center_coordinates : bool
        If ``True``, return the centre of each block as the
        coordinates. Default ``True``.

    Returns
    -------
    coords : tuple of ndarray
        ``(longitude, latitude)`` of block centres.
    counts : ndarray
        Number of points in each block.

    Raises
    ------
    ImportError
        If ``verde`` is not installed.
    """
    try:
        import verde as vd
    except ImportError as exc:
        raise ImportError(
            "The 'verde' package is required for point density analysis. "
            "Install it with: pip install geoscience-data-quality[gravity]"
        ) from exc

    def _count(array: np.ndarray) -> int:
        return array.size

    # Create dummy data matching the coordinate arrays
    dummy_data = np.ones(coordinates[0].shape)

    coords, counts = vd.BlockReduce(
        _count,
        center_coordinates=center_coordinates,
        spacing=spacing,
    ).filter(coordinates, data=dummy_data)

    return coords, counts
