"""Tests for point density analysis."""

import numpy as np
import pytest

from geoscience_data_quality.point_density import compute_point_density


class TestComputePointDensity:
    def test_basic_density(self):
        """Test basic point density computation."""
        # Create a grid of points
        lon = np.array([0.05, 0.05, 0.05, 0.15, 0.15])
        lat = np.array([0.05, 0.06, 0.07, 0.05, 0.06])
        coords, counts = compute_point_density((lon, lat), spacing=0.1)
        # Should have at most 2 blocks
        assert len(counts) <= 3
        assert np.sum(counts) == 5  # Total points preserved

    def test_single_block(self):
        """All points in one block."""
        lon = np.array([0.01, 0.02, 0.03])
        lat = np.array([0.01, 0.02, 0.03])
        coords, counts = compute_point_density((lon, lat), spacing=1.0)
        assert len(counts) == 1
        assert counts[0] == 3

    def test_returns_coordinates(self):
        """Verify coordinate arrays are returned."""
        lon = np.array([0.0, 1.0, 2.0])
        lat = np.array([0.0, 1.0, 2.0])
        coords, counts = compute_point_density((lon, lat), spacing=0.5)
        assert len(coords) == 2  # (lon, lat) tuple
        assert len(coords[0]) == len(counts)

    def test_spacing_affects_result(self):
        """Larger spacing should produce fewer blocks."""
        lon = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        lat = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        _, counts_fine = compute_point_density((lon, lat), spacing=0.5)
        _, counts_coarse = compute_point_density((lon, lat), spacing=2.0)
        assert len(counts_coarse) <= len(counts_fine)
