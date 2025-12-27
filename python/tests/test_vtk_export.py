"""Tests for VTK export functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import biotransport as bt


class TestWriteVtk:
    """Tests for write_vtk function."""

    def test_1d_mesh_export(self):
        """Test VTK export for 1D mesh."""
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        x = bt.x_nodes(mesh)
        temperature = 300 + 100 * x  # Linear temperature profile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_1d.vtk"
            result = bt.write_vtk(mesh, {"temperature": temperature}, filepath)

            assert result.exists()
            assert result.suffix == ".vtk"

            # Verify file contents
            content = result.read_text()
            assert "# vtk DataFile Version 3.0" in content
            assert "STRUCTURED_POINTS" in content
            assert "DIMENSIONS 11 1 1" in content
            assert "SCALARS temperature double" in content
            assert "POINT_DATA 11" in content

    def test_2d_mesh_export(self):
        """Test VTK export for 2D mesh."""
        mesh = bt.StructuredMesh(5, 5, 0.0, 1.0, 0.0, 1.0)
        x, y = bt.xy_grid(mesh)
        concentration = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_2d.vtk"
            result = bt.write_vtk(mesh, {"concentration": concentration}, filepath)

            assert result.exists()

            content = result.read_text()
            assert "STRUCTURED_POINTS" in content
            assert "DIMENSIONS 6 6 1" in content
            assert "POINT_DATA 36" in content
            assert "SCALARS concentration double" in content

    def test_multiple_fields(self):
        """Test exporting multiple scalar fields."""
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        n = mesh.num_nodes()

        fields = {
            "temperature": np.linspace(300, 400, n),
            "pressure": np.linspace(101325, 101400, n),
            "velocity": np.zeros(n),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "multi_field.vtk"
            result = bt.write_vtk(mesh, fields, filepath)

            content = result.read_text()
            assert "SCALARS temperature double" in content
            assert "SCALARS pressure double" in content
            assert "SCALARS velocity double" in content

    def test_auto_adds_vtk_extension(self):
        """Test that .vtk extension is added automatically."""
        mesh = bt.StructuredMesh(5, 0.0, 1.0)
        data = np.zeros(mesh.num_nodes())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "no_extension"
            result = bt.write_vtk(mesh, {"field": data}, filepath)

            assert result.suffix == ".vtk"
            assert result.name == "no_extension.vtk"

    def test_creates_parent_directories(self):
        """Test that parent directories are created automatically."""
        mesh = bt.StructuredMesh(5, 0.0, 1.0)
        data = np.zeros(mesh.num_nodes())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dirs" / "output.vtk"
            result = bt.write_vtk(mesh, {"field": data}, filepath)

            assert result.exists()
            assert result.parent.exists()

    def test_field_size_mismatch_raises(self):
        """Test that mismatched field size raises ValueError."""
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        wrong_size_data = np.zeros(5)  # Should be 11

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.vtk"
            with pytest.raises(ValueError, match="has 5 values"):
                bt.write_vtk(mesh, {"bad_field": wrong_size_data}, filepath)

    def test_empty_fields_ok(self):
        """Test that empty fields dict is allowed."""
        mesh = bt.StructuredMesh(5, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "no_fields.vtk"
            result = bt.write_vtk(mesh, {}, filepath)

            content = result.read_text()
            assert "POINT_DATA" not in content

    def test_sanitizes_field_names(self):
        """Test that field names with spaces/dashes are sanitized."""
        mesh = bt.StructuredMesh(5, 0.0, 1.0)
        data = np.zeros(mesh.num_nodes())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "sanitize.vtk"
            result = bt.write_vtk(
                mesh, {"my field-name": data, "another one": data}, filepath
            )

            content = result.read_text()
            assert "SCALARS my_field_name double" in content
            assert "SCALARS another_one double" in content


class TestWriteVtkSeries:
    """Tests for write_vtk_series function."""

    def test_time_series_export(self):
        """Test exporting a time series to VTK with PVD collection."""
        mesh = bt.StructuredMesh(10, 0.0, 1.0)
        n = mesh.num_nodes()

        # Create time series data
        snapshots = [
            (0.0, {"concentration": np.ones(n) * 1.0}),
            (0.1, {"concentration": np.ones(n) * 0.9}),
            (0.2, {"concentration": np.ones(n) * 0.8}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "series"
            pvd_path = bt.write_vtk_series(mesh, snapshots, base_path)

            # Check PVD file exists
            assert pvd_path.exists()
            assert pvd_path.suffix == ".pvd"

            # Check individual VTK files exist
            for i in range(3):
                vtk_file = Path(tmpdir) / f"series_{i:04d}.vtk"
                assert vtk_file.exists()

            # Check PVD content
            pvd_content = pvd_path.read_text()
            assert '<?xml version="1.0"?>' in pvd_content
            assert '<VTKFile type="Collection"' in pvd_content
            assert 'timestep="0.0"' in pvd_content
            assert 'timestep="0.1"' in pvd_content
            assert 'timestep="0.2"' in pvd_content

    def test_empty_series(self):
        """Test that empty time series works."""
        mesh = bt.StructuredMesh(5, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "empty"
            pvd_path = bt.write_vtk_series(mesh, [], base_path)

            assert pvd_path.exists()
            content = pvd_path.read_text()
            assert "<Collection>" in content
            assert "</Collection>" in content


class TestCylindricalMeshVtk:
    """Tests for VTK export with cylindrical meshes."""

    def test_cylindrical_mesh_export(self):
        """Test VTK export for 2D axisymmetric (r-z) cylindrical mesh."""
        # 2D axisymmetric mesh: nr=5, nz=10, r in [0.01, 0.1], z in [0, 0.2]
        mesh = bt.CylindricalMesh(5, 10, 0.01, 0.1, 0.0, 0.2)
        n = mesh.num_nodes()
        pressure = np.linspace(0, 100, n)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "cylindrical.vtk"
            result = bt.write_vtk(mesh, {"pressure": pressure}, filepath)

            assert result.exists()
            content = result.read_text()
            assert "STRUCTURED_GRID" in content
            assert "POINTS" in content
