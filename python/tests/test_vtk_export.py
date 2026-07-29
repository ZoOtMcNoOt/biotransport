"""Tests for VTK export functionality."""

from __future__ import annotations

import math
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

    def test_wrong_2d_shape_rejected_even_when_size_matches(self):
        mesh = bt.StructuredMesh(3, 2, 0.0, 1.0, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad_shape.vtk"
            with pytest.raises(ValueError, match="must have shape"):
                bt.write_vtk(mesh, {"field": np.ones((4, 3))}, filepath)

    @pytest.mark.parametrize(
        ("data", "error", "message"),
        [
            (1.0, ValueError, "flat or have shape"),
            ([0.0, 1.0, np.nan], ValueError, "finite"),
            ([object(), object(), object()], TypeError, "numeric"),
            (
                np.array([1.0 + 2.0j, 2.0 + 0.0j, 3.0 - 1.0j]),
                TypeError,
                "complex",
            ),
            (np.array([True, False, True]), TypeError, "boolean"),
            (np.array(["1.0", "2.0", "3.0"]), TypeError, "text and object"),
            (
                np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False]),
                ValueError,
                "masked",
            ),
        ],
    )
    def test_invalid_field_data_fails_before_writing(self, data, error, message):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "bad.vtk"
            with pytest.raises(error, match=message):
                bt.write_vtk(mesh, {"field": data}, filepath)
            assert not filepath.parent.exists()

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

    def test_sanitizes_all_non_identifier_characters(self):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        data = np.zeros(mesh.num_nodes())
        with tempfile.TemporaryDirectory() as tmpdir:
            result = bt.write_vtk(
                mesh, {"drug/concentration (%)": data}, Path(tmpdir) / "names.vtk"
            )
            assert "SCALARS drug_concentration____ double" in result.read_text()

    def test_rejects_invalid_or_colliding_field_names(self):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        data = np.zeros(mesh.num_nodes())
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad_names.vtk"
            with pytest.raises(ValueError, match="must not be empty"):
                bt.write_vtk(mesh, {"": data}, filepath)
            with pytest.raises(ValueError, match="both sanitize"):
                bt.write_vtk(mesh, {"a-b": data, "a b": data}, filepath)
            with pytest.raises(TypeError, match="must be strings"):
                bt.write_vtk(mesh, {1: data}, filepath)
            assert not filepath.exists()

    @pytest.mark.parametrize(
        ("title", "message"),
        [
            ("bad\nheader", "newlines"),
            ("temperature \N{DEGREE SIGN}C", "ASCII"),
            ("", "must not be empty"),
        ],
    )
    def test_rejects_invalid_title_before_creating_directories(self, title, message):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "result.vtk"
            with pytest.raises(ValueError, match=message):
                bt.write_vtk(mesh, {}, filepath, title=title)
            assert not filepath.parent.exists()

    def test_rejects_unsupported_mesh_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "unsupported.vtk"
            with pytest.raises(TypeError, match="StructuredMesh"):
                bt.write_vtk(object(), {}, filepath)

            mesh_3d = bt.CylindricalMesh(2, 8, 2, 0.1, 1.0, 0.0, 2 * math.pi, -1.0, 1.0)
            with pytest.raises(ValueError, match="full 3D cylindrical"):
                bt.write_vtk(mesh_3d, {}, filepath)

    def test_rejects_non_finite_mesh_geometry(self):
        mesh = bt.StructuredMesh(2, 0.0, np.nan)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad_mesh.vtk"
            with pytest.raises(ValueError, match="coordinates must be finite"):
                bt.write_vtk(mesh, {}, filepath)
            assert not filepath.exists()


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

    @pytest.mark.parametrize(
        "times",
        [
            [0.0, 0.0],
            [1.0, 0.5],
        ],
    )
    def test_series_requires_strictly_increasing_times(self, times):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        snapshots = [(time, {"field": np.zeros(3)}) for time in times]
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "nested" / "series"
            with pytest.raises(ValueError, match="strictly increasing"):
                bt.write_vtk_series(mesh, snapshots, base_path)
            assert not base_path.parent.exists()

    @pytest.mark.parametrize("time", [np.nan, np.inf, -np.inf])
    def test_series_rejects_non_finite_times(self, time):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="time must be finite"):
                bt.write_vtk_series(
                    mesh, [(time, {"field": np.zeros(3)})], Path(tmpdir) / "series"
                )

    @pytest.mark.parametrize("time", [True, "0.1", np.complex128(0.1 + 0.2j)])
    def test_series_rejects_non_numeric_time_semantics(self, time):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "nested" / "series"
            with pytest.raises(TypeError, match="time must be a real number"):
                bt.write_vtk_series(mesh, [(time, {"field": np.zeros(3)})], base_path)
            assert not base_path.parent.exists()

    def test_series_preflights_every_snapshot_before_writing(self):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        snapshots = [
            (0.0, {"field": np.zeros(3)}),
            (1.0, {"field": np.array([0.0, np.nan, 0.0])}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "nested" / "series"
            with pytest.raises(ValueError, match="finite"):
                bt.write_vtk_series(mesh, snapshots, base_path)
            assert not base_path.parent.exists()

    def test_series_escapes_xml_filename(self):
        mesh = bt.StructuredMesh(2, 0.0, 1.0)
        snapshots = [(0.0, {"field": np.zeros(3)})]
        with tempfile.TemporaryDirectory() as tmpdir:
            pvd_path = bt.write_vtk_series(mesh, snapshots, Path(tmpdir) / "drug&heat")
            assert 'file="drug&amp;heat_0000.vtk"' in pvd_path.read_text()


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

    def test_radial_mesh_export(self):
        mesh = bt.CylindricalMesh(5, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = bt.write_vtk(
                mesh,
                {"concentration": np.linspace(0.0, 1.0, mesh.num_nodes())},
                Path(tmpdir) / "radial.vtk",
            )
            content = result.read_text()
            assert "DIMENSIONS 6 1 1" in content
            assert "POINTS 6 double" in content
            assert "POINT_DATA 6" in content

    def test_axisymmetric_field_shape_is_validated(self):
        mesh = bt.CylindricalMesh(2, 3, 0.0, 1.0, -1.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match=r"shape \(4, 3\)"):
                bt.write_vtk(
                    mesh,
                    {"field": np.ones((3, 4))},
                    Path(tmpdir) / "transposed.vtk",
                )
