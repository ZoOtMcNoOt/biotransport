"""Tests for utils module."""

import os
import tempfile
from pathlib import Path


from biotransport.utils import (
    _find_repo_root_from_cwd,
    get_results_dir,
    get_result_path,
)


class TestFindRepoRoot:
    """Tests for _find_repo_root_from_cwd function."""

    def test_find_repo_root_returns_path_or_none(self):
        """Test that function returns Path or None."""
        result = _find_repo_root_from_cwd()
        assert result is None or isinstance(result, Path)

    def test_find_repo_root_from_workspace(self):
        """Test finding repo root from within workspace."""
        # When running from the biotransport directory, should find root
        result = _find_repo_root_from_cwd()
        if result is not None:
            # Should contain expected files
            assert (result / "pyproject.toml").is_file()
            assert (result / "python" / "biotransport").is_dir()


class TestGetResultsDir:
    """Tests for get_results_dir function."""

    def test_get_results_dir_creates_directory(self):
        """Test that results dir is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_results_dir(base_dir=tmpdir)

            assert os.path.isdir(result)
            assert result.endswith("results")

    def test_get_results_dir_with_subfolder(self):
        """Test results dir with subfolder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_results_dir("my_experiment", base_dir=tmpdir)

            assert os.path.isdir(result)
            assert "my_experiment" in result

    def test_get_results_dir_with_timestamp(self):
        """Test results dir with timestamp subfolder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_results_dir("timestamp", base_dir=tmpdir)

            assert os.path.isdir(result)
            # Should contain timestamp pattern (YYYYMMDD-HHMMSS)
            dirname = os.path.basename(result)
            assert len(dirname) == 15  # 8 date + 1 dash + 6 time

    def test_get_results_dir_with_env_var(self):
        """Test results dir respects environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_var = "TEST_BIOTRANSPORT_RESULTS"
            try:
                os.environ[env_var] = tmpdir
                result = get_results_dir(env_var=env_var)

                assert result.startswith(tmpdir)
            finally:
                del os.environ[env_var]

    def test_get_results_dir_env_var_overrides_base_dir(self):
        """Test that env var takes precedence over base_dir."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                env_var = "TEST_BIOTRANSPORT_RESULTS2"
                try:
                    os.environ[env_var] = tmpdir1
                    result = get_results_dir(base_dir=tmpdir2, env_var=env_var)

                    # Should use env var, not base_dir
                    assert result.startswith(tmpdir1)
                finally:
                    del os.environ[env_var]

    def test_get_results_dir_nested_subfolder(self):
        """Test results dir with nested subfolder path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_results_dir("level1/level2/level3", base_dir=tmpdir)

            assert os.path.isdir(result)
            assert "level1" in result
            assert "level2" in result
            assert "level3" in result

    def test_get_results_dir_idempotent(self):
        """Test that calling twice returns same path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = get_results_dir("test", base_dir=tmpdir)
            result2 = get_results_dir("test", base_dir=tmpdir)

            assert result1 == result2


class TestGetResultPath:
    """Tests for get_result_path function."""

    def test_get_result_path_basic(self):
        """Test basic result path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_result_path("output.csv", base_dir=tmpdir)

            assert result.endswith("output.csv")
            assert "results" in result

    def test_get_result_path_with_subfolder(self):
        """Test result path with subfolder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_result_path("data.npy", "experiment1", base_dir=tmpdir)

            assert result.endswith("data.npy")
            assert "experiment1" in result

    def test_get_result_path_creates_dirs(self):
        """Test that result path creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_result_path("file.txt", "exp", base_dir=tmpdir)

            # The directory should exist
            parent = os.path.dirname(result)
            assert os.path.isdir(parent)  # 'exp' dir under 'results'

    def test_get_result_path_extension_preserved(self):
        """Test various file extensions are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions = [".csv", ".npy", ".vtk", ".png", ".json", ".h5"]

            for ext in extensions:
                result = get_result_path(f"file{ext}", base_dir=tmpdir)
                assert result.endswith(ext)

    def test_get_result_path_with_env_var(self):
        """Test result path respects environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_var = "TEST_RESULTS_PATH"
            try:
                os.environ[env_var] = tmpdir
                result = get_result_path("output.txt", env_var=env_var)

                assert result.startswith(tmpdir)
            finally:
                del os.environ[env_var]


class TestUtilsIntegration:
    """Integration tests for utils module."""

    def test_create_file_in_results_dir(self):
        """Test creating a file in the results directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = get_result_path(
                "test_output.txt", "integration", base_dir=tmpdir
            )

            # Create a file
            with open(filepath, "w") as f:
                f.write("test data")

            # Verify file exists
            assert os.path.isfile(filepath)
            with open(filepath) as f:
                assert f.read() == "test data"

    def test_multiple_experiments_same_base(self):
        """Test multiple experiment directories under same base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp1_dir = get_results_dir("exp1", base_dir=tmpdir)
            exp2_dir = get_results_dir("exp2", base_dir=tmpdir)
            exp3_dir = get_results_dir("exp3", base_dir=tmpdir)

            # All should exist and be different
            assert os.path.isdir(exp1_dir)
            assert os.path.isdir(exp2_dir)
            assert os.path.isdir(exp3_dir)
            assert exp1_dir != exp2_dir != exp3_dir
