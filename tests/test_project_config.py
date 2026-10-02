"""Minimal project configuration tests.

Verifies that the Python environment satisfies the 3.12 requirement,
pyproject.toml metadata is consistent, and the package structure is valid.
Compatible with both standard library unittest and pytest.
"""

from pathlib import Path
import sys
import tomllib
import unittest


class TestProjectConfiguration(unittest.TestCase):
    """Test suite for project baseline setup and Python 3.12 environment."""

    def test_python_version(self) -> None:
        """Verify runtime Python version is at least 3.12."""
        self.assertGreaterEqual(
            sys.version_info[:2],
            (3, 12),
            f"Python 3.12+ required, found {sys.version_info.major}.{sys.version_info.minor}",
        )

    def test_pyproject_metadata(self) -> None:
        """Verify pyproject.toml exists and declares expected metadata and constraints."""
        repo_root = Path(__file__).resolve().parent.parent
        pyproject_file = repo_root / "pyproject.toml"
        self.assertTrue(pyproject_file.is_file(), "pyproject.toml must exist at repo root")

        with pyproject_file.open("rb") as f:
            data = tomllib.load(f)

        project_section = data.get("project", {})
        self.assertEqual(project_section.get("name"), "tgh")
        self.assertEqual(project_section.get("requires-python"), ">=3.12")
        self.assertIn("version", project_section)

    def test_package_import(self) -> None:
        """Verify that tgh source package is importable and exposes __version__."""
        repo_root = Path(__file__).resolve().parent.parent
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        import tgh

        self.assertTrue(hasattr(tgh, "__version__"))
        self.assertIsInstance(tgh.__version__, str)


if __name__ == "__main__":
    unittest.main()
