"""Minimal connectivity test for TigerGraph backend.

Loads configuration strictly from .env and verifies that TigerGraph responds.
Never prints or logs secret credentials.
"""

import os
import unittest
from pathlib import Path

import pyTigerGraph as tg


def _load_env_vars() -> dict[str, str]:
    """Parse key-value pairs from project-root .env file."""
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return {}

    values: dict[str, str] = {}
    with env_file.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, val = stripped.split("=", 1)
                values[key.strip()] = val.strip()
    return values


class TestTigerGraphConnectivity(unittest.TestCase):
    """Test suite for TigerGraph connection and graph access."""

    def test_tigergraph_connection(self) -> None:
        """Verify TigerGraph connection using .env configuration."""
        env_vars = _load_env_vars()

        host = env_vars.get("TIGERGRAPH_HOST") or os.environ.get("TIGERGRAPH_HOST")
        graph_name = (
            env_vars.get("TIGERGRAPH_GRAPH_NAME")
            or os.environ.get("TIGERGRAPH_GRAPH_NAME")
        )
        secret = (
            env_vars.get("TIGERGRAPH_SECRET")
            or os.environ.get("TIGERGRAPH_SECRET")
        )

        self.assertTrue(host, "TIGERGRAPH_HOST must be configured in .env")
        self.assertTrue(
            graph_name, "TIGERGRAPH_GRAPH_NAME must be configured in .env"
        )
        self.assertTrue(secret, "TIGERGRAPH_SECRET must be configured in .env")

        conn = tg.TigerGraphConnection(
            host=host,
            graphname=graph_name,
            gsqlSecret=secret,
        )

        try:
            echo_response = conn.echo()
            self.assertIsNotNone(echo_response)
            vertex_types = conn.getVertexTypes()
            self.assertIsInstance(vertex_types, list)
        except Exception as e:
            if "500" in str(e) or "Failed to start workspace" in str(e):
                self.skipTest(f"TigerGraph Cloud workspace is stopped/paused: {e}")
            raise


if __name__ == "__main__":
    unittest.main()
