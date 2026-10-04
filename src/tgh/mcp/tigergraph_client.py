"""Layer 3: TigerGraph MCP client with strict tool allowlist enforcement."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from mcp_types import TextContent
from tigergraph_mcp.server import ConnectionManager, MCPServer

# Explicit allowlist of agent-accessible read/query/vector tools
ALLOWED_MCP_TOOLS: tuple[str, ...] = (
    "tigergraph__get_graph_schema",
    "tigergraph__show_graph_details",
    "tigergraph__get_node",
    "tigergraph__get_nodes",
    "tigergraph__get_node_edges",
    "tigergraph__get_edges",
    "tigergraph__get_neighbors",
    "tigergraph__run_installed_query",
    "tigergraph__search_top_k_similarity",
    "tigergraph__fetch_vector",
)

# Explicit blocklist of strictly forbidden tools (mutation, DDL, arbitrary GSQL, jobs)
FORBIDDEN_MCP_TOOLS: tuple[str, ...] = (
    "tigergraph__gsql",
    "tigergraph__generate_gsql",
    "tigergraph__generate_cypher",
    "tigergraph__create_graph",
    "tigergraph__drop_graph",
    "tigergraph__clear_graph_data",
    "tigergraph__add_node",
    "tigergraph__add_nodes",
    "tigergraph__delete_node",
    "tigergraph__delete_nodes",
    "tigergraph__add_edge",
    "tigergraph__add_edges",
    "tigergraph__delete_edge",
    "tigergraph__delete_edges",
    "tigergraph__install_query",
    "tigergraph__drop_query",
    "tigergraph__create_loading_job",
    "tigergraph__drop_loading_job",
    "tigergraph__run_loading_job_with_file",
    "tigergraph__run_loading_job_with_data",
    "tigergraph__add_vector_attribute",
    "tigergraph__drop_vector_attribute",
    "tigergraph__upsert_vectors",
    "tigergraph__update_schema",
)


@dataclass(frozen=True)
class MCPToolCallResult:
    """Standardized response from an allowed MCP tool invocation."""

    tool_name: str
    success: bool
    data: Any
    summary: str = ""
    error: str | None = None
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
            "error": self.error,
        }


class TigerGraphMCPClient:
    """Safe, persistent client to official tigergraph-mcp with allowlist enforcement."""

    def __init__(
        self,
        env_dict: dict[str, str] | None = None,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize client and map environment configuration for tigergraph-mcp."""
        self.allowed_tools = set(allowed_tools or ALLOWED_MCP_TOOLS)
        self.forbidden_tools = set(FORBIDDEN_MCP_TOOLS)

        if env_dict:
            # Map project environment variables to official tigergraph-mcp variables
            if "TIGERGRAPH_HOST" in env_dict:
                os.environ["TG_HOST"] = env_dict["TIGERGRAPH_HOST"]
            if "TIGERGRAPH_GRAPH_NAME" in env_dict:
                os.environ["TG_GRAPHNAME"] = env_dict["TIGERGRAPH_GRAPH_NAME"]
            if "TIGERGRAPH_SECRET" in env_dict:
                os.environ["TG_SECRET"] = env_dict["TIGERGRAPH_SECRET"]

        ConnectionManager._connection_pool.clear()
        ConnectionManager.load_profiles()
        self.server = MCPServer()

    async def list_allowed_tools(self) -> list[str]:
        """Return list of allowed tools served by this client."""
        all_tools = await self.server._handle_list_tools()
        return [t.name for t in all_tools if t.name in self.allowed_tools]

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check whether tool is allowed and not on the forbidden blocklist."""
        normalized = (
            tool_name
            if tool_name.startswith("tigergraph__")
            else f"tigergraph__{tool_name}"
        )
        if normalized in self.forbidden_tools:
            return False
        return normalized in self.allowed_tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolCallResult:
        """Call an official TigerGraph MCP tool if permitted by the allowlist.

        Args:
            tool_name: Full name of tool (e.g. 'tigergraph__get_node' or 'get_node').
            arguments: Dictionary of arguments passed to tool.

        Returns:
            MCPToolCallResult with structured parsed JSON data.
        """
        norm_name = (
            tool_name
            if tool_name.startswith("tigergraph__")
            else f"tigergraph__{tool_name}"
        )
        args = arguments or {}

        # 1. Enforcement Check
        if not self.is_tool_allowed(norm_name):
            err_msg = (
                f"Access denied: Tool '{norm_name}' is not in allowed list "
                "or is explicitly forbidden."
            )
            return MCPToolCallResult(
                tool_name=norm_name,
                success=False,
                data=None,
                error=err_msg,
            )

        # 2. Invoke official server handler
        try:
            res_contents: list[TextContent] = await self.server._handle_call_tool(
                norm_name, args
            )
        except Exception as e:
            return MCPToolCallResult(
                tool_name=norm_name,
                success=False,
                data=None,
                error=f"MCP call failed: {e}",
            )

        if not res_contents:
            return MCPToolCallResult(
                tool_name=norm_name,
                success=False,
                data=None,
                error="Empty response from MCP tool",
            )

        raw_text = res_contents[0].text
        data, success, summary, err = self._parse_mcp_output(raw_text)

        return MCPToolCallResult(
            tool_name=norm_name,
            success=success,
            data=data,
            summary=summary,
            error=err,
            raw_text=raw_text,
        )

    def _parse_mcp_output(
        self,
        raw_text: str,
    ) -> tuple[Any, bool, str, str | None]:
        """Extract JSON payload from markdown-formatted MCP TextContent."""
        try:
            # TigerGraph MCP returns: ```json\n{...}\n```\n\n**Summary**...
            if "```json" in raw_text:
                json_part = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(json_part)
                success = parsed.get("success", False)
                summary = parsed.get("summary", "")
                err = parsed.get("error", None)
                data = parsed.get("data", None)
                return data, success, summary, err
            elif raw_text.strip().startswith("{"):
                parsed = json.loads(raw_text.strip())
                return (
                    parsed.get("data", None),
                    parsed.get("success", True),
                    parsed.get("summary", ""),
                    parsed.get("error", None),
                )
        except Exception:
            pass

        return None, True, raw_text[:200], None
