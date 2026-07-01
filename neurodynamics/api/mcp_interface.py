"""MCP interface stub for integration with Claude and other MCP clients."""
from typing import Any, Dict, List


class MCPInterface:
    """Exposes neurodynamics tools as MCP-compatible tool definitions."""

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "simulate_neuron",
                "description": "Simulate a neuron model",
                "parameters": {
                    "model": {"type": "string", "enum": ["lif", "adex", "izhikevich"]},
                    "duration": {"type": "number"},
                    "current": {"type": "number"},
                }
            },
            {
                "name": "analyze_dynamics",
                "description": "Analyze dynamical properties of time series",
                "parameters": {
                    "data": {"type": "array"},
                    "analysis": {"type": "string", "enum": ["entropy", "lyapunov"]},
                }
            },
        ]

    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        if tool_name == "simulate_neuron":
            from .rest import simulate, SimulateRequest
            return simulate(SimulateRequest(**params))
        raise ValueError(f"Unknown tool: {tool_name}")
