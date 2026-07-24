"""
Platform Stub Toolset
======================

Provides a small set of named tools the ReAct agent can call.
Each tool maps to a human-readable action string; the environment
simulator (in agent_loop.py) interprets these strings to generate
a plausible Observation based on the scenario context.

Tools are intentionally generic so they cover all five InGen platforms:
  - Fari (eldercare)         → check_log, escalate, log_event, confirm
  - Senpai (education)       → check_log, log_event, confirm, escalate
  - Sentinel Prime (security)→ escalate, confirm, log_event, query_map
  - Aido Rover (patrol)      → move, halt, query_map, check_log, log_event, confirm
  - Aido Humanoid (research) → move, halt, check_log, log_event, confirm, escalate

Each tool call returns a structured dict that gets inserted into the
ReAct transcript as an Observation.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation parsed from the agent's Action output."""
    tool_name: str
    tool_input: str


# ---------------------------------------------------------------------------
# Tool registry — the canonical list of tools the agent may invoke
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, str] = {
    "move": (
        "Move the platform in the specified direction or toward a target. "
        "Input: target or direction description."
    ),
    "halt": (
        "Immediately stop all movement. "
        "Input: reason for halt or 'emergency'."
    ),
    "query_map": (
        "Query the topological or facility map for route, location, or layout information. "
        "Input: query string (e.g., 'alternate route to Sector 4', 'location of charging station')."
    ),
    "check_log": (
        "Check the platform's internal log for recent events, readings, or records. "
        "Input: what to check (e.g., 'battery level', 'medication log', 'motion sensor history')."
    ),
    "escalate": (
        "Escalate to a human operator, caregiver, or emergency service. "
        "Input: message to send and escalation target (e.g., 'caregiver', '911', 'security desk')."
    ),
    "confirm": (
        "Confirm completion of a task step or verify the current state. "
        "Input: what to confirm (e.g., 'charging handshake', 'door locked', 'medication logged')."
    ),
    "log_event": (
        "Write an entry to the platform's event log. "
        "Input: event description to log."
    ),
    "report": (
        "Generate and transmit a structured report on the current situation. "
        "Input: report content or type (e.g., 'anomaly report', 'incident summary')."
    ),
    "assess": (
        "Assess the current environment, resident, or situation for relevant information. "
        "Input: what to assess (e.g., 'resident condition', 'obstacle type', 'threat level')."
    ),
    "reroute": (
        "Calculate and activate an alternative path or plan, discarding the previous one. "
        "Input: reason for reroute and constraints (e.g., 'avoid flooded area, conserve battery')."
    ),
}


def tool_descriptions_for_prompt() -> str:
    """Format the tool registry as a tool list for the ReAct system prompt."""
    lines = []
    for name, desc in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def parse_action(raw_action: str) -> ToolCall:
    """
    Parse a raw Action string from the ReAct output.

    Expects format:  <tool_name>: <tool_input>
    Falls back gracefully if parsing fails.
    """
    raw_action = raw_action.strip()
    if ":" in raw_action:
        parts = raw_action.split(":", 1)
        tool_name = parts[0].strip().lower().replace(" ", "_")
        tool_input = parts[1].strip()
    else:
        # Treat the entire string as the tool input for an unknown action
        tool_name = "log_event"
        tool_input = raw_action

    # Normalize to registry; fall back to log_event for unknown tools
    if tool_name not in TOOL_REGISTRY:
        tool_name = "log_event"

    return ToolCall(tool_name=tool_name, tool_input=tool_input)
