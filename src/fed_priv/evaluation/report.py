"""Markdown report generation from benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(results: dict[str, Any], path: str | Path) -> None:
    """Write a human-readable markdown summary."""
    lines = [
        "# Federated & Privacy-Preserving ML Benchmark Report",
        "",
        f"**Config hash:** `{results.get('config_hash', 'n/a')}`",
        f"**Timestamp:** {results.get('timestamp', 'n/a')}",
        "",
    ]

    modules = results.get("modules", {})
    for module_name, module_data in modules.items():
        lines.append(f"## {module_name.replace('_', ' ').title()}")
        lines.append("")
        rows = module_data.get("results", [])
        if not rows:
            lines.append("_No results._")
            lines.append("")
            continue
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            cells = [
                str(round(row[h], 4)) if isinstance(row[h], float) else str(row[h]) for h in headers
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    Path(path).write_text("\n".join(lines))
