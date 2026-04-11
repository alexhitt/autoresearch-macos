"""
Build system context summary for disposition evaluation.

Replicates buildSystemContextSummary() from application.ts in Python.
Reads: Stella/Context.md, _System/Config/rules/*.md, projects-registry.yaml.

Usage:
  python scoring/build_system_context.py           # prints context
  python scoring/build_system_context.py --output system_context.txt
"""

import argparse
import os
import re

VAULT_BASE = os.path.expanduser("~/vaults/Alex Hitt")
STELLA_CONTEXT_PATH = os.path.join(VAULT_BASE, "Stella/Context.md")
RULES_DIR = os.path.join(VAULT_BASE, "_System/Config/rules")
REGISTRY_PATH = os.path.join(VAULT_BASE, "_System/Config/projects-registry.yaml")


def read_disposition_context():
    """Extract structured, decision-useful sections from Stella/Context.md.

    Matches production buildDispositionContext() in application.ts —
    pulls specific sections instead of arbitrary slicing.
    """
    try:
        with open(STELLA_CONTEXT_PATH) as f:
            raw = f.read()
    except FileNotFoundError:
        return "AI OS: automation, dashboards, research intelligence, prediction markets, NightMind orchestration, Obsidian vault."

    sections = []

    # Extract "Active Projects" table
    projects_match = re.search(r"## Active Projects\n\n((?:\|.*\n)+)", raw)
    if projects_match:
        sections.append("## Active Projects\n" + projects_match.group(1).strip())

    # Extract "Strategic Priorities"
    priorities_match = re.search(r"## Strategic Priorities[^\n]*\n\n((?:(?!\n## ).+\n?)+)", raw)
    if priorities_match:
        sections.append("## Strategic Priorities\n" + priorities_match.group(1).strip())

    # Extract "Active Systems" inventory
    systems_match = re.search(r"## Active Systems[^\n]*\n\n((?:(?!\n## ).+\n?)+)", raw)
    if systems_match:
        sections.append("## Active Systems\n" + systems_match.group(1).strip())

    if not sections:
        return raw[:2000]

    return "\n\n".join(sections)


def build_project_state():
    """Parse projects-registry.yaml matching production buildProjectStateContext().

    Includes built field and OPERATIONAL STATE label (not just in-progress).
    """
    try:
        with open(REGISTRY_PATH) as f:
            raw = f.read()
    except FileNotFoundError:
        return "(projects registry not available)"

    lines = []
    current_project = ""
    current_status = ""
    current_blocked = ""
    current_in_progress = ""
    current_built = ""

    def is_null(v):
        return not v or v in ("null", "None", "nothing", "Nothing")

    def emit():
        if current_project and current_status:
            parts = [f"{current_project}: {current_status}"]
            if not is_null(current_blocked):
                parts.append(f"BLOCKED: {current_blocked[:120]}")
            if not is_null(current_in_progress):
                parts.append(f"OPERATIONAL STATE: {current_in_progress[:200]}")
            if not is_null(current_built):
                parts.append(f"BUILT: {current_built[:120]}")
            lines.append(f"- {' | '.join(parts)}")

    for line in raw.split("\n"):
        project_match = re.match(r"^  ([\w-]+):$", line)
        if project_match:
            emit()
            current_project = project_match.group(1)
            current_status = ""
            current_blocked = ""
            current_in_progress = ""
            current_built = ""
            continue

        status_match = re.match(r"^\s+status:\s*(.+)", line)
        if status_match:
            current_status = status_match.group(1).strip()

        blocked_match = re.match(r"^\s+blocked:\s*(.+)", line)
        if blocked_match:
            current_blocked = blocked_match.group(1).strip()

        ip_match = re.match(r"^\s+in_progress:\s*(.+)", line)
        if ip_match:
            current_in_progress = ip_match.group(1).strip()

        built_match = re.match(r"^\s+built:\s*(.+)", line)
        if built_match:
            current_built = built_match.group(1).strip()

    emit()
    return "\n".join(lines) if lines else "(no project state available)"


def build_system_context():
    """Build system context matching production buildSystemContextSummary().

    Production removed rules index (adds ~800 tokens of noise that dilutes
    decision quality). Uses structured section extraction instead of raw slice.
    """
    disposition_context = read_disposition_context()
    project_state = build_project_state()

    return "\n".join([
        "## AI OS — Structured System State",
        disposition_context,
        "",
        "## Live Project Status (from projects-registry.yaml)",
        "CRITICAL: Use this to determine what is CURRENTLY running. "
        '"active" status alone is NOT sufficient — read the OPERATIONAL STATE field. '
        "Systems that are halted, decommissioned, have zero usage, or have zero trades "
        "are NOT operationally ready for improvements.",
        project_state,
    ])


def main():
    parser = argparse.ArgumentParser(description="Build system context for disposition evaluation")
    parser.add_argument("--output", help="Write to file instead of stdout")
    args = parser.parse_args()

    context = build_system_context()

    if args.output:
        with open(args.output, "w") as f:
            f.write(context)
        print(f"System context written to {args.output} ({len(context)} chars)")
    else:
        print(context)
        print(f"\n--- {len(context)} chars ---")


if __name__ == "__main__":
    main()
