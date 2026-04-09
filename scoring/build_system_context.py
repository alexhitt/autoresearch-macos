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


def read_stella_context(max_chars=4000):
    try:
        with open(STELLA_CONTEXT_PATH) as f:
            return f.read()[:max_chars]
    except FileNotFoundError:
        return "AI OS: automation, dashboards, research intelligence, prediction markets, NightMind orchestration, Obsidian vault."


def build_rules_index():
    try:
        entries = sorted(os.listdir(RULES_DIR))
    except FileNotFoundError:
        return "(rules directory not available)"

    lines = []
    for filename in entries:
        if not filename.endswith(".md") or ".archived" in filename:
            continue
        filepath = os.path.join(RULES_DIR, filename)
        try:
            with open(filepath) as f:
                content = f.read()
        except (FileNotFoundError, PermissionError):
            continue

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else filename.replace(".md", "")

        # Extract first substantive paragraph for context
        paragraphs = [
            p.strip()
            for p in content.split("\n\n")
            if p.strip() and not p.strip().startswith("---") and not p.strip().startswith("#")
        ]
        summary = paragraphs[0].replace("\n", " ").strip()[:150] if paragraphs else ""

        name = filename.replace(".md", "")
        line = f"- {name}: {title}"
        if summary:
            line += f" — {summary}"
        lines.append(line)

    return "\n".join(lines) if lines else "(no rules found)"


def build_project_state():
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

    def emit():
        if current_project and current_status:
            parts = [f"{current_project}: {current_status}"]
            if current_blocked and current_blocked not in ("null", "None"):
                parts.append(f"BLOCKED: {current_blocked}")
            if current_in_progress and current_in_progress not in ("null", "Nothing", "nothing"):
                parts.append(f"in-progress: {current_in_progress[:80]}")
            lines.append(f"- {' | '.join(parts)}")

    for line in raw.split("\n"):
        project_match = re.match(r"^  ([\w-]+):$", line)
        if project_match:
            emit()
            current_project = project_match.group(1)
            current_status = ""
            current_blocked = ""
            current_in_progress = ""
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

    emit()
    return "\n".join(lines) if lines else "(no project state available)"


def build_system_context():
    stella = read_stella_context()
    rules = build_rules_index()
    projects = build_project_state()

    return "\n".join([
        "## Current AI OS State",
        stella,
        "",
        "## Live Project Status (from projects-registry.yaml)",
        "Use this to determine what is CURRENTLY active, blocked, or halted. Do NOT classify a claim as apply_now if it targets a blocked or halted system unless it specifically addresses the blocker.",
        projects,
        "",
        "## Existing System Rules (already implemented)",
        "The following rules already govern the AI OS. Do NOT recommend something as apply_now if an existing rule already covers it:",
        rules,
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
