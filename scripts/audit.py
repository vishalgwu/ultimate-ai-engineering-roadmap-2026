"""Agent 21 guardrail: structural audit of every chapter.

Rules (per `## ` concept section, except chapters flagged as reference-style):
  1. Must contain the four mandatory subsections.
  2. Must contain at least one fenced code block; ```python blocks must parse (ast).
Also reports required-concept keywords that are missing from each chapter
(scripts/required_concepts.json), so gaps force an expansion pass.
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["Conceptual Deep Dive", "Cross-Connection Scenarios",
            "Production Anti-Patterns", "Minimal Working Code"]
# Chapters that are navigational/reference by nature: only code-parse + keyword checks.
REFERENCE_STYLE = {"01_about_this_guide", "16_communication_and_career",
                   "17_project_roadmap", "18_learning_roadmap", "19_cheat_sheet"}


def sections(text):
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return [(p.split("\n", 1)[0].strip(), p) for p in parts]


def main():
    concepts = json.loads((ROOT / "scripts" / "required_concepts.json").read_text(encoding="utf-8"))
    problems = 0
    files = sorted((ROOT / "chapters").glob("*.md"))
    if len(files) != 19:
        print(f"[WARN] expected 19 chapters, found {len(files)}")
    for md in files:
        text = md.read_text(encoding="utf-8")
        low = text.lower()
        issues = []
        for title, body in sections(text):
            if md.stem not in REFERENCE_STYLE and re.match(r"\d+\.\d+", title):
                for req in REQUIRED:
                    if req.lower() not in body.lower():
                        issues.append(f"section '{title}' missing '{req}'")
                if "```" not in body:
                    issues.append(f"section '{title}' has no code block")
        for m in re.finditer(r"```python\n(.*?)```", text, flags=re.S):
            try:
                ast.parse(m.group(1))
            except SyntaxError as e:
                issues.append(f"python block syntax error line {e.lineno}: {e.msg}")
        for kw in concepts.get(md.stem, []):
            if kw.lower() not in low:
                issues.append(f"missing required concept: {kw}")
        words = len(text.split())
        status = "PASS" if not issues else "FAIL"
        print(f"{status} {md.name} ({words} words)")
        for i in issues:
            print("   -", i)
        problems += len(issues)
    print(f"\nTotal issues: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
