# The Ultimate AI Engineering Roadmap (2026)

A 19-chapter, code-first guide to building production AI systems. Every concept section has four parts:
**Conceptual Deep Dive**, **Cross-Connection Scenarios**, **Production Anti-Patterns**, **Minimal Working Code**.

## Layout

| Path | Purpose |
|---|---|
| `chapters/` | The 19 chapters as Markdown (source of truth) |
| `pdfs/` | Generated PDFs, one per chapter |
| `scripts/build_pdfs.py` | Markdown -> PDF compiler (`markdown` + `xhtml2pdf`) |
| `scripts/audit.py` | Agent 21 structural audit (sections, code parses, required concepts) |
| `scripts/merge_extensions.py` | Appends `extensions/*.md` expansion sections to their chapters (idempotent) |
| `extensions/` | Expansion sections (`NN.md`, `NNb.md`) already merged into the chapters |
| `CROSSCHECK.md` | What was audited, what was added, and the known remaining gaps |
| `scripts/required_concepts.json` | Per-chapter concept checklist the audit enforces |

## Chapters

1. About this Guide
2. AI Engineering Fundamentals
3. Python for AI Engineers
4. APIs, HTTP & Reliability
5. ML & Deep Learning Foundations
6. Prompt Engineering
7. Structured Output & Tool Use
8. Embeddings & Vector Search
9. Retrieval-Augmented Generation
10. Evaluation
11. Agents
12. Fine-Tuning & Open Models
13. Serving & Deployment
14. Observability, Safety & Security
15. Modern Workflow & Testing
16. Communication & Career
17. Project Roadmap
18. Learning Roadmap
19. Cheat Sheet

Chapter numbers match agent numbers 01-19 from the brief.

## Build

```bash
pip install -r requirements.txt
python scripts/audit.py
python scripts/build_pdfs.py
```

Code samples target Python 3.10+. Provider SDK calls are shown against a generic client so they port across vendors;
check current SDK docs for exact model names and parameters, which change quickly.
