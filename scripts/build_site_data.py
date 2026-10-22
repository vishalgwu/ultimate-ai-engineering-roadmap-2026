"""Generate site/data.js and copy chapters + PDFs into site/ for the Vercel (Netflix-style) UI.

Run after audit/build_pdfs:  python scripts/build_site_data.py
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

# Editorial metadata (the "streaming catalogue" layer): tagline, level, genres, cast = key tools/ideas.
META = {
    1: ("About this Guide", "Your map before the journey: scope, philosophy and how to read the series.", "Beginner", ["Orientation", "Roadmap"], ["Reading tracks", "Responsible AI", "Lab setup"]),
    2: ("AI Engineering Fundamentals", "Software stops being deterministic. Learn to think in probabilities, budgets and failure modes.", "Beginner", ["Mindset", "Foundations"], ["Sampling", "Token economics", "Model routing"]),
    3: ("Python for AI Engineers", "Async, generators, typing and data tooling: the craft behind every pipeline.", "Beginner", ["Python", "Performance"], ["asyncio", "Pydantic", "Polars", "DuckDB"]),
    4: ("APIs, HTTP & Reliability", "Retries, breakers, streaming and idempotency: how production services survive the real world.", "Intermediate", ["Reliability", "Backend"], ["SSE", "WebSockets", "Redis", "Circuit breakers"]),
    5: ("ML & Deep Learning Foundations", "Tensors to transformers: attention, tokenisation, KV cache, GPUs and mixture-of-experts.", "Intermediate", ["Deep Learning", "Theory"], ["Attention", "BPE", "RoPE", "FSDP"]),
    6: ("Prompt Engineering", "From few-shot to ReAct to DSPy: programming models with language.", "Beginner", ["Prompting", "Reasoning"], ["Chain-of-Thought", "ReAct", "DSPy", "Injection defence"]),
    7: ("Structured Output & Tool Use", "Make models speak JSON and call tools safely, from schemas to MCP.", "Intermediate", ["Tools", "Contracts"], ["Pydantic", "XGrammar", "MCP", "Function calling"]),
    8: ("Embeddings & Vector Search", "Meaning as geometry: dense, sparse and multi-vector retrieval at scale.", "Intermediate", ["Search", "Vectors"], ["HNSW", "Qdrant", "pgvector", "ColBERT"]),
    9: ("Retrieval-Augmented Generation", "Ground answers in your data: chunking, hybrid search, reranking and GraphRAG.", "Intermediate", ["RAG", "Knowledge"], ["RRF", "HyDE", "GraphRAG", "Rerankers"]),
    10: ("Evaluation", "If you can't measure it, you can't ship it: golden sets, judges, statistics, A/B tests.", "Advanced", ["Quality", "Statistics"], ["RAGAS", "LLM-as-judge", "pass@k", "Bootstrap CIs"]),
    11: ("Agents", "Plan, act, remember, recover: autonomy with budgets, approvals and protocols.", "Advanced", ["Agents", "Orchestration"], ["LangGraph", "Memory", "A2A", "Computer use"]),
    12: ("Fine-Tuning & Open Models", "LoRA to GRPO: customise, quantise and serve open-weight models.", "Advanced", ["Training", "Open Models"], ["QLoRA", "DPO", "GRPO", "AWQ/GGUF"]),
    13: ("Serving & Deployment", "Continuous batching, PagedAttention, Kubernetes and the economics of GPUs.", "Advanced", ["Infrastructure", "GPU"], ["vLLM", "Triton", "KEDA", "Speculative decoding"]),
    14: ("Observability, Safety & Security", "Trace everything, red-team relentlessly, and keep humans in the loop.", "Advanced", ["Safety", "Security"], ["OpenTelemetry", "Guardrails", "OWASP LLM", "Audit logs"]),
    15: ("Modern Workflow & Testing", "CI/CD for prompts, models and indexes: gates, registries and safe rollouts.", "Intermediate", ["DevOps", "Testing"], ["GitOps", "Eval gates", "Registry", "FinOps"]),
    16: ("Communication & Career", "Translate metrics, ace system design, lead incidents, grow your career.", "All levels", ["Career", "Soft Skills"], ["Metric trees", "STAR", "Postmortems", "Interviews"]),
    17: ("Project Roadmap", "Nine production blueprints, from knowledge assistants to text-to-SQL.", "Intermediate", ["Projects", "Capstone"], ["RAG assistant", "Support copilot", "Code reviewer", "Voice agent"]),
    18: ("Learning Roadmap", "A 24-week season plan with labs, rubrics and a reading list of foundational papers.", "All levels", ["Plan", "Study"], ["Labs", "Rubrics", "Papers", "Cadence"]),
    19: ("Cheat Sheet", "Decision matrices, formulas, glossary, prompt patterns and debugging tables.", "All levels", ["Reference", "Quick wins"], ["Formulas", "Glossary", "Debugging", "Numbers"]),
}

ROWS = [
    {"id": "start", "title": "Start Here", "chapters": [1, 2, 18, 19]},
    {"id": "foundations", "title": "Foundations: The Origin Story", "chapters": [3, 4, 5]},
    {"id": "build", "title": "Build with Models", "chapters": [6, 7, 8, 9]},
    {"id": "quality", "title": "Quality & Autonomy", "chapters": [10, 11]},
    {"id": "ship", "title": "Customise, Ship & Protect", "chapters": [12, 13, 14, 15]},
    {"id": "career", "title": "Projects & Career", "chapters": [16, 17, 18]},
]
TOP10 = [9, 11, 10, 6, 7, 12, 13, 8, 14, 4]


def main() -> None:
    (SITE / "chapters").mkdir(parents=True, exist_ok=True)
    (SITE / "pdfs").mkdir(parents=True, exist_ok=True)
    chapters, concepts = [], []
    for md in sorted((ROOT / "chapters").glob("*.md")):
        num = int(md.name[:2])
        text = md.read_text(encoding="utf-8")
        title, tagline, level, genres, cast = META[num]
        secs = []
        parts = re.split(r"^## ", text, flags=re.M)[1:]
        for part in parts:
            head, _, body = part.partition("\n")
            m = re.match(r"(\d+\.\d+) (.+)$", head.strip())
            if not m:
                continue
            secs.append({"num": m.group(1), "title": m.group(2).strip(), "words": len(body.split())})
            concepts.append({"chapter": num, "num": m.group(1), "title": m.group(2).strip()})
        words = len(text.split())
        code = len(re.findall(r"```(?!\s*$)\w+", text))
        shutil.copyfile(md, SITE / "chapters" / md.name)
        pdf = ROOT / "pdfs" / (md.stem + ".pdf")
        if pdf.exists():
            shutil.copyfile(pdf, SITE / "pdfs" / pdf.name)
        chapters.append({
            "num": num, "slug": md.stem, "title": title, "tagline": tagline, "level": level,
            "genres": genres, "cast": cast, "sections": secs, "words": words,
            "minutes": max(1, round(words / 220)), "codeBlocks": code,
            "md": f"chapters/{md.name}", "pdf": f"pdfs/{md.stem}.pdf",
        })
    data = {"chapters": chapters, "rows": ROWS, "top10": TOP10, "concepts": concepts,
            "totals": {"chapters": len(chapters), "sections": len(concepts),
                       "words": sum(c["words"] for c in chapters), "code": sum(c["codeBlocks"] for c in chapters)}}
    (SITE / "data.js").write_text("window.ROADMAP = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"site data: {data['totals']}")


if __name__ == "__main__":
    main()
