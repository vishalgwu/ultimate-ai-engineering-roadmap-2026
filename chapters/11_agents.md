# Chapter 11 — Agents

## 11.1 What an Agent Is, and When Not to Build One

### Conceptual Deep Dive
An **agent** is an LLM in a loop that decides actions, calls tools, observes results, and continues until a goal is met or a limit hits. Spectrum of autonomy: single call → chain/workflow (fixed steps) → router → agent (LLM chooses steps) → multi-agent. Prefer the lowest autonomy that solves the task: workflows are cheaper, faster, testable and predictable. Build agents when the number/order of steps is unknowable up front (research, coding, open-ended support). Core loop: *perceive (context) → plan → act (tool) → observe → update state → stop?* Key reliability levers: tight tool set, clear stop criteria, step/cost/time budgets, deterministic scaffolding around the LLM, and human checkpoints on irreversible actions. Errors compound: 95% per-step reliability over 10 steps ≈ 60% end-to-end.

### Cross-Connection Scenarios
Tool schemas (7), ReAct (6.3), eval trajectories (10), guardrails (14), idempotent side effects (4.5).

### Production Anti-Patterns
- Using an agent where a 3-step workflow suffices.
- No budget/timeouts; runaway loops.
- Agents granted broad credentials.

### Minimal Working Code
```python
import json
import time

def agent_loop(llm, tools: dict, goal: str, max_steps=8, budget_s=60):
    messages = [{"role": "user", "content": goal}]
    t0 = time.time()
    for step in range(max_steps):
        if time.time() - t0 > budget_s:
            return "budget exceeded"
        reply = llm(messages)                    # returns {"content":..., "tool_call":{name,args}|None}
        messages.append({"role": "assistant", "content": reply["content"]})
        call = reply.get("tool_call")
        if not call:
            return reply["content"]              # model decided it is done
        try:
            result = tools[call["name"]](**call["args"])
        except Exception as e:
            result = f"ERROR: {e}"
        messages.append({"role": "tool", "content": json.dumps(result, default=str)[:4000]})
    return "max steps reached"
```

## 11.2 Planning Patterns

### Conceptual Deep Dive
- **ReAct**: interleaved reasoning and acting; flexible, greedy.
- **Plan-and-Execute**: planner writes a full plan; executor runs steps; a replanner revises on failure. Cheaper (small model executes) and more inspectable.
- **ReWOO**: plan with variable placeholders, execute tools in parallel, then solve.
- **Reflexion / self-critique**: after failure, write a verbal lesson into memory and retry.
- **Tree/graph search** (LATS, ToT): explore alternatives with scoring.
- **Hierarchical**: manager decomposes, workers handle sub-tasks.
- **Verifier loops**: separate checker validates outputs (tests for code, schema for data) before accepting.
Prefer verifiable feedback (unit tests, compilers, SQL execution) over LLM opinions.

### Cross-Connection Scenarios
Reasoning prompts (6.2); evals of plan quality (10); cost control by routing executor to small models (2.2).

### Production Anti-Patterns
- Plans that never get revised when step 2 invalidates step 5.
- Planner and executor with different tool knowledge.
- Self-critique without any external signal (confidently wrong loops).

### Minimal Working Code
```python
def plan_and_execute(planner, executor, replanner, goal, max_replans=2):
    plan = planner(goal)                        # list[str]
    done, replans = [], 0
    while plan:
        step = plan.pop(0)
        ok, out = executor(step, done)
        if ok:
            done.append((step, out))
            continue
        if replans >= max_replans:
            return {"status": "failed", "done": done, "failed_step": step}
        replans += 1
        plan = replanner(goal, done, step, out)
    return {"status": "ok", "done": done}
```

## 11.3 State Management and Orchestration with LangGraph

### Conceptual Deep Dive
Model an agent as a **graph**: nodes are functions (LLM call, tool, human review), edges are transitions (conditional routing), and a typed **state** object flows through, updated by **reducers** (e.g. append to messages). **LangGraph** adds persistence via **checkpointers** (resume after crash, time-travel debugging), **human-in-the-loop interrupts** (pause for approval), streaming, subgraphs, and a `Send` API for map-reduce parallelism. Benefits over a `while` loop: explicit control flow, durable execution, observability per node, and testability of each node in isolation. Alternatives: Temporal/Durable Functions for workflow durability; PydanticAI, OpenAI Agents SDK, Claude Agent SDK, CrewAI, LlamaIndex Workflows, Semantic Kernel, Haystack.

### Cross-Connection Scenarios
Checkpoint stores use Postgres/Redis (13); traces per node (14); interrupts are guardrails (14).

### Production Anti-Patterns
- Storing everything in global variables rather than typed state.
- No checkpointing → lost work on pod restart.
- Unbounded message-list growth in state.

### Minimal Working Code
```python
# pip install langgraph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    attempts: int

def draft(state: State):
    return {"messages": [("assistant", "draft answer")], "attempts": state["attempts"] + 1}

def check(state: State):
    return "done" if state["attempts"] >= 2 else "retry"

g = StateGraph(State)
g.add_node("draft", draft)
g.add_edge(START, "draft")
g.add_conditional_edges("draft", check, {"done": END, "retry": "draft"})
app = g.compile()
print(app.invoke({"messages": [("user", "hi")], "attempts": 0})["attempts"])
```

## 11.4 Memory Architectures

### Conceptual Deep Dive
- **Working memory (short-term)**: the context window / conversation buffer. Manage with sliding windows, summarisation of old turns, and pruning of stale tool outputs.
- **Episodic memory**: records of past interactions/events ("last Tuesday the user asked about X"), stored with timestamps and retrieved by recency + similarity.
- **Semantic memory**: distilled facts and preferences ("user prefers metric units"), stored as key–value or knowledge-graph entries; updated by an LLM "memory manager" that adds, edits or deletes facts.
- **Procedural memory**: learned instructions/skills; e.g. the system prompt or tool recipes refined over time.
Retrieval scoring (Generative Agents): `α·recency + β·importance + γ·relevance`. Frameworks: Letta (MemGPT, OS-style paging between core and archival memory), mem0, Zep, LangGraph store. Concerns: privacy and deletion rights (GDPR), memory poisoning via injected "facts", and staleness/conflict resolution.

### Cross-Connection Scenarios
Vector search (8), RAG (9), privacy and PII (14), context engineering (6.4).

### Production Anti-Patterns
- Appending every turn forever.
- Writing LLM-inferred "facts" without confidence or provenance.
- No user-visible way to view/delete memories.

### Minimal Working Code
```python
import math
import time

class Memory:
    def __init__(self, embed):
        self.embed, self.items = embed, []

    def add(self, text: str, importance: float = 0.5, kind: str = "episodic"):
        self.items.append({"text": text, "v": self.embed(text), "t": time.time(),
                           "imp": importance, "kind": kind})

    def recall(self, query: str, k=3, half_life_s=86400):
        q = self.embed(query)
        def score(m):
            sim = sum(a * b for a, b in zip(q, m["v"]))
            recency = 0.5 ** ((time.time() - m["t"]) / half_life_s)
            return sim + 0.5 * recency + 0.5 * m["imp"]
        return sorted(self.items, key=score, reverse=True)[:k]

def summarize_if_long(llm, history: list[str], max_items=20):
    if len(history) <= max_items:
        return history
    old, recent = history[:-10], history[-10:]
    return [f"Summary: {llm('Summarize: ' + chr(10).join(old))}"] + recent
```

## 11.5 Multi-Agent Coordination: AutoGen, CrewAI and Patterns

### Conceptual Deep Dive
Patterns: **supervisor/worker** (router agent delegates), **swarm/handoff** (agents transfer control to peers), **debate/critic** (propose + critique), **pipeline** (specialists in sequence), **blackboard** (shared state). **AutoGen** (and its successor Microsoft Agent Framework) models conversational agents exchanging messages in group chats with speaker-selection; **CrewAI** assigns role-based agents tasks in a crew; LangGraph expresses these as explicit graphs. Multi-agent helps with context isolation (each agent sees a small context), parallelism, and specialisation — but adds cost, latency, and failure surface. Keep communication structured (typed messages, not free chat), set turn limits, and give one agent final authority.

### Cross-Connection Scenarios
Protocols: MCP for tools and A2A for agent-to-agent interoperability (7.4); eval of end-to-end trajectories (10); cost control (2.4).

### Production Anti-Patterns
- "Agents chatting" with no termination condition.
- Using five agents where one with tools works.
- Shared mutable state without ownership.

### Minimal Working Code
```python
def supervisor(llm, workers: dict, task: str, max_rounds=6):
    log, results = [], {}
    for _ in range(max_rounds):
        decision = llm(f"Task: {task}\nResults so far: {results}\n"
                       f"Workers: {list(workers)}\nReply 'DONE: <answer>' or '<worker>: <instruction>'")
        if decision.startswith("DONE:"):
            return decision[5:].strip()
        name, _, instruction = decision.partition(":")
        if name.strip() not in workers:
            results["error"] = f"unknown worker {name}"
            continue
        results[name.strip()] = workers[name.strip()](instruction.strip())
    return "round limit reached"
```

## 11.6 Evaluating and Securing Agents

### Conceptual Deep Dive
Evaluate **trajectories**, not just final answers: tool-selection accuracy, argument correctness, redundant steps, recovery after errors, cost per solved task, and success rate over N repeated runs (**pass^k** consistency). Benchmarks: SWE-bench, τ-bench, WebArena, GAIA — use for orientation, then build task-specific simulators. Security: least privilege, per-action authorisation, sandboxed execution (containers/gVisor/Firecracker), network egress allow-lists, secret isolation, human approval for destructive actions, and audit logs.

### Cross-Connection Scenarios
Eval (10), injection (6.5, 14), observability traces (14).

### Production Anti-Patterns
- Evaluating on one run per case.
- Code-execution tools running on the host.
- Missing kill switch.

### Minimal Working Code
```python
import subprocess
import sys
import tempfile

def run_sandboxed(code: str, timeout=5) -> str:
    """Demo only. In production use containers/gVisor with no network and resource limits."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    p = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=timeout)
    return (p.stdout + p.stderr)[:2000]

def pass_hat_k(results: list[list[bool]]) -> float:
    """Fraction of tasks solved in ALL k attempts."""
    return sum(all(r) for r in results) / len(results)

print(pass_hat_k([[True, True], [True, False]]))
```

## 11.7 Context Management for Long-Running Agents

### Conceptual Deep Dive
Agents accumulate tool outputs that fill the window. Techniques: **truncate and reference** (store large outputs externally, keep a short summary plus a handle the agent can re-read), **tool-result clearing** (drop old observations once consumed), **rolling summarisation** of earlier turns, **structured scratchpad/notes** the agent writes and re-reads (a file or state field), and **sub-agents** that do noisy exploration in a separate context and return only a distilled result. Prefer deterministic compaction rules over asking the model what to forget. Edge cases: summaries that drop a constraint the user stated early (pin hard requirements outside the summarised region), compaction invalidating prompt caching, and cyclic loops where the agent repeats an action because the earlier failure was trimmed.

### Cross-Connection Scenarios
Context engineering (6.4), working memory (11.4), cost control (2.4), prompt caching (2.4).

### Production Anti-Patterns
- Keeping every raw tool output forever.
- Summarising away the user's original goal and constraints.
- No loop detection.

### Minimal Working Code
```python
import hashlib

class ContextManager:
    def __init__(self, pinned: str, max_chars: int = 12_000, keep_last: int = 6):
        self.pinned, self.max, self.keep = pinned, max_chars, keep_last
        self.items: list[dict] = []
        self.blobs: dict[str, str] = {}

    def add_tool_result(self, name: str, text: str, inline_limit: int = 800):
        if len(text) > inline_limit:
            h = hashlib.sha1(text.encode()).hexdigest()[:8]
            self.blobs[h] = text
            text = f"{text[:inline_limit // 2]}... [truncated; full output id={h}]"
        self.items.append({"role": "tool", "name": name, "content": text})

    def render(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.pinned}] + self.items
        while sum(len(m["content"]) for m in msgs) > self.max and len(msgs) > self.keep + 1:
            msgs.pop(1)                      # drop oldest non-pinned item
        return msgs

cm = ContextManager("Goal: fix bug #12. Never edit tests.")
cm.add_tool_result("grep", "x" * 5000)
print(len(cm.render()), list(cm.blobs))
```

```python
def detect_loop(actions: list[tuple[str, str]], window: int = 6, repeats: int = 3) -> bool:
    """actions: [(tool_name, args_repr)]. True if the same call recurs too often recently."""
    recent = actions[-window:]
    return any(recent.count(a) >= repeats for a in set(recent))

print(detect_loop([("search", "q1")] * 3 + [("open", "u")]))
```

## 11.8 Durable Execution, Human-in-the-Loop and Error Recovery

### Conceptual Deep Dive
Long agent runs outlive processes. **Durable execution** persists state after each step (LangGraph checkpointers, Temporal workflows) so a crash resumes instead of restarting and re-paying. Design steps to be **idempotent** or record completed side effects. **Human-in-the-loop**: pause at an interrupt with a serialisable payload (proposed action), accept approve/edit/reject, then resume. **Error recovery** ladder: retry transient errors with backoff, feed deterministic tool errors back to the model once or twice, replan, fall back to a simpler path, then escalate to a human with the full trace. Edge cases: approvals that expire, state schema changes between deploy versions (resume of old checkpoints), concurrent runs on the same thread, and timeouts while waiting for humans (days).

### Cross-Connection Scenarios
Idempotency (4.5), LangGraph (11.3), approvals (7.7), observability (14.1).

### Production Anti-Patterns
- Keeping agent state only in process memory.
- Resuming from a checkpoint and re-running a side effect that already happened.
- Infinite retry on a deterministic tool error.

### Minimal Working Code
```python
import json
import os

class RunState:
    def __init__(self, path: str):
        self.path = path
        self.s = json.load(open(path)) if os.path.exists(path) else {"step": 0, "done": {}}

    def save(self):
        tmp = self.path + ".tmp"
        json.dump(self.s, open(tmp, "w"))
        os.replace(tmp, self.path)           # atomic

def run_steps(steps, state: RunState):
    for i in range(state.s["step"], len(steps)):
        name, fn = steps[i]
        if name not in state.s["done"]:
            state.s["done"][name] = fn()      # side effect recorded before advancing
        state.s["step"] = i + 1
        state.save()
    return state.s["done"]
```

```python
def recover(step_fn, replan_fn, escalate_fn, transient=(TimeoutError,), max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return step_fn()
        except transient:
            continue
        except ValueError as e:                 # deterministic error: give the model one chance
            alt = replan_fn(str(e))
            if alt is not None:
                return alt()
            break
    return escalate_fn()
```
