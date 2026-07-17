# Financial Research Analyst Agent

An AI agent that autonomously researches a company using multiple tools — live
stock data, web search, RAG over an uploaded filing, and deterministic
financial calculations — reasons about which tools it actually needs for a
given question, and generates a structured investment report.

Built with **LangChain**, **LangGraph** (ReAct agent), **Groq** (LLM
inference), **Pinecone** (vector store + hosted embeddings), **FastAPI**, and
evaluated end-to-end with **LangSmith**.

## Tech stack & skills demonstrated

- **Python** — FastAPI, Pydantic, async/await, custom class design
- **LangChain / LangGraph** — ReAct agents, LCEL chains (`prompt | llm | parser`),
  custom tools (`@tool`), custom `Embeddings` adapter, document loaders, text splitting
- **LLM integration** — Groq (`llama-3.3-70b-versatile`) for agent reasoning
- **Vector databases / RAG** — Pinecone (hosted embeddings + vector storage,
  namespace-based multi-tenancy)
- **External APIs** — yfinance (market data), Tavily (web search)
- **API design** — REST endpoints, request/response validation, error handling
- **Evaluation & observability** — LangSmith datasets/experiments, tracing
  individual run failures down to root cause, iterating on error rate across
  successive experiment runs
- **AI safety / reliability** — system prompt guardrails against data
  fabrication and entity conflation, retry/fallback handling for LLM
  provider-level tool-calling failures, documented known limitations
- **Engineering practices** — environment variable management, `.gitignore`
  hygiene, modular file structure, API quota management, security hardening
  checklist for future deployment

## Why this project

Most RAG demos are just "chat with your PDF" — a single embed → retrieve →
answer pipeline that doesn't really need an agent at all. This project is
built around a **ReAct agent that decides, per question, which of 4 tools
to call and in what order** — not a hardcoded pipeline. A narrow question
("what's the stock price?") triggers one tool call; a broad question
("full investment analysis") triggers several.

## Architecture

```
User question
      ↓
FastAPI (/analyze)
      ↓
ReAct Agent (agent.py) — reasons about which tools it needs
      ├── get_stock_price       (yfinance — live market data)
      ├── search_recent_news    (Tavily — web search)
      ├── query_company_filing  (Pinecone RAG — retrieval over uploaded filing)
      └── calculate_financial_ratio (deterministic math, no LLM involved)
      ↓
Raw findings (agent's synthesized answer after tool calls)
      ↓
Sequential LCEL chain (report_prompt | llm | StrOutputParser())
      ↓
Structured report: Summary / Key Financials / Risks / Recommendation
```

## Project structure

```
financial-agent/
├── tools/
│   ├── __init__.py        # exports all_tools list
│   ├── stock_price.py     # Tool 1: live stock price (yfinance)
│   ├── news_search.py     # Tool 2: web search (Tavily)
│   ├── filing_rag.py      # Tool 3: RAG over uploaded filing (Pinecone)
│   └── calculator.py      # Tool 4: deterministic financial math
├── agent.py                # ReAct agent + system prompt + report-formatting chain
├── main.py                 # FastAPI endpoints
├── run_eval.py              # LangSmith evaluation target/runner
├── document.txt            # fictional sample filing for testing
├── requirements.txt
└── .env.example
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your real keys:
   - `GROQ_API_KEY` — https://console.groq.com/keys (agent's LLM)
   - `PINECONE_API_KEY` + `PINECONE_INDEX_NAME` — https://app.pinecone.io (vector store + embeddings)
   - `TAVILY_API_KEY` — https://tavily.com (web search tool)
   - `LANGSMITH_API_KEY` — https://smith.langchain.com (evaluation & tracing)

3. Run the API:
   ```bash
   fastapi dev main.py
   ```

4. Test via the interactive docs at `http://localhost:8000/docs`, or Postman.

## Usage

**Step 1 — ingest a filing (one time per document):**
```json
POST /ingest-filing
{
  "pdf_path": "sample_data/techstack_q3_2026.txt",
  "namespace": "techstack-test"
}
```

**Step 2 — ask the agent a question:**
```json
POST /analyze
{
  "question": "Give a full investment analysis of TechStack Inc using namespace techstack-test"
}
```

Response includes both `raw_findings` (the agent's raw output after calling
tools) and `report` (the same findings reformatted into a structured report).

## Evaluation with LangSmith

The agent is evaluated against an 8-example dataset (`financial-agent-eval-v1`)
covering narrow single-fact questions, multi-tool questions, and deliberate
edge cases (e.g. a non-existent ticker, an ambiguous company name), with
`manual_review` feedback and automatic latency/token/cost tracking per run.

Running the eval:
```bash
python run_eval.py
```

Each run appears as a new experiment in the LangSmith dataset UI, so
successive iterations can be compared side by side on error rate and latency.

### Debugging process & fixes (error rate: 40% → 100% → 13%)

Iterating on this eval surfaced several distinct failure modes, each requiring
a different fix rather than one blanket solution:

| Problem | Root cause | Fix |
|---|---|---|
| `TypeError: unexpected keyword argument 'api_token'` | Wrong constructor kwarg for the Gemini client (`api_token` vs `api_key`) | Corrected the kwarg name |
| `tool_use_failed` on `get_stock_price` for an intentionally invalid ticker (`TSTK`) | Model hesitated when asked to format a tool call for an unfamiliar/fake symbol, emitting malformed pseudo-XML instead of a structured tool call | Added an explicit system-prompt instruction: always call the tool as given, even for unfamiliar tickers, and relay the tool's own "no data" response honestly |
| `tool_use_failed` on `search_recent_news`, intermittently | Groq's `llama-3.3-70b-versatile` occasionally emits `<function=name{...}>` instead of a valid structured tool call for this tool specifically — not fixed by prompting alone, and not fully fixed by retries alone since `temperature=0` made failures deterministic (identical malformed output on every retry) | Added retry with a small non-zero temperature so retries aren't guaranteed to repeat the same broken generation; added a regex-based fallback that extracts the function name + arguments straight out of Groq's error message and invokes the tool directly, recovering a real answer instead of crashing |
| Experiment jumped to 100% error rate | Not a code bug — hit Groq's free-tier daily token quota (100,000 TPD) after repeated full-dataset eval runs while iterating | Switched to testing single examples while debugging instead of re-running the full 8-example dataset each time; reserved full-dataset runs for after a fix looked solid |
| `GraphRecursionError` on a multi-step question | Agent didn't converge within the 8-step `recursion_limit` | Existing fallback message surfaces this clearly in `raw_findings` instead of crashing the pipeline, flagged for further investigation into tool-output handling |
| `RateLimitError` (429) mid-run | Same daily quota exhaustion, surfacing mid-agent rather than at the start | Added `RateLimitError` to the retry/except handling alongside `BadRequestError` so quota errors degrade gracefully instead of crashing the eval run |

Error rate across successive experiment runs on the same dataset:

```
baseline-run-cad8d866   40%   (initial run, before any fixes)
baseline-run-05e1271b   25%   (after ticker-handling prompt fix)
baseline-run-d7883594  100%   (Groq daily quota exhausted mid-iteration)
baseline-run-0acd1239   88%   (quota still recovering)
baseline-run-7c6a080f   13%   (retry + prompt + recovery fixes, fresh quota)
```

The 100%/88% spikes are a useful reminder that not every regression in an
eval is a code regression — ruling out infrastructure/quota causes before
re-diagnosing the agent itself saved a lot of wasted debugging.

### Remaining known issue

`llama-3.3-70b-versatile` on Groq still shows occasional malformed tool-call
generation for certain tool/query combinations that neither prompting,
retries, nor temperature tuning fully eliminate. The regex-based recovery
fallback mitigates the impact (a crash becomes a successful, if unusual,
answer) but doesn't address the root cause. The next planned step is
evaluating a different LLM provider/model for the agent's reasoning step to
see whether its native tool-calling is more reliable for this workload, while
keeping the rest of the pipeline (tools, prompt, LangSmith eval harness)
unchanged.

## Known limitation: entity conflation via web search

During testing, the `search_recent_news` tool occasionally returned real-world
information about an unrelated company that happens to share a name with the
fictional test company used here. Since the tool matches on name only, it has
no way to verify it's returning data about the *same* entity described in the
uploaded filing.

**Mitigation implemented:** the system prompt instructs the agent to cross-check
search results against verified details from the filing (sector, scale,
business description) and discard results that appear to describe a different
company, rather than blending them in silently. This reduces — but does not
fully eliminate — the risk, since the check ultimately relies on the LLM's
judgment rather than a hard, code-enforced rule. A stricter fix would involve
deterministic keyword/metadata matching before search results ever reach the
agent.

This is a known, general limitation of tool-using LLM agents (the same failure
mode search engines have when a name is ambiguous — e.g. "Apple" the company
vs. the fruit), not specific to this implementation.

## Security notes (not yet implemented — scoped for a future deployment pass)

This project currently runs locally with no public exposure, so the following
were deliberately deferred rather than skipped:

| Priority | Item | Why it matters |
|---|---|---|
| High | API key auth on endpoints | Currently anyone with the URL could call the API |
| High | Path traversal validation on `pdf_path` | Unvalidated file paths could read arbitrary files |
| Medium | Rate limiting | Protects against runaway API costs (Groq/Pinecone/Tavily) |
| Medium | Stricter input validation | Prevents malformed/abusive requests |
| Lower | Error message sanitization | Avoids leaking internal details in API responses |
| Lower | CORS tightening | `allow_origins` currently set for local dev only |

## What I'd add next (if continuing this project)

- Evaluate an alternative LLM provider/model for more reliable native tool-calling
- Deterministic (non-LLM) entity verification before passing search results to the agent
- Retry logic (`tenacity`) on all external API calls (Pinecone, Tavily, yfinance) —
  already in place for `get_stock_price` and `search_recent_news`, extend to
  the remaining tools
- The security items above, before any public deployment
- A `compare_companies` tool that runs the agent twice and diffs results
- A simple frontend showing tool calls live as the agent works