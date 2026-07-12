# Financial Research Analyst Agent

An AI agent that autonomously researches a company using multiple tools — live
stock data, web search, RAG over an uploaded filing, and deterministic
financial calculations — reasons about which tools it actually needs for a
given question, and generates a structured investment report.

Built with **LangChain**, **LangGraph** (ReAct agent), **Groq** (LLM
inference), **Pinecone** (vector store + hosted embeddings), and **FastAPI**.

## Tech stack & skills demonstrated

- **Python** — FastAPI, Pydantic, async/await, custom class design
- **LangChain / LangGraph** — ReAct agents, LCEL chains (`prompt | llm | parser`),
  custom tools (`@tool`), custom `Embeddings` adapter, document loaders, text splitting
- **LLM integration** — Groq (`llama-3.3-70b-versatile`) for agent reasoning
- **Vector databases / RAG** — Pinecone (hosted embeddings + vector storage,
  namespace-based multi-tenancy)
- **External APIs** — yfinance (market data), Tavily (web search)
- **API design** — REST endpoints, request/response validation, error handling
- **AI safety / reliability** — system prompt guardrails against data
  fabrication and entity conflation, documented known limitations
- **Engineering practices** — environment variable management, `.gitignore`
  hygiene, modular file structure, retry-handling considerations, security
  hardening checklist for future deployment

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
├── document.txt           # fictional sample filing for testing
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

- Deterministic (non-LLM) entity verification before passing search results to the agent
- Retry logic (`tenacity`) on all external API calls (Pinecone, Tavily, yfinance)
- The security items above, before any public deployment
- A `compare_companies` tool that runs the agent twice and diffs results
- LangSmith tracing to visualize the agent's reasoning steps
- A simple frontend showing tool calls live as the agent works