<div align="center">

# 📖 Agentic RAG Assistant — Developer Guide

**A complete walkthrough of the architecture, codebase, and development workflow.**

</div>

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Backend — Module-by-Module](#2-backend--module-by-module)
3. [Frontend — Component Tree & Hooks](#3-frontend--component-tree--hooks)
4. [Streaming Contract — End to End](#4-streaming-contract--end-to-end)
5. [RAG Pipeline — Ingestion to Retrieval](#5-rag-pipeline--ingestion-to-retrieval)
6. [Web Search — Tavily & DuckDuckGo Fallback](#6-web-search--tavily--duckduckgo-fallback)
7. [Evaluation Suite](#7-evaluation-suite)
8. [Configuration Reference](#8-configuration-reference)
9. [Development Workflow](#9-development-workflow)
10. [Docker Deployment](#10-docker-deployment)
11. [Project File Map](#11-project-file-map)
12. [Troubleshooting](#12-troubleshooting)
13. [Extending the Project](#13-extending-the-project)

---

## 1. Architecture Overview

The system has four layers, each cleanly separated:

```
┌──────────────────────────────────────────────────────────────────┐
│  FRONTEND — React 19 · Vite · Tailwind v4                       │
│  useAgentStream hook → fetch + ReadableStream → SSE parsing      │
└────────────────────────────┬─────────────────────────────────────┘
                             │  POST /api/chat/stream
                             │  ← SSE: token · tool_start · tool_end · sources · end
┌────────────────────────────▼─────────────────────────────────────┐
│  API — FastAPI · sse-starlette                                   │
│  /health  /ingest  /chat/stream  /feedback                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  AGENT — LangGraph create_agent (ReAct loop)                     │
│  Model: gpt-5.5 (reasoning)  ·  Memory: AsyncSqliteSaver         │
│  Tools: retrieve_documents · web_search                          │
└───────────┬────────────────────────────────────┬─────────────────┘
            │                                    │
┌───────────▼──────────┐          ┌──────────────▼────────────────┐
│  Chroma Vector Store  │          │  Tavily / DuckDuckGo (web)    │
│  text-embedding-3-sm  │          │  keyless fallback if no key   │
└───────────────────────┘          └───────────────────────────────┘
```

### Why these choices?

| Decision | Rationale |
|---|---|
| `create_agent` (not hand-rolled `StateGraph`) | The use case is a standard tool-calling ReAct loop; the prebuilt agent provides this out-of-the-box while staying fully traceable and streamable. |
| SSE (not WebSockets) | Data flows one direction (server→client). SSE is simpler, auto-reconnects, and needs no extra protocol. The frontend uses `fetch` + `ReadableStream` (not `EventSource`) so it can `POST` a JSON body. |
| Chroma (not Pinecone/Weaviate) | Zero-setup persisted vector store. Good for a demo; swap for a managed store in production. |
| Tailwind v4 with `@theme` | CSS-first configuration, class-based dark mode with oklch colors for perceptual uniformity. |

---

## 2. Backend — Module-by-Module

All backend code lives in `src/rag_agent/`. The package is built via [Hatch](https://hatch.pypa.io/) (`pyproject.toml`).

### `config.py` — Settings

```python
class Settings(BaseSettings):
    openai_api_key: str           # Required — LLMs + embeddings
    tavily_api_key: str | None    # Optional — web search
    langsmith_tracing: bool       # Master switch for tracing
    model_fast: str               # Routine-step model (gpt-5.4-mini)
    model_heavy: str              # Synthesis model (gpt-5.5)
    embedding_model: str          # text-embedding-3-small
    chroma_dir: str               # Persisted vector store path
    retriever_k: int              # Chunks per retrieval (default 4)
    ...
```

Uses `pydantic-settings` with `.env` file support. Also calls `load_dotenv(override=False)` to populate `os.environ` for libraries that read env directly (LangSmith, Tavily).

**Key insight**: The `@property web_backend` returns `"tavily"` or `"duckduckgo"` based on whether a Tavily key is configured.

---

### `llms.py` — Model Factories

Two factories:
- **`fast_model()`** — `ChatOpenAI(model="gpt-5.4-mini", streaming=True)` for routine steps.
- **`heavy_model()`** — `ChatOpenAI(model="gpt-5.5", streaming=True, reasoning={"effort": "medium"})` for planning + synthesis. The `reasoning` kwarg routes through the OpenAI Responses API.

Both include a `_assert_gpt5()` guard that raises `ValueError` if someone accidentally configures a `gpt-4*` or `gpt-3*` model.

---

### `embeddings.py` — Embedding Factory

```python
@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)
```

Cached singleton. Uses `text-embedding-3-small` (1536 dimensions).

---

### `vectorstore.py` — Chroma Store

```python
@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )
```

Persisted to `./chroma_db/`. The `collection_count()` helper returns the current number of stored chunks (used by the `/health` endpoint).

---

### `ingest.py` — Document Ingestion Pipeline

```
File → Loader (PyPDFLoader / TextLoader) → RecursiveCharacterTextSplitter → Chroma
```

- **Chunk size**: 1000 characters
- **Chunk overlap**: 150 characters
- **Supported formats**: `.pdf`, `.txt`, `.md`, `.markdown`
- **Auto-seeding**: `ensure_seeded()` ingests `data/sample_docs/` on first boot when the collection is empty.

Each document chunk preserves a `source` metadata field (the original filename), which is used for citation in the agent's answers.

---

### `tools.py` — Agent Tools

Two tools are registered with the agent:

#### `retrieve_documents(query: str) → str`
1. Invokes the Chroma retriever with `k=4`
2. Emits a `sources` payload on LangGraph's custom stream channel (via `_emit_sources`)
3. Returns formatted text with `[1]`, `[2]` citation markers

#### `web_search(query: str) → str`
Built dynamically by `_build_web_search_tool()`:
- **With Tavily key**: Uses `TavilySearch(max_results=5)`
- **Without**: Falls back to `DuckDuckGoSearchResults(output_format="list")` — keyless, but rate-limited

Both tools call `_emit_sources()` which uses `get_stream_writer()` to push source metadata to the LangGraph custom stream channel. This is how the frontend receives source cards **before** the agent finishes its answer.

---

### `agent.py` — LangGraph Agent

```python
def build_agent(checkpointer=None):
    return create_agent(
        model=heavy_model(),
        tools=build_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
```

The system prompt instructs the agent to:
1. Plan before acting
2. Prefer `retrieve_documents` for private docs, `web_search` for external facts
3. Combine both sources and note disagreements
4. Always cite with `[1]`, `[2]` inline markers
5. Admit when evidence is lacking

The agent uses `AsyncSqliteSaver` for per-`thread_id` conversation memory.

---

### `streaming.py` — LangGraph → SSE Bridge

This is the core streaming translation layer. It runs:

```python
agent.astream(
    {"messages": [...]},
    config=config,
    stream_mode=["messages", "updates", "custom"],
)
```

And maps each channel to SSE events:

| LangGraph Channel | Chunk Type | SSE Event |
|---|---|---|
| `messages` | `AIMessageChunk` with text | `token` |
| `updates` | `AIMessage` with `tool_calls` | `tool_start` |
| `updates` | `ToolMessage` | `tool_end` |
| `custom` | dict with `kind: "sources"` | `sources` |

**Design decisions:**
- Tool events come from the `updates` channel, not `messages` — streamed tool-call argument deltas are unreliable, so `tool_start`/`tool_end` are derived from completed node updates.
- Token text streams from `messages` — only `AIMessageChunk.text` is used, which concatenates text blocks and ignores reasoning tokens.
- Errors are caught and surfaced as SSE `error` events (not HTTP 500s mid-stream).
- A `seen_tool_starts` set prevents duplicate `tool_start` events.

---

### `api.py` — FastAPI Application

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | System health: models, web backend, tracing status, indexed chunks |
| `/api/ingest` | POST | Multipart file upload → split → embed → persist to Chroma |
| `/api/chat/stream` | POST | SSE stream of the agent's full turn |
| `/api/feedback` | POST | Forward thumbs up/down to LangSmith |

**Lifespan management:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_seeded()                              # Auto-ingest sample docs
    async with AsyncSqliteSaver.from_conn_string(...) as saver:
        app.state.agent = build_agent(checkpointer=saver)
        yield
```

The agent and its checkpointer are created once and shared across all requests. The `/api/ingest` endpoint writes uploaded files to temp, ingests them, then cleans up.

---

### `schemas.py` — Pydantic Models

Four request/response models:
- `ChatRequest` — `message` (required, min 1 char) + optional `thread_id`
- `FeedbackRequest` — `run_id`, `score`, optional `comment`, `key` defaults to `"user_score"`
- `IngestResponse` — `chunks_added` count + `files` list
- `HealthResponse` — full system status

---

## 3. Frontend — Component Tree & Hooks

```
App
├── AppShell (layout)
│   ├── Header
│   │   ├── HealthBadge (chunk count, web backend, tracing)
│   │   ├── UploadDialog (drag-and-drop file upload)
│   │   └── ThemeToggle (light/dark)
│   ├── ChatPanel (main area)
│   │   ├── MessageList
│   │   │   ├── Empty state (Sparkles icon + description)
│   │   │   ├── MessageBubble (user / assistant)
│   │   │   │   └── Markdown (react-markdown + remark-gfm)
│   │   │   │       ├── CitationChip ([1], [2] → scrolls to source card)
│   │   │   │       └── CodeBlock (Shiki syntax highlighting)
│   │   │   └── Live streaming cursor (animated pulse)
│   │   └── Composer (textarea + send/stop button)
│   └── Inspector sidebar (right panel, ≥lg)
│       ├── ActivityTimeline
│       │   └── TimelineStep (icon + status + detail)
│       └── SourcesPanel
│           └── SourceCard (numbered, with snippet + link)
```

### Key Hooks

#### `useAgentStream.ts`
The central state machine, implemented as a `useReducer`:

```typescript
interface StreamState {
  status: "idle" | "streaming" | "done" | "error";
  messages: ChatMessage[];      // committed messages
  answer: string;               // live, in-progress assistant text
  timeline: TimelineNode[];     // tool call timeline
  sources: Source[];             // retrieved/web sources
  error?: string;
  threadId?: string;
}
```

**Actions:**
- `user` — user sends a message → clears answer/timeline/sources, sets status to streaming
- `event` — an SSE event arrives → dispatched by `type` to update state
- `abort` — user stops → commits current answer, resets to idle

The `send()` function does `fetch("/api/chat/stream", { method: "POST", ... })` and reads the body as an async iterable via `parseSSEStream()`.

#### `useTheme.ts`
Uses `useSyncExternalStore` with a `MutationObserver` on `<html>` to reactively track the `.dark` class. Persists to `localStorage`.

### `lib/sse.ts` — POST-SSE Parser

Since `EventSource` only supports GET, this is a custom async generator that:
1. Pipes the `Response.body` through `TextDecoderStream`
2. Buffers text and splits on `\r?\n\r?\n` (frame boundaries)
3. Extracts `event:` and `data:` fields from each frame
4. `JSON.parse`s the data and yields typed `AgentEvent` objects

### `lib/events.ts` — Type Definitions

Mirrors the backend's SSE contract exactly:
```typescript
export type AgentEvent =
  | { type: "start"; thread_id: string }
  | { type: "token"; thread_id: string; delta: string; node?: string }
  | { type: "tool_start"; thread_id: string; id: string; tool: string; args?: unknown }
  | { type: "tool_end"; ... }
  | { type: "sources"; thread_id: string; tool: string; sources: Source[] }
  | { type: "error"; thread_id: string; message: string; fatal?: boolean }
  | { type: "end"; thread_id: string };
```

### CSS Design System (`index.css`)

Built on Tailwind v4's `@theme` directive with oklch colors for perceptual uniformity:

| Token | Light | Dark |
|---|---|---|
| `--color-bg` | `oklch(99% ...)` | `oklch(18% ...)` |
| `--color-surface` | `oklch(97% ...)` | `oklch(22% ...)` |
| `--color-accent` | `oklch(58% 0.18 264)` | `oklch(72% 0.15 264)` |

Custom animations: `step-enter` (timeline items slide in), `pulse-soft` (running tool indicator).

Custom `.prose-chat` styles for markdown rendering, Shiki dual-theme code blocks, and thin scrollbars.

---

## 4. Streaming Contract — End to End

Here's the complete flow of a single user message:

```
1. User types message in Composer → press Enter
2. useAgentStream.send(message) fires
3. fetch("POST /api/chat/stream", {message, thread_id})
4. FastAPI receives ChatRequest
5. agent_event_stream() is called → yields SSE dicts

   ┌─ agent.astream(stream_mode=["messages","updates","custom"]) ─┐
   │                                                                │
   │  (a) "updates" channel: AIMessage with tool_calls             │
   │      → yield SSE tool_start {id, tool, args}                  │
   │                                                                │
   │  (b) "custom" channel: {kind:"sources", sources:[...]}       │
   │      → yield SSE sources {tool, sources}                      │
   │                                                                │
   │  (c) "updates" channel: ToolMessage                           │
   │      → yield SSE tool_end {id, tool, ok, result_preview}     │
   │                                                                │
   │  (d) "messages" channel: AIMessageChunk with .text            │
   │      → yield SSE token {delta, node}                          │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘

6. EventSourceResponse streams SSE to the client
7. parseSSEStream() yields AgentEvents
8. useAgentStream reducer updates state per event
9. React re-renders: timeline, sources, answer text
10. On "end" event: answer committed to messages[]
```

---

## 5. RAG Pipeline — Ingestion to Retrieval

### Ingestion Flow

```
Upload (UI / boot)
     │
     ▼
 File loader (PyPDFLoader / TextLoader)
     │
     ▼
 RecursiveCharacterTextSplitter
   chunk_size=1000, overlap=150
     │
     ▼
 OpenAI text-embedding-3-small
     │
     ▼
 Chroma (persist_directory="./chroma_db")
```

### Retrieval Flow (at query time)

```
Agent calls retrieve_documents(query)
     │
     ▼
 Chroma.as_retriever(k=4).invoke(query)
     │
     ▼
 Top-4 chunks returned
     │
     ├── _emit_sources() → custom stream → SSE sources event
     │
     └── Formatted as "[1] (source: filename)\n<content>"
         → returned to the agent as tool result
```

### Source Attribution

The agent receives numbered passages `[1]`, `[2]`, etc. The system prompt instructs it to cite these numbers in its final answer. The frontend's `Markdown` component detects `[n]` patterns and replaces them with clickable `CitationChip` components that scroll to the corresponding `SourceCard`.

---

## 6. Web Search — Tavily & DuckDuckGo Fallback

```
TAVILY_API_KEY set?
    │
    ├── YES → TavilySearch(max_results=5)
    │         Structured results with title, url, content, score
    │
    └── NO  → DuckDuckGoSearchResults(output_format="list")
              Keyless, but subject to rate limits
              Errors caught and surfaced as "temporarily unavailable"
```

Both paths:
1. Map results to a uniform `Source` schema (`id`, `kind`, `title`, `url`, `snippet`, `score`)
2. Emit sources via the custom stream channel
3. Return formatted text to the agent

---

## 7. Evaluation Suite

All eval scripts are in `evals/`. They require `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`.

### `create_dataset.py`

Seeds a LangSmith dataset (`sainath-rag-qa`) with 12 question/reference-answer pairs:
- 10 are answerable from the Aurora sample docs (tests RAG retrieval)
- 2 require general knowledge (tests web search fallback)

```bash
uv run python -m evals.create_dataset
```

### `run_evals.py`

Runs all 12 examples through the agent and scores with 4 evaluators:

| Evaluator | Type | What it measures |
|---|---|---|
| `correctness` | LLM-as-judge (openevals) | Agent answer vs. reference answer |
| `groundedness` | LLM-as-judge (RAG) | Is the answer supported by retrieved context? |
| `retrieval_relevance` | LLM-as-judge (RAG) | Is the retrieved context relevant to the question? |
| `cites_sources` | Heuristic (regex) | Does the answer contain `[n]`, URLs, or `(source:` markers? |

```bash
uv run python -m evals.run_evals
```

**Note**: `groundedness` and `retrieval_relevance` return `score=None` for web-search-only questions (no document context to evaluate against). This is by design.

### `run_pairwise.py`

Comparative experiment: runs the dataset against `gpt-5.5` and `gpt-5.4-mini`, then uses an LLM judge to pick the better answer per example.

```bash
uv run python -m evals.run_pairwise
```

Produces a side-by-side comparison view in the LangSmith UI.

---

## 8. Configuration Reference

All settings are defined in `src/rag_agent/config.py` and loaded from environment variables or `.env`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | LLMs + embeddings (gpt-5 series) |
| `TAVILY_API_KEY` | ➖ | — | Web search; falls back to DuckDuckGo if unset |
| `LANGSMITH_TRACING` | ➖ | `false` | Master switch for LangSmith tracing |
| `LANGSMITH_API_KEY` | ➖ | — | Required if tracing is enabled |
| `LANGSMITH_PROJECT` | ➖ | `sainath-rag-agent` | LangSmith project name for traces |
| `LANGSMITH_ENDPOINT` | ➖ | `https://api.smith.langchain.com` | LangSmith API endpoint |
| `MODEL_FAST` | ➖ | `gpt-5.4-mini` | Routine-step model |
| `MODEL_HEAVY` | ➖ | `gpt-5.5` | Planning / synthesis model |
| `EMBEDDING_MODEL` | ➖ | `text-embedding-3-small` | Embedding model |
| `REASONING_EFFORT` | ➖ | `medium` | Reasoning effort for gpt-5.5 |
| `CHROMA_DIR` | ➖ | `./chroma_db` | Chroma persist directory |
| `CHROMA_COLLECTION` | ➖ | `documents` | Chroma collection name |
| `SQLITE_PATH` | ➖ | `./memory.sqlite` | Agent conversation memory |
| `RETRIEVER_K` | ➖ | `4` | Number of chunks retrieved per query |
| `SAMPLE_DOCS_DIR` | ➖ | `./data/sample_docs` | Auto-ingested on first boot |

### Setup

```bash
cp .env.example .env
# Edit .env and fill in at minimum OPENAI_API_KEY
```

---

## 9. Development Workflow

### Prerequisites

- **Python 3.12** (managed by `uv`)
- **Node.js 22+** (for frontend)
- **uv** — Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

### Backend

```bash
# Install dependencies
uv sync

# Run the backend (auto-reloads on changes)
uv run uvicorn rag_agent.api:app --reload
# → http://localhost:8000

# Run linting
uv run ruff check src/

# Run tests
uv run pytest
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run the dev server (proxies /api → :8000)
npm run dev
# → http://localhost:5173

# Type-check without emitting
npm run typecheck

# Production build
npm run build
```

### Full Local Stack

Open two terminals:
```bash
# Terminal 1: backend
uv run uvicorn rag_agent.api:app --reload

# Terminal 2: frontend
cd frontend && npm run dev
```

Visit `http://localhost:5173`. The Vite dev server proxies `/api` to `localhost:8000`, so everything works as a single origin.

---

## 10. Docker Deployment

### Quick Start

```bash
cp .env.example .env              # fill in API keys
export OPENAI_API_KEY=sk-...      # also export for compose

docker compose up -d --build
# UI:  http://localhost:8080
# API: http://localhost:8000
```

### How It Works

**`docker-compose.yml`** defines two services:

| Service | Image | Port | Purpose |
|---|---|---|---|
| `backend` | `Dockerfile.backend` (uv + Python 3.12) | 8000 | FastAPI + LangGraph agent |
| `frontend` | `frontend/Dockerfile` (Node build + nginx) | 8080 | Static UI + reverse proxy |

**Backend container:**
- Uses `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
- Runs `uv sync --no-dev` to install dependencies
- Copies `data/` for first-boot ingestion
- Mounts a `state` volume for Chroma + SQLite persistence

**Frontend container:**
- Multi-stage build: Node 26 → build → nginx 1.27
- `nginx.conf` serves static files at `/` and proxies `/api/` to `backend:8000`
- SSE-specific config: `proxy_buffering off`, `proxy_read_timeout 3600s`

### Volumes

The `state` named volume persists:
- `chroma_db/` — vector store (survives container restarts)
- `memory.sqlite` — conversation memory

### Teardown

```bash
docker compose down           # stop and remove containers
docker compose down -v        # also remove the state volume (resets everything)
```

---

## 11. Project File Map

```
agentic-rag-assistant/
│
├── src/rag_agent/                   # Python backend
│   ├── __init__.py                  # Package marker
│   ├── config.py                    # Settings via pydantic-settings
│   ├── llms.py                      # ChatOpenAI factories (fast + heavy)
│   ├── embeddings.py                # OpenAIEmbeddings singleton
│   ├── vectorstore.py               # Chroma store singleton
│   ├── ingest.py                    # File → chunks → Chroma pipeline
│   ├── tools.py                     # retrieve_documents + web_search
│   ├── agent.py                     # LangGraph create_agent
│   ├── streaming.py                 # astream → SSE event mapping
│   ├── schemas.py                   # Pydantic request/response models
│   └── api.py                       # FastAPI app (/health /ingest /chat/stream /feedback)
│
├── frontend/                        # React frontend
│   ├── src/
│   │   ├── main.tsx                 # Entry point
│   │   ├── App.tsx                  # Root component
│   │   ├── index.css                # Design system (Tailwind v4 @theme)
│   │   ├── hooks/
│   │   │   ├── useAgentStream.ts    # SSE state machine
│   │   │   └── useTheme.ts          # Dark mode toggle
│   │   ├── lib/
│   │   │   ├── api.ts               # fetchHealth, uploadDocuments
│   │   │   ├── events.ts            # AgentEvent TypeScript types
│   │   │   ├── sse.ts               # POST-SSE parser (async generator)
│   │   │   └── utils.ts             # cn() class joiner
│   │   └── components/
│   │       ├── layout/
│   │       │   ├── AppShell.tsx      # Main layout (header + chat + inspector)
│   │       │   └── ThemeToggle.tsx   # Light/dark toggle button
│   │       ├── chat/
│   │       │   ├── ChatPanel.tsx     # Chat area wrapper
│   │       │   ├── Composer.tsx      # Input textarea + send/stop
│   │       │   ├── MessageList.tsx   # Scrollable message list
│   │       │   ├── MessageBubble.tsx # User/assistant message bubble
│   │       │   └── Markdown.tsx      # Markdown renderer + citation chips
│   │       ├── agent/
│   │       │   ├── ActivityTimeline.tsx  # Tool call timeline panel
│   │       │   ├── TimelineStep.tsx      # Individual step (icon + status)
│   │       │   └── stepIcons.tsx         # Tool → icon/label mapping
│   │       ├── sources/
│   │       │   ├── SourcesPanel.tsx      # Sources panel wrapper
│   │       │   └── SourceCard.tsx        # Individual source card
│   │       └── upload/
│   │           └── UploadDialog.tsx      # Drag-and-drop file upload modal
│   ├── public/                      # Static assets (favicons)
│   ├── index.html                   # HTML shell + dark mode flash prevention
│   ├── package.json                 # Dependencies
│   ├── vite.config.ts               # Vite + Tailwind + /api proxy
│   ├── tsconfig.json                # TypeScript config root
│   ├── tsconfig.app.json            # App TS config
│   ├── tsconfig.node.json           # Node TS config (vite.config.ts)
│   ├── Dockerfile                   # Multi-stage Node → nginx
│   ├── nginx.conf                   # SPA fallback + /api reverse proxy
│   └── .dockerignore                # Exclude node_modules etc.
│
├── evals/                           # LangSmith evaluation suite
│   ├── __init__.py
│   ├── create_dataset.py            # Seed the Q/A dataset
│   ├── run_evals.py                 # Run 4-evaluator experiment
│   └── run_pairwise.py             # Model comparison experiment
│
├── data/sample_docs/                # Demo corpus (auto-ingested)
│   ├── aurora_product_spec.md
│   ├── aurora_faq.md
│   └── aurora_changelog.md
│
├── docs/                            # Screenshots for README
│   ├── hero-dark.png
│   ├── chat-light.png
│   └── empty-state.png
│
├── pyproject.toml                   # Python project metadata + deps
├── uv.lock                         # Lockfile
├── .python-version                  # 3.12
├── .env.example                     # Template for environment variables
├── .gitignore
├── .dockerignore
├── docker-compose.yml               # Backend + frontend services
├── Dockerfile.backend               # Backend container
├── LICENSE                          # MIT
├── README.md                        # Project overview + architecture docs
└── GUIDE.md                         # This file
```

---

## 12. Troubleshooting

### Backend won't start

| Symptom | Cause | Fix |
|---|---|---|
| `ValidationError: openai_api_key` | Missing API key | Set `OPENAI_API_KEY` in `.env` or environment |
| `ModuleNotFoundError: rag_agent` | Dependencies not installed | Run `uv sync` |
| `Refusing to use obsolete chat model` | `MODEL_FAST` or `MODEL_HEAVY` set to gpt-4* | Use gpt-5 series models |
| Port 8000 in use | Another process on :8000 | Kill it or use `--port 8001` |

### Frontend won't start

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot find module` | Missing deps | Run `npm install` in `frontend/` |
| API calls return 502/504 | Backend not running | Start the backend first |
| SSE stream seems buffered | Proxy buffering | Check Vite proxy config or nginx `proxy_buffering off` |

### RAG returns "No relevant passages"

1. Check `/api/health` → `documents_indexed` should be > 0
2. If 0, the sample docs weren't ingested. Check that `data/sample_docs/` exists and contains `.md` files
3. Upload your own docs via the UI to test

### Docker issues

| Symptom | Fix |
|---|---|
| Backend can't connect to OpenAI | Ensure `OPENAI_API_KEY` is exported in shell AND in `.env` |
| Data lost on restart | The `state` volume persists data. Use `docker compose down` (without `-v`) to keep it |
| Frontend shows blank page | Check browser console. Ensure the nginx proxy is correctly routing `/api/` |

### LangSmith tracing not working

1. Set `LANGSMITH_TRACING=true` in `.env`
2. Set `LANGSMITH_API_KEY=lsv2_pt_...`
3. Both must be set; tracing won't enable with just one

---

## 13. Extending the Project

### Adding a new tool

1. Define the tool in `src/rag_agent/tools.py`:
   ```python
   @tool("my_tool")
   def my_tool(query: str) -> str:
       """Description for the agent."""
       # ... do work ...
       _emit_sources("my_tool", sources_list)  # optional
       return result_text
   ```

2. Add it to `build_tools()`:
   ```python
   def build_tools() -> list:
       return [retrieve_documents, _build_web_search_tool(), my_tool]
   ```

3. Update the system prompt in `agent.py` to describe the new tool.

4. Optionally add a UI icon in `frontend/src/components/agent/stepIcons.tsx`:
   ```tsx
   if (tool === "my_tool") return <MyIcon className={className} />;
   ```

### Adding a new document format

1. In `ingest.py`, add the extension to `_TEXT_EXTS` or create a new set
2. Add a loader branch in `_load_file()`
3. Update the `accept` attribute in `UploadDialog.tsx`

### Switching to a managed vector store

Replace `vectorstore.py` with your store's LangChain integration (Pinecone, Weaviate, Qdrant, etc.). The rest of the codebase only calls `get_vectorstore()`, so the change is localized.

### Adding authentication

The README roadmap mentions auth. To add it:
1. Add auth middleware to FastAPI (e.g., JWT bearer tokens)
2. Pass the user ID to the agent's config for per-user document isolation
3. Update the Chroma collection to include user-scoped metadata filtering

---

<div align="center">

*Built by [Sainath S Borule](https://github.com/SainathSBorule) — AI/ML Engineer*

</div>
