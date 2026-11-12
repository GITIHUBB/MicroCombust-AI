"""FastAPI backend for the "Flame in Freefall" microgravity combustion dashboard.

Routes
------
``GET      /``               health probe
``POST     /query-agent``    RAG question answering over the NASA paper index
``GET/POST /rank-materials`` material flammability risk ranking

Rebuilt from the surviving ``__pycache__/main.cpython-313.pyc`` bytecode. The
routes and the Gemini prompt are unchanged; added on top are CORS, tolerant
request parsing (the dashboard posts ``query`` while the original model only
accepted ``question``) and the response keys ``normalizeAnswer`` /
``normalizeRanking`` in the dashboard actually read, so the UI shows live
results instead of silently dropping into demo mode.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
from contextlib import asynccontextmanager
from typing import Any, Optional
from urllib.parse import quote

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION = "langchain"
GEMINI_MODEL = "gemini-2.5-flash"
# Dynamic without drifting: 0.7 keeps replies varied from turn to turn while
# still landing on the grounded, cited answers the dashboard depends on.
GEMINI_TEMPERATURE = 0.7
# Conversation turns replayed to the model on every request. Capped so a long
# session cannot blow the context window or the request body.
MAX_HISTORY_TURNS = 40

# Secrets come from `.env` next to this file so the key never lands in source.
# Put `GEMINI_API_KEY=...` in that file and restart - nothing else changes.
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Retrieval shape. The index stores several chunks per page (and repeats some),
# so we pull 10 candidates, drop near-duplicates and anything barely resembling
# the question, then re-rank them against the query before keeping the best few.
MAX_PASSAGES = 4  # top 3-4 is enough grounding without diluting the prompt
FETCH_K = 10  # candidates pulled from Chroma before re-ranking
# L2 distance above which a chunk is effectively unrelated to the query.
# confidence_of() maps 1.40 -> ~0.02 cosine similarity; off-topic probes on
# this index scored 1.46+.
MAX_DISTANCE = 1.40

COHERE_URL = "https://api.cohere.com/v2/rerank"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Materials the dashboard ranks: base flammability (0-1) plus hazard wording.
MATERIALS = [
    {"id": "PMMA", "label": "PMMA (Acrylic)", "base": 0.80,
     "risk_level": "High Hazard", "spread": "Fast", "extinction": "0.18 m/s airflow"},
    {"id": "Silicone", "label": "Silicone Rubber", "base": 0.75,
     "risk_level": "High Hazard", "spread": "Moderate-Fast", "extinction": "0.12 m/s airflow"},
    {"id": "Cotton", "label": "Cotton Fabric", "base": 0.55,
     "risk_level": "Caution", "spread": "Moderate", "extinction": "0.08 m/s airflow"},
    {"id": "Nomex", "label": "Nomex Fabric", "base": 0.45,
     "risk_level": "Caution", "spread": "Slow", "extinction": "0.05 m/s airflow"},
    {"id": "Kevlar", "label": "Kevlar Blend", "base": 0.25,
     "risk_level": "Low Hazard / Safe", "spread": "Self-Extinguishing", "extinction": "N/A"},
]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _tier(score: float) -> str:
    if score >= 0.6:
        return "High"
    return "Medium" if score >= 0.35 else "Low"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _file_name(source: Any) -> str:
    return str(source or "").replace("\\", "/").split("/")[-1]


# --- Retrieval, not keywords, decides what is grounded ------------------------
# There is deliberately no domain-vocabulary gate here. `similarity_search`
# already discards anything past MAX_DISTANCE, so a greeting, small talk or an
# off-topic question simply comes back with no passages and is answered
# conversationally by Gemini. A keyword list could only ever guess, and the
# guesses used to route perfectly good questions into a canned reply.


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the ONNX embedding session so the very first user query is quick.
    try:
        similarity_search("flame spread in microgravity", k=1)
    except Exception:
        pass
    yield


app = FastAPI(title="NASA Space Fire Safety API", lifespan=lifespan)

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

# Ensure allow_origins is always a list for CORSMiddleware
if FRONTEND_URL == "*":
    origins = ["*"]
else:
    origins = [FRONTEND_URL]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_collection: Any = None
_collection_error: Optional[str] = None


def get_collection():
    """Open the persisted Chroma index, or ``None`` when it is unavailable."""
    global _collection, _collection_error
    if _collection is None and _collection_error is None:
        try:
            _collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
        except Exception as exc:  # pragma: no cover - defensive
            _collection_error = str(exc)
    return _collection


def similarity_search(question: str, k: int = MAX_PASSAGES):
    """Return the ``k`` best chunks: fetch 10 -> filter -> de-dupe -> re-rank."""
    collection = get_collection()
    if collection is None:
        return []
    try:
        result = collection.query(query_texts=[question], n_results=FETCH_K)
    except Exception:
        return []
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    candidates = []
    seen = set()
    for text, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        text = text or ""
        if _as_float(dist, 99.0) > MAX_DISTANCE:
            continue
        # Same source + page + opening words means the passage was ingested
        # twice; keeping it twice would waste a slot on a repeat.
        key = (str(meta.get("source")), meta.get("page"), text[:120])
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"metadata": meta, "page_content": text, "distance": dist})
        if len(candidates) >= FETCH_K:
            break

    # Re-rank against the query before anything reaches the prompt.
    return rerank(question, candidates, k)


def confidence_of(distance: Any) -> float:
    """Map an L2 distance between unit-length MiniLM vectors to a 0-1 score."""
    try:
        gap = float(distance)
    except (TypeError, ValueError):
        return 0.5
    # For unit vectors: cosine similarity = 1 - d^2 / 2.
    return round(_clamp(1.0 - (gap * gap) / 2.0, 0.0, 1.0), 4)


# --- Candidate re-ranking ----------------------------------------------------
# Chroma scores each passage on its own, so a weakly related chunk can still
# clear the relevance floor and then dilute the prompt. Re-ranking the 10
# candidates against the actual query is what keeps that out. Three tiers,
# best available first:
#   1. Cohere Rerank  - set COHERE_API_KEY in `.env`
#   2. cross-encoder   - `pip install sentence-transformers`
#   3. lexical fusion  - no dependencies, always available
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the "
    "this to was what when which with you your".split()
)

_cross_encoder: Any = None
_cross_encoder_failed = False


def _tokens(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _cohere_ranks(query: str, docs: list) -> Optional[list[int]]:
    """Cohere Rerank ordering, or ``None`` so the next tier can take over."""
    key = os.getenv("COHERE_API_KEY", "").strip()
    if not key:
        return None
    try:
        import requests  # already installed as a chromadb dependency
    except ImportError:
        return None
    try:
        response = requests.post(
            COHERE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
                "query": query,
                "documents": [doc["page_content"][:2000] for doc in docs],
                "top_n": len(docs),
            },
            timeout=20,
        )
        response.raise_for_status()
        order = [
            int(item["index"])
            for item in response.json().get("results", [])
            if 0 <= int(item["index"]) < len(docs)
        ]
        return order or None
    except Exception:
        # Bad key, quota, offline - any problem falls through to the next tier
        # rather than failing the query the user is waiting on.
        return None


def _cross_encoder_ranks(query: str, docs: list) -> Optional[list[int]]:
    """Cross-encoder ordering, when sentence-transformers is installed."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is None and not _cross_encoder_failed:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(CROSS_ENCODER)
        except Exception:
            _cross_encoder_failed = True
            return None
    if _cross_encoder is None:
        return None
    try:
        scores = _cross_encoder.predict([(query, doc["page_content"]) for doc in docs])
    except Exception:
        return None
    return sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)


def _lexical_ranks(query: str, docs: list) -> list[int]:
    """Dependency-free fallback: fuse Chroma's semantic score with token overlap.

    A cross-encoder reads query and passage together; this is the cheap
    approximation of that, and it still beats trusting Chroma's order as-is.
    """
    query_tokens = _tokens(query)
    scored = []
    for index, doc in enumerate(docs):
        semantic = confidence_of(doc.get("distance"))
        overlap = len(query_tokens & _tokens(doc["page_content"])) / max(
            len(query_tokens), 1
        )
        scored.append((0.6 * semantic + 0.4 * overlap, index))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored]


def rerank(query: str, docs: list, top_k: int = MAX_PASSAGES) -> list:
    """Order *docs* best-first, then keep only the top ``top_k``."""
    if not docs:
        return []
    order = (
        _cohere_ranks(query, docs)
        or _cross_encoder_ranks(query, docs)
        or _lexical_ranks(query, docs)
    )
    return [docs[i] for i in order[:top_k]]


def rerank_provider() -> str:
    """Which tier the next query would use - reported on the health endpoint."""
    if os.getenv("COHERE_API_KEY", "").strip():
        return "cohere"
    if importlib.util.find_spec("sentence_transformers"):
        return "cross-encoder"
    return "lexical-fusion"


def source_url(source: Any) -> str:
    """Build a citable NASA link from an indexed file name."""
    stem = os.path.splitext(_file_name(source))[0].strip()
    if re.fullmatch(r"\d{6,}", stem):
        return f"https://ntrs.nasa.gov/citations/{stem}"
    return f"https://ntrs.nasa.gov/search?q={quote(stem) or 'microgravity+combustion'}"


def source_title(meta: dict, source: Any) -> str:
    name = _file_name(source) or "NASA source"
    page = meta.get("page_label") or meta.get("page")
    return f"{name} (p. {page})" if page is not None else name


def _api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the Gemini key, rejecting empty values and `.env` placeholders."""
    key = (explicit or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key or key.lower().startswith(("your-", "paste-", "changeme", "xxx", "<")):
        return None
    return key


def _no_key_answer() -> str:
    """Only when no Gemini key is set *and* retrieval found nothing.

    Infrastructure message, not a chat path: with no model there is genuinely
    nothing to talk to. Never reachable while the key is configured, so it
    cannot make one kind of question look different from another.
    """
    return (
        "I can't answer that right now - there's no Gemini key configured, so "
        "I have no model to think with, and the NASA index had nothing close "
        "enough to quote from.\n\n"
        "Add GEMINI_API_KEY to `.env` next to main.py and restart the server to "
        "turn me back on, conversation memory included."
    )


def _passage_list(question: str, docs: list) -> str:
    """Numbered, truncated list of the passages an answer would be grounded in."""
    lines = [f'Closest passages to "{question}":', ""]
    for position, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        source = _file_name(meta.get("source")) or "unknown source"
        page = meta.get("page_label") or meta.get("page", "N/A")
        snippet = " ".join(str(doc["page_content"]).split())
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        lines.append(f"{position}. {source}, p. {page}")
        lines.append(f"   {snippet}")
        lines.append("")
    lines.append("Open a citation below to read the full document on NTRS.")
    return "\n".join(lines)


def _preview_answer(question: str, docs: list) -> str:
    """Retrieval preview shown when no Gemini key is configured.

    The one path where an answer cannot come from Gemini, because there is no
    key to call it with - so it says so instead of imitating a reply.
    """
    return "\n".join(
        [
            "I can't write this one in my own voice yet - there's no Gemini key "
            "configured, so all I have is the raw retrieval below rather than a "
            "proper answer.",
            "Add GEMINI_API_KEY to `.env` next to main.py and restart the "
            "server and I'll answer normally.",
            "",
            _passage_list(question, docs),
        ]
    )


def _unavailable_answer(question: str, docs: list, exc: BaseException) -> str:
    """Handed back once retries are exhausted.

    One voice and one shape whether or not retrieval had anything, so a science
    question failing mid-session does not surface a visibly different style of
    message from a casual one. Only reachable when the Gemini call itself has
    failed, after 3 attempts.
    """
    reason = _describe_error(exc)
    if not _is_transient(exc):
        # Permanent (bad key, blocked prompt): no point waiting, just say so.
        lead = (
            f"I couldn't reach Gemini for that one ({reason}). Check "
            "GEMINI_API_KEY in `.env` next to main.py and restart the server."
        )
    else:
        lead = (
            f"Gemini is busy right now ({reason}) - I retried 3 times with "
            "backoff and it still didn't recover, so I'd rather say so than "
            "guess at an answer."
        )
    if not docs:
        return lead + " Give it another go in a moment."
    return (
        lead
        + "\n\nHere is what I would have grounded the answer in:\n\n"
        + _passage_list(question, docs)
    )


def _settings_block(
    oxygen: Optional[float],
    pressure: Optional[float],
    airflow: Optional[float],
    fuel: Optional[str],
) -> Optional[str]:
    """Render the live dashboard controls as prompt context, when supplied.

    Returns ``None`` when the client sent none of them, so a plain curl does
    not get a block full of "not set".
    """
    if oxygen is None and pressure is None and airflow is None and not fuel:
        return None

    def fmt(value: Optional[float], unit: str = "") -> str:
        return "not set" if value is None else f"{value:g}{unit}"

    return "\n".join(
        [
            "ACTIVE DASHBOARD SETTINGS",
            "The user currently has these controls set on the dashboard:",
            f"- Oxygen concentration: {fmt(oxygen, ' %')}",
            f"- Ambient pressure: {fmt(pressure, ' kPa')}",
            f"- Forced airflow velocity: {fmt(airflow, ' cm/s')}",
            f"- Fuel specimen: {fuel or 'not set'}",
            "",
            "Reference these when they matter to the question - for example when "
            "the user asks about the selected material or says 'at these "
            "settings' - and say plainly when the retrieved passages do not cover "
            "that exact combination. Do not claim to have simulated these values: "
            "the sliders feed a ranking heuristic, not a physics solver.",
        ]
    )


# The persona contract. Deliberately a module-level constant so it is obvious
# (and testable) that every request - combustion physics, general trivia, small
# talk or a polite decline - is generated from one identical instruction.
# Query-type-specific formatting rules are absent on purpose: rules like
# "casual replies get 2-4 sentences" or "science replies stay under 300 words"
# are what made the two kinds of answer read like different assistants.
_PERSONA_INSTRUCTION = """You are the official AI guide for the 'Flame in Freefall' microgravity combustion dashboard.

Tone & Formatting Rules:
- Voice: Warm, conversational, articulate, and engaging across ALL topics.
- Uniform Style: Maintain a consistent, natural formatting style regardless of whether answering a science query, casual chat, or general knowledge question. Avoid changing into a robotic or overly template-driven format for science questions.
- Domain Expertise: Answer microgravity, fire-safety, and simulation questions with expert clarity without breaking character.
- General Knowledge: Answer general trivia (e.g., "Who is the richest person in the world?") directly and naturally using your general knowledge.
- Out-of-Scope Requests: For unsupported actions (like setting alarms or checking live device settings), humbly and politely decline in 1 short, natural sentence.
- Context Memory: Use the provided chat history to remember user details (such as names or prior conversation topics)."""


def _system_instruction(
    gravity_filter: Optional[str],
    settings_block: Optional[str] = None,
) -> str:
    """One system instruction, byte-identical in voice for every query type.

    ``_PERSONA_INSTRUCTION`` owns the tone. Everything appended below it is
    factual state (dashboard controls) or retrieval plumbing (how to treat
    attached passages) - never a competing rule about how a reply should be
    shaped, so there is nothing here that can push a science answer into a
    different format from a casual one.
    """
    gravity = (gravity_filter or "").strip() or "not specified"
    parts = [
        _PERSONA_INSTRUCTION,
        "",
        "Dashboard context:",
        f"- Gravity mode selected: {gravity}.",
        "- The oxygen / pressure / airflow sliders and 'RUN SIMULATION & RANK "
        "RISK' drive a ranking heuristic over NASA data rather than a physics "
        "simulation, so describe that output as an estimate.",
        "",
        "Retrieved context:",
        "- PASSAGES may be attached to the user's message. When they answer the "
        "question, use them as your source and cite inline like "
        "[19930010989.pdf p.202] - keeping this same voice and formatting as "
        "everything else.",
        "- When they do not answer the question, ignore them instead of forcing "
        "a citation.",
        "- When none are attached, answer from your own knowledge and never "
        "invent figures.",
    ]
    if settings_block:
        parts.extend(["", settings_block])
    return "\n".join(parts)


def _user_turn(question: str, context_text: str) -> str:
    """The current turn: the user's message, plus passages when retrieval found any.

    The question is always first and unframed, so a casual or general-knowledge
    message reaches Gemini completely bare - identical to how it arrives when
    nothing was retrieved. Passages are presented as optional source material
    rather than as a cue to switch into a formal answer format.
    """
    passages = context_text.strip()
    if not passages:
        return question
    return (
        f"{question}\n\n"
        "Retrieved passages (use these only if they actually answer the "
        "question above):\n" + passages
    )


# --- Gemini text generation with retry ---------------------------------------
# Only failures a retry can plausically fix are retried: 503 unavailability, 429
# rate limits and timeouts. Everything else (bad key = 400, blocked prompt, ...)
# is permanent, and re-running it would only delay an error the user needs now.
_TRANSIENT_CODES = {408, 425, 429, 500, 502, 503, 504}
_TRANSIENT_WORDS = (
    "rate limit",
    "rate_limit",
    "rate-limit",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "unavailable",
    "overloaded",
    "timeout",
    "timed out",
    "temporarily",
    "connection reset",
)


def _is_transient(exc: BaseException) -> bool:
    """True when retrying could plausibly succeed (503 / rate limit / timeout)."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        code = int(getattr(exc, "code", None))
    except (TypeError, ValueError):
        code = None
    if code is not None:
        return code in _TRANSIENT_CODES
    text = str(exc).lower()
    return any(word in text for word in _TRANSIENT_WORDS)


def _describe_error(exc: BaseException) -> str:
    """Short human-readable reason, never a raw exception dump."""
    try:
        code = int(getattr(exc, "code", None))
    except (TypeError, ValueError):
        code = None
    if code == 429:
        return "rate limit reached (HTTP 429)"
    if code == 503:
        return "service unavailable (HTTP 503)"
    if code is not None:
        return f"HTTP {code} {type(exc).__name__}"
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:160] + "..." if len(text) > 160 else text


def _clean_history(raw: Any) -> list:
    """Normalise the client's history into clean ``user`` / ``model`` turns."""
    if not isinstance(raw, (list, tuple)):
        return []
    turns = []
    for turn in list(raw)[-MAX_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        text = str(
            turn.get("text") or turn.get("content") or turn.get("message") or ""
        ).strip()
        if not text:
            continue
        if role == "assistant":
            role = "model"
        if role not in ("user", "model"):
            continue
        turns.append({"role": role, "text": text})
    # Gemini's content list must open with a user turn.
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


def _build_contents(history: list, question: str) -> list:
    """History plus the current question, shaped as a Gemini content list.

    Adjacent turns of the same role are folded together, because Gemini expects
    roles to alternate and otherwise rejects the request with a 400.
    """
    from google.genai import types

    merged: list = []
    for turn in list(history) + [{"role": "user", "text": question}]:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {
                "role": turn["role"],
                "text": merged[-1]["text"] + "\n" + turn["text"],
            }
        else:
            merged.append(dict(turn))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return [
        types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        for turn in merged
    ]


@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def _call_gemini(contents: list, system_instruction: str, api_key: str) -> str:
    """One Gemini generation, retried up to 3 times on transient errors."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=GEMINI_TEMPERATURE,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text


def _gemini_answer(
    question: str,
    gravity_filter: Optional[str],
    context_text: str,
    api_key: str,
    settings_block: Optional[str] = None,
    history: Any = None,
) -> str:
    return _call_gemini(
        _build_contents(
            _clean_history(history), _user_turn(question, context_text)
        ),
        _system_instruction(gravity_filter, settings_block),
        api_key,
    )


class QueryRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    gravity_filter: Optional[str] = None
    api_key: Optional[str] = None
    # Prior turns of this conversation as
    # ``[{"role": "user" | "assistant" | "model", "text": ...}, ...]``.
    # Replayed to Gemini on every call so the assistant remembers what was
    # said earlier. Optional: plain curl and older clients simply omit it.
    history: Optional[list] = None
    # Live dashboard controls, so answers can be phrased in terms of the
    # parameters the user is actually looking at. All optional: older clients
    # (and plain curl) simply omit them.
    oxygen_conc: Optional[float] = None
    pressure: Optional[float] = None
    airflow: Optional[float] = None
    fuel: Optional[str] = None


@app.get("/")
def read_root():
    collection = get_collection()
    return {
        "message": "NASA Space Fire Safety RAG API is active.",
        "index_ready": collection is not None,
        "index_error": _collection_error,
        "gemini_configured": _api_key() is not None,
        "gemini_model": GEMINI_MODEL if _api_key() else None,
        "max_passages": MAX_PASSAGES,
        "fetch_k": FETCH_K,
        "rerank": rerank_provider(),
    }


@app.post("/query-agent")
def query_agent(request: QueryRequest):
    question = (request.question or request.query or "").strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="A `question` (or `query`) field is required."
        )

    # Live dashboard controls, surfaced to the model as prompt context.
    settings_block = _settings_block(
        request.oxygen_conc, request.pressure, request.airflow, request.fuel
    )

    # Retrieval runs for every message. Casual or off-topic wording simply finds
    # nothing past the relevance floor, and an empty passage list is what marks
    # the answer as ungrounded - no keyword gate decides what may be asked.
    docs = similarity_search(question, k=MAX_PASSAGES)

    retrieved_sources = []
    context_text = ""
    citations = []
    for doc in docs:
        meta = doc["metadata"]
        source = str(meta.get("source") or "Unknown File")
        page = meta.get("page", "N/A")
        snippet = doc["page_content"]
        retrieved_sources.append(
            {
                "source": source,
                "filename": _file_name(source),
                "page": page,
                "snippet": snippet,
                "title": source_title(meta, source),
                "url": source_url(source),
                "confidence": confidence_of(doc.get("distance")),
            }
        )
        context_text += (
            f"\n[PASSAGE {len(citations) + 1}] {source} (page {page})\n{snippet}\n"
        )
        citations.append({
            "title": source_title(meta, source),
            "url": source_url(source),
            "confidence": confidence_of(doc.get("distance")),
        })

    # One code path for every message: hand the question, whatever passages
    # retrieval found, and the conversation history to Gemini. Chit-chat simply
    # arrives with no passages and is answered as chit-chat.
    api_key = _api_key(request.api_key)
    gemini_used = False
    if api_key is None:
        ai_answer = _preview_answer(question, docs) if docs else _no_key_answer()
    else:
        try:
            ai_answer = _gemini_answer(
                question,
                request.gravity_filter,
                context_text,
                api_key,
                settings_block,
                request.history,
            )
            gemini_used = True
        except Exception as exc:
            # Retries are already exhausted by tenacity (up to 3 attempts with
            # exponential backoff), so hand back something actionable instead
            # of dumping the raw exception.
            ai_answer = _unavailable_answer(question, docs, exc)

    confidence = (
        round(sum(c["confidence"] for c in citations) / len(citations), 4)
        if citations
        else None
    )

    return {
        "status": "success",
        "answer": ai_answer,
        "citations": citations,
        "confidence": confidence,
        "question": question,
        "ai_answer": ai_answer,
        "retrieved_sources": retrieved_sources,
        "sources": citations,
        "gravity_filter": request.gravity_filter,
        "gemini_used": gemini_used,
        "passages_used": len(docs),
        "conversational": not docs,
    }


async def _read_json(request: Optional[Request]) -> dict:
    """Read an optional JSON body, merged with any query-string parameters."""
    payload: dict = {}
    if request is None:
        return payload
    try:
        raw = await request.body()
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload.update(parsed)
    except Exception:
        pass
    payload.update(dict(request.query_params))
    return payload


def build_ranking(payload: dict) -> list:
    """Rank every material for the requested experiment conditions."""
    oxygen = _clamp(_as_float(payload.get("oxygen_conc"), 21.0), 15.0, 35.0)
    airflow = _clamp(_as_float(payload.get("airflow"), 15.0), 0.0, 50.0)
    pressure = _clamp(_as_float(payload.get("pressure"), 101.3), 50.0, 150.0)
    gravity = str(payload.get("gravity") or "0g")

    gravity_factor = 1.1 if gravity == "0g" else 1.0 if gravity == "0.16g" else 0.9
    oxygen_factor = 0.55 + ((oxygen - 15.0) / 20.0) * 0.9
    pressure_factor = _clamp(0.7 + 0.3 * (pressure / 101.3), 0.7, 1.3)
    flow_factor = 0.9 + (airflow / 50.0) * 0.3

    rankings = []
    for material in MATERIALS:
        score = _clamp(
            material["base"] * oxygen_factor * pressure_factor * flow_factor * gravity_factor,
            0.0,
            1.0,
        )
        rankings.append({
            "material": material["label"],
            "risk_level": material["risk_level"],
            "microgravity_flame_spread": material["spread"],
            "extinction_limit": material["extinction"],
            "score": round(score, 4),
            "tier": _tier(score),
            "risk": _tier(score),
        })
    rankings.sort(key=lambda item: item["score"], reverse=True)
    return rankings


def _rank_response(payload: dict) -> dict:
    rankings = build_ranking(payload)
    return {
        "status": "success",
        "rankings": rankings,
        "ranking": rankings,
        "results": rankings,
        "gravity": str(payload.get("gravity") or "0g"),
        "oxygen_conc": payload.get("oxygen_conc"),
        "airflow": payload.get("airflow"),
        "material": payload.get("material"),
        "note": "Server-side ranking; scores are 0-1 flammability estimates.",
    }


@app.post("/rank-materials")
async def rank_materials_post(request: Request):
    return _rank_response(await _read_json(request))


@app.get("/rank-materials")
async def rank_materials_get(request: Request):
    return _rank_response(await _read_json(request))


def pick_port(preferred: int) -> int:
    """Return *preferred* when free, otherwise the first free fallback port."""
    for port in (preferred, 8080, 8081, 8001):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    return preferred


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT") or pick_port(8000))
    print(f"NASA Space Fire Safety API -> http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)



