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


# --- Conversational fallback -------------------------------------------------
# Vocabulary that marks a message as belonging to this domain at all. Greetings
# ("hello", "hi", "hey"), small talk and off-topic requests contain none of it,
# so without this check they miss the retrieval relevance floor and come back as
# "no passages matched" - which reads as a broken tool rather than an assistant.
#
# Each alternative is a *prefix* anchored by a leading \b and deliberately has no
# trailing \b, so stems match their real forms: combust -> combustion,
# ignit -> ignition, flammab -> flammability, extinct -> extinction.
_SCIENCE_RE = re.compile(
    r"\b("
    # combustion behaviour
    r"flame|fire|burn|combust|candle|ignit|extinct|soot|smoke|smolder|pyroly|"
    # atmosphere and gravity
    r"oxygen|o2|co2|microgravity|gravity|weightless|0g|1g|pressure|airflow|vent|"
    # where the work happens
    r"orbit|space|iss|spacecraft|station|cabin|module|nasa|sofie|bass|saffire|"
    r"flex|ntrs|"
    # physics and measurement
    r"spread|pool|droplet|radiat|thermal|heat|temperature|"
    # materials
    r"material|fuel|flammab|polymer|fabric|pmma|acrylic|nomex|kevlar|silicone|"
    r"cotton|teflon|ptfe|polycarbonate|wood|plastic"
    r")",
    re.IGNORECASE,
)


def is_conversational(text: str) -> bool:
    """True for greetings, small talk, or anything outside microgravity fire."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return True
    # Domain vocabulary anywhere wins, so "help me with flame spread" or
    # "hi, is PMMA safe?" still go down the research path.
    if _SCIENCE_RE.search(cleaned):
        return False
    # No domain vocabulary: greeting, thanks, chit-chat or off-topic request.
    return True


_FRIENDLY_ANSWER = """Hey - I'm the Flame in Freefall assistant, an AI guide over NASA's microgravity combustion and fire-safety papers.

Questions I can answer well:
- How does flame spread change at 0g compared with 1g?
- What oxygen and pressure limits let a flame sustain itself in microgravity?
- Which materials burn worst in space - PMMA, Nomex, Kevlar, silicone, cotton?
- How does forced airflow shift the extinction limit?

You can also drive the numbers yourself on the dashboard:
- Gravity mode (0g / 0.16g / 1g) - the control in the top bar.
- Oxygen %, ambient pressure and airflow - the sliders on the left.
- Press "RUN SIMULATION & RANK RISK" to rank all five materials at those settings.

Then ask me about the result and I'll answer it with page-level citations. What would you like to dig into?"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the ONNX embedding session so the very first user query is quick.
    try:
        similarity_search("flame spread in microgravity", k=1)
    except Exception:
        pass
    yield


app = FastAPI(title="NASA Space Fire Safety API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


def _no_match_answer() -> str:
    """Shown when nothing in the index is close enough to ground an answer."""
    return (
        "Nothing in the NASA index matched that question closely enough to answer "
        "from, so an answer here would be guesswork.\n\n"
        "Try wording it around flame spread, extinction limits, oxygen "
        "concentration, gravity level, or a specific material (PMMA, Nomex, "
        "Kevlar, silicone, cotton)."
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
    """Retrieval preview shown when no Gemini key is configured."""
    return "\n".join(
        [
            "Written answers need a Gemini key, so this is only the raw retrieval "
            "preview - not a synthesised answer.",
            "Paste it into `.env` next to main.py as GEMINI_API_KEY=... and restart "
            "the server.",
            "",
            _passage_list(question, docs),
        ]
    )


def _unavailable_answer(question: str, docs: list, exc: BaseException) -> str:
    """Clean fallback handed back once retries are exhausted."""
    reason = _describe_error(exc)
    if not _is_transient(exc):
        # Permanent (bad key, blocked prompt): no point waiting, just say so.
        return (
            f"The Gemini call failed ({reason}).\n\n"
            "Check GEMINI_API_KEY in `.env` next to main.py and restart the server."
        )
    return (
        f"Gemini is unavailable right now: {reason}. I retried up to 3 times "
        "with exponential backoff and it did not recover, so rather than guess "
        "here are the passages the answer would be grounded in.\n\n"
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


def _build_prompt(
    question: str,
    gravity_filter: Optional[str],
    context_text: str,
    settings_block: Optional[str] = None,
) -> str:
    gravity = (gravity_filter or "").strip() or "not specified"
    passages = context_text.strip()

    # The fallback rule leads, so it wins for greetings and off-topic messages
    # and an empty passage list is expected rather than an error condition.
    prompt = (
        "You are the Flame in Freefall assistant: a NASA spacecraft fire-safety "
        "research analyst for microgravity combustion.\n\n"
        "FALLBACK RULE (check this first):\n"
        "If the message is a casual greeting (hello, hi, hey), small talk, a "
        "thank-you, a request for a joke, or anything else outside microgravity "
        "combustion, do NOT report missing passages or retrieval errors. Reply "
        "naturally and helpfully in 3-6 sentences as the NASA Microgravity Fire "
        "Safety AI assistant: say who you are, name two or three things the user "
        "can ask about (flame spread, oxygen limits, material flammability at "
        "0g), and guide them to the dashboard controls - the gravity mode in the "
        "header, the oxygen / pressure / airflow sliders, and the "
        "'RUN SIMULATION & RANK RISK' button.\n\n"
    )

    if settings_block:
        prompt += settings_block + "\n\n"

    if passages:
        prompt += (
            "ANSWERING A RESEARCH QUESTION\n"
            "Write an answer a researcher would find useful:\n"
            "- Open with a direct answer to the question in 1-3 sentences.\n"
            "- Then explain your reasoning, combining passages that agree and "
            "calling out where they conflict or describe different conditions.\n"
            "- Support each claim with an inline citation like "
            "[19930010989.pdf p.202].\n"
            "- Keep the numbers the passages give (spread rates, oxygen limits, "
            "gravity levels) rather than paraphrasing them away.\n"
            "- Gravity level the user selected: " + gravity + ".\n"
            "- If the passages do not contain enough to answer, say precisely "
            "what is missing instead of filling the gap with outside knowledge "
            "or invented figures.\n"
            "- Stay under about 300 words. Use short paragraphs or a labelled "
            "list.\n\n"
        )
    else:
        prompt += (
            "There are no retrieved passages attached to this message, so the "
            "fallback rule above applies.\n\n"
        )

    prompt += "QUESTION\n" + question + "\n"
    if passages:
        prompt += "\nPASSAGES\n" + passages + "\n"
    return prompt


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


@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def _call_gemini(prompt: str, api_key: str) -> str:
    """One Gemini generation, retried up to 3 times on transient errors."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
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
) -> str:
    return _call_gemini(
        _build_prompt(question, gravity_filter, context_text, settings_block),
        api_key,
    )


class QueryRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    gravity_filter: Optional[str] = None
    api_key: Optional[str] = None
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

    # Greetings and off-topic messages are routed before retrieval: they have no
    # domain vocabulary, so they would miss the relevance floor and otherwise
    # surface the "no passages matched" error.
    conversational = is_conversational(question)

    docs = [] if conversational else similarity_search(question, k=MAX_PASSAGES)

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

    # Decide how to answer: chit-chat, nothing matched, no key, key + call,
    # key + failure.
    api_key = _api_key(request.api_key)
    gemini_used = False
    if conversational:
        if api_key is None:
            ai_answer = _FRIENDLY_ANSWER
        else:
            try:
                ai_answer = _gemini_answer(
                    question, request.gravity_filter, "", api_key, settings_block
                )
                gemini_used = True
            except Exception:
                # Small talk should never surface an API error to the user.
                ai_answer = _FRIENDLY_ANSWER
    elif not docs:
        ai_answer = _no_match_answer()
    elif api_key is None:
        ai_answer = _preview_answer(question, docs)
    else:
        try:
            ai_answer = _gemini_answer(
                question, request.gravity_filter, context_text, api_key, settings_block
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
        "conversational": conversational,
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



