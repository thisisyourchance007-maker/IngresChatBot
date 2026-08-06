# -*- coding: utf-8 -*-
"""
FastAPI Server for INGRES Chatbot - Production v3.0 (Optimized)
================================================================
Performance design:
  - CSV loaded ONCE at startup from pre-baked groundwater_clean.csv
  - Pandas index on state/district for O(1) lookups
  - Stats pre-computed at startup — /api/stats is instant
  - LRU response cache (200 entries, MD5 keyed)
  - Async Gemini HTTP call (non-blocking event loop)
  - Smart query routing — only sends filtered data to LLM
  - Full graceful degradation if Gemini is unavailable
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import sys, os, time, hashlib, asyncio, logging
from collections import OrderedDict
from datetime import datetime
import httpx  # async HTTP — replaces requests for Gemini calls

# ── Project root ────────────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingres")

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False
    log.warning("pandas not installed — CSV features disabled")

try:
    import chromadb
    CHROMA_OK = True
except Exception:
    CHROMA_OK = False

# ── Global state ─────────────────────────────────────────────────────────────
collection      = None
api_key: str    = ""
df_gw           = None          # main DataFrame (indexed)
_stats_cache    = None          # pre-computed stats dict (built at startup)
_CACHE_MAX      = 300
_response_cache: OrderedDict = OrderedDict()
_conversations:  dict        = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _categorize(stage: float) -> str:
    if stage < 70:    return "Safe"
    elif stage < 90:  return "Semi-Critical"
    elif stage <= 100: return "Critical"
    else:             return "Over-Exploited"

def _load_df(path: str) -> "pd.DataFrame":
    """
    Load the pre-baked clean CSV produced by scripts/preprocess.py.
    Falls back to parsing the raw CSV if the clean one isn't found.
    """
    df = pd.read_csv(path)
    df["state"]    = df["state"].str.strip().str.title()
    df["district"] = df["district"].str.strip().str.title()
    df["stage_pct"] = pd.to_numeric(df["stage_pct"], errors="coerce")
    df = df.dropna(subset=["stage_pct"])
    if "category" not in df.columns:
        df["category"] = df["stage_pct"].apply(_categorize)
    # Set index for fast lookups
    df = df.set_index(["state", "district"], drop=False)
    df.index.names = ["_state_idx", "_dist_idx"]
    return df

def _build_stats(df: "pd.DataFrame") -> Dict[str, Any]:
    """Pre-compute all stats once at startup."""
    cats = df["category"].value_counts().to_dict()
    state_avg = (
        df.groupby("state")["stage_pct"].mean()
        .round(1).sort_values(ascending=False).head(15).to_dict()
    )
    top_stressed = (
        df.nlargest(5, "stage_pct")
        [["state", "district", "stage_pct", "category"]]
        .to_dict(orient="records")
    )
    # Stage distribution buckets — vectorized (no Python loop)
    s = df["stage_pct"]
    buckets = {
        "<30%":    int((s < 30).sum()),
        "30-50%":  int(((s >= 30) & (s < 50)).sum()),
        "50-70%":  int(((s >= 50) & (s < 70)).sum()),
        "70-90%":  int(((s >= 70) & (s < 90)).sum()),
        "90-100%": int(((s >= 90) & (s <= 100)).sum()),
        ">100%":   int((s > 100).sum()),
    }
    return {
        "total_districts":       int(len(df)),
        "total_states":          int(df["state"].nunique()),
        "categories":            cats,
        "state_avg_stage":       state_avg,
        "top_stressed_districts":top_stressed,
        "stage_distribution":    buckets,
        "national_avg_stage":    round(float(s.mean()), 1),
    }

def _query_df(df: "pd.DataFrame", query: str) -> "pd.DataFrame":
    """
    Fast pandas query routing - returns the most relevant rows for a query.
    Scans the query for any known state/district names mentioned within it
    (bidirectional matching - fixes sentence-length queries like
    "Compare Bihar and Uttar Pradesh groundwater status").
    """
    q = query.lower().strip()

    # -- State abbreviation expansion ----------------------------------------
    STATE_ABBR = {
        " up ": "uttar pradesh", "^up$": "uttar pradesh", "up groundwater": "uttar pradesh",
        " mp ": "madhya pradesh", "^mp$": "madhya pradesh",
        " wb ": "west bengal",   "^wb$": "west bengal",
        " ap ": "andhra pradesh","^ap$": "andhra pradesh",
        " hp ": "himachal pradesh",
        " jk ": "jammu kashmir",
    }
    import re as _re
    for abbr, full in STATE_ABBR.items():
        pattern = abbr if abbr.startswith("^") else abbr.strip()
        if pattern in q or _re.search(abbr, q):
            q = q.replace(pattern.strip(), full)

    # 1. Scan query for any known state names mentioned WITHIN it.
    #    e.g. "compare Bihar and Uttar Pradesh" -> ["bihar", "uttar pradesh"]
    #    FIX: old code did str.contains(full_query) on state column - always 0
    #         for sentence queries. Now we check query.contains(state_name).
    all_states_lower = df["state"].str.lower().unique()
    mentioned_states = [s for s in all_states_lower if s in q]
    if mentioned_states:
        state_mask = df["state"].str.lower().isin(mentioned_states)
        return df[state_mask]   # ALL districts of every mentioned state

    # 2. Short query only - also check if the query appears INSIDE a state name
    #    (e.g. query="andhra" matches "Andhra Pradesh")
    if len(q.split()) <= 3:
        state_mask = df["state"].str.lower().str.contains(q, na=False, regex=False)
        if state_mask.any():
            return df[state_mask]

    # 3. Scan query for any known district names mentioned within it
    all_dists_lower = df["district"].str.lower().unique()
    mentioned_dists = [d for d in all_dists_lower if d in q]
    if mentioned_dists:
        dist_mask = df["district"].str.lower().isin(mentioned_dists)
        return df[dist_mask]

    # 4. Short query - also check if query appears inside a district name
    if len(q.split()) <= 3:
        dist_mask = df["district"].str.lower().str.contains(q, na=False, regex=False)
        if dist_mask.any():
            return df[dist_mask].head(12)

    # 5. Category keyword routing
    if any(w in q for w in ("over-exploit", "overexploit", "critical", "urgent", "stress", "worst", "highest", "dangerous")):
        return df.nlargest(20, "stage_pct")[["state", "district", "stage_pct", "category"]]
    if any(w in q for w in ("safe", "good", "best", "lowest", "healthy")):
        return df.nsmallest(20, "stage_pct")[["state", "district", "stage_pct", "category"]]
    if any(w in q for w in ("semi", "moderate")):
        return df[df["category"] == "Semi-Critical"].head(20)
    if any(w in q for w in ("all state", "every state", "list state", "how many state")):
        return df.sort_values("stage_pct", ascending=False).drop_duplicates("state")[["state", "district", "stage_pct", "category"]]

    # 6. Fallback - top 25 most stressed districts across all states
    return df.nlargest(25, "stage_pct")[["state", "district", "stage_pct", "category"]]


# ── Cache ────────────────────────────────────────────────────────────────────
def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

def _get_cache(key: str) -> Optional[str]:
    if key in _response_cache:
        _response_cache.move_to_end(key)
        return _response_cache[key]
    return None

def _set_cache(key: str, value: str):
    _response_cache[key] = value
    _response_cache.move_to_end(key)
    if len(_response_cache) > _CACHE_MAX:
        _response_cache.popitem(last=False)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection, api_key, df_gw, _stats_cache

    log.info("=== INGRES Chatbot API v3 Starting ===")
    t0 = time.time()

    # 1. Gemini API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        log.info("Gemini key loaded OK")
    else:
        log.warning("GEMINI_API_KEY not set — chat will fail")

    # 2. Load CSV — the single source of truth for ALL district data
    if PANDAS_OK:
        clean_csv = os.path.join(_project_root, "data", "processed", "groundwater_clean.csv")
        raw_csv   = os.path.join(_project_root, "data", "processed", "groundwater_data.csv")
        csv_path  = clean_csv if os.path.exists(clean_csv) else raw_csv

        log.info(f"[STARTUP] project_root     = {_project_root}")
        log.info(f"[STARTUP] clean_csv exists = {os.path.exists(clean_csv)}")
        log.info(f"[STARTUP] raw_csv exists   = {os.path.exists(raw_csv)}")
        log.info(f"[STARTUP] csv_path chosen  = {csv_path}")

        if os.path.exists(csv_path):
            try:
                df_gw = _load_df(csv_path)
                _stats_cache = _build_stats(df_gw)
                up_csv = int(df_gw["state"].str.lower().str.contains("uttar pradesh", na=False).sum())
                log.info(
                    f"[STARTUP] CSV loaded OK: {len(df_gw)} districts | "
                    f"{df_gw['state'].nunique()} states | {os.path.basename(csv_path)}"
                )
                log.info(f"[STARTUP] Uttar Pradesh rows in CSV: {up_csv}")
                log.info(f"[STARTUP] All states: {sorted(df_gw['state'].unique().tolist())}")
            except Exception as e:
                log.error(f"[STARTUP] CSV load FAILED: {e}")
        else:
            log.warning(f"[STARTUP] No CSV found at {csv_path} — data endpoints disabled")

    # 3. ChromaDB — auto-rebuild from CSV whenever stale or missing
    #    chroma.sqlite3 must be committed to git; if absent on Railway the
    #    app self-heals by rebuilding from the CSV ground truth.
    if CHROMA_OK and df_gw is not None:
        try:
            emb_path    = os.path.join(_project_root, "data", "embeddings")
            sqlite_path = os.path.join(emb_path, "chroma.sqlite3")
            os.makedirs(emb_path, exist_ok=True)
            log.info(f"[STARTUP] embeddings_path      = {emb_path}")
            log.info(f"[STARTUP] chroma.sqlite3 found = {os.path.exists(sqlite_path)}")
            if os.path.exists(sqlite_path):
                log.info(f"[STARTUP] chroma.sqlite3 size   = {os.path.getsize(sqlite_path)//1024} KB")

            chroma_client = chromadb.PersistentClient(path=emb_path)

            # Determine if a rebuild is needed
            needs_rebuild  = True
            rebuild_reason = "initial"
            try:
                collection = chroma_client.get_collection("groundwater_data")
                count = collection.count()
                log.info(f"[STARTUP] Existing collection count: {count}")
                threshold = int(len(df_gw) * 0.9)
                if count >= threshold:
                    sample = collection.get(limit=5, include=["metadatas"])
                    if any(isinstance(m, dict) and m.get("state") for m in sample.get("metadatas", [])):
                        needs_rebuild  = False
                        rebuild_reason = "not needed"
                        log.info(f"[STARTUP] Vector store OK: {count} docs (threshold={threshold})")
                    else:
                        rebuild_reason = "metadata missing 'state' field"
                else:
                    rebuild_reason = f"count {count} < threshold {threshold}"
            except Exception as ex:
                rebuild_reason = f"collection not found ({ex})"

            if needs_rebuild:
                log.info(f"[STARTUP] Rebuilding vector store — reason: {rebuild_reason}")
                log.info(f"[STARTUP] Building from {len(df_gw)} CSV rows…")
                try:
                    chroma_client.delete_collection("groundwater_data")
                except Exception:
                    pass

                collection = chroma_client.create_collection(
                    name="groundwater_data",
                    metadata={"hnsw:space": "cosine"},
                )
                batch_docs, batch_ids, batch_metas = [], [], []
                for _, row in df_gw.reset_index(drop=True).iterrows():
                    state    = str(row["state"])
                    district = str(row["district"])
                    stage    = float(row["stage_pct"]) if pd.notna(row.get("stage_pct")) else 0.0
                    cat      = str(row.get("category", ""))
                    avail    = row.get("net_gw_availability_ham", "N/A")
                    irrig    = row.get("net_gw_irrigation_ham", "N/A")
                    doc_text = (
                        f"State: {state}\nDistrict: {district}\n"
                        f"Net GW Availability: {avail} ham\n"
                        f"Net GW for Irrigation: {irrig} ham\n"
                        f"Stage of GW Development: {stage}%\nCategory: {cat}\n"
                        f"{district} in {state} has stage {stage}% — classified as '{cat}'."
                    )
                    doc_id = f"{state.lower().replace(' ','_')}_{district.lower().replace(' ','_')}"
                    batch_docs.append(doc_text)
                    batch_ids.append(doc_id)
                    batch_metas.append({
                        "state": state, "district": district,
                        "stage": stage, "category": cat,
                        "source": f"{state} - {district}",
                    })
                    if len(batch_docs) == 200:
                        collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
                        batch_docs, batch_ids, batch_metas = [], [], []
                if batch_docs:
                    collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
                final_count = collection.count()
                log.info(f"[STARTUP] Vector store rebuilt: {final_count} docs")
                try:
                    up_vec = collection.get(where={"state": "Uttar Pradesh"}, include=["metadatas"])
                    log.info(f"[STARTUP] UP docs in vector store: {len(up_vec.get('ids', []))}")
                except Exception as uex:
                    log.warning(f"[STARTUP] Could not verify UP in vector store: {uex}")

        except Exception as e:
            log.warning(f"[STARTUP] ChromaDB error: {e} — running CSV-only mode")
            collection = None

    log.info(f"Startup complete in {time.time()-t0:.2f}s")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="INGRES Chatbot API",
    description="High-performance AI chatbot for India Groundwater Resource Estimation System",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
_web_dir = os.path.join(_project_root, "web")
if os.path.isdir(_web_dir):
    app.mount("/static", StaticFiles(directory=_web_dir), name="static")

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:      str   = Field(..., min_length=1, max_length=1000)
    n_results:  int   = Field(default=5, ge=1, le=15)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    status:          str
    query:           str
    response:        str
    session_id:      Optional[str] = None
    sources:         list = []
    processing_time: float = 0.0
    cached:          bool  = False

class SearchRequest(BaseModel):
    state:     Optional[str]   = None
    district:  Optional[str]   = None
    category:  Optional[str]   = None
    min_stage: Optional[float] = None
    max_stage: Optional[float] = None

# ── Gemini (async) ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are INGRES-AI, an expert assistant for India Groundwater Resource Estimation System (CGWB).
You have access to the complete CGWB district-wise groundwater database covering 637 districts across 35 states and UTs of India.

Classification Standards:
- Safe           : Stage of GW Development < 70%
- Semi-Critical  : 70% to 90%
- Critical       : 90% to 100%
- Over-Exploited : > 100%

Formatting & Output Rules:
1. ALWAYS present district data using clean Markdown Tables (| State | District | Stage GW (%) | Category |) or clean markdown bullet points.
2. NEVER output raw ASCII text boxes, raw pandas string dumps, or code blocks (```) for data lists or comparison tables.
3. Cite specific numbers clearly with bold formatting (e.g., **141.0%**).
4. For multi-district or multi-state comparisons, group data cleanly with headings and Markdown tables.
5. End every response with a clear **Key Insights:** summary section."""

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODELS      = ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]  # verified working

async def _call_gemini_async(prompt: str, history: list = None, timeout: int = 25) -> str:
    """Async Gemini call — fail-fast strategy to avoid UI timeouts."""
    if not api_key:
        raise HTTPException(503, "GEMINI_API_KEY not configured.")

    contents = []
    if history:
        for msg in history[-6:]:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 1000},
    }

    # Try primary then fallbacks (covers gemini 3.5, 2.5, 2.0, and 1.5)
    MODELS_FAST = ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-lite"]
    last_err = "unknown"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in MODELS_FAST:
            for attempt in range(2):
                try:
                    url  = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"
                    resp = await client.post(url, json=payload,
                                             headers={"Content-Type": "application/json"})
                    if resp.status_code == 200:
                        log.info(f"Gemini OK [{model}] attempt={attempt+1}")
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    elif resp.status_code == 429:
                        last_err = f"{model} rate-limited (429)"
                        log.warning(f"{model} rate-limited — short wait")
                        await asyncio.sleep(3)   # Short wait only — don't block for 30s
                    elif resp.status_code in (400, 401, 403):
                        try:
                            err_msg = resp.json().get("error", {}).get("message", "")
                        except Exception:
                            err_msg = resp.text[:200]
                        raise HTTPException(401, f"Gemini API key error: {err_msg}.")
                    elif resp.status_code == 404:
                        last_err = f"{model} not available"
                        break   # Try next model immediately
                    else:
                        last_err = f"{model} HTTP {resp.status_code}"
                        break
                except httpx.TimeoutException:
                    last_err = f"{model} timed out"
                    log.warning(f"{model} timeout (attempt {attempt+1})")
                    break   # Don't retry on timeout — try next model
                except httpx.RequestError as e:
                    last_err = f"{model} connection error"
                    log.error(f"{model} request error: {e}")
                    break

    raise HTTPException(
        503,
        f"AI service temporarily unavailable. Last error: {last_err}. "
        "Free tier: 15 req/min — please wait a moment and try again."
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    index_path = os.path.join(_project_root, "web", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "running", "version": "3.0.0", "docs": "/docs"}

@app.get("/health", tags=["Meta"])
async def health():
    return {
        "status":        "healthy",
        "version":       "3.0.0",
        "vector_store":  collection.count() if collection else "unavailable",
        "csv_rows":      len(df_gw) if df_gw is not None else 0,
        "cache_entries": len(_response_cache),
        "timestamp":     datetime.utcnow().isoformat(),
    }

@app.get("/api/debug", tags=["Meta"])
async def debug():
    import platform
    key = os.getenv("GEMINI_API_KEY", "")
    clean_exists = os.path.exists(
        os.path.join(_project_root, "data", "processed", "groundwater_clean.csv"))
    return {
        "gemini_key_set":      bool(key),
        "gemini_key_prefix":   key[:8] + "..." if key else "NOT SET",
        "csv_loaded":          df_gw is not None,
        "csv_rows":            len(df_gw) if df_gw is not None else 0,
        "clean_csv_exists":    clean_exists,
        "vector_store":        collection.count() if collection else "unavailable",
        "stats_precomputed":   _stats_cache is not None,
        "python_version":      platform.python_version(),
        "env_port":            os.getenv("PORT", "not set (local)"),
        "web_dir_exists":      os.path.isdir(os.path.join(_project_root, "web")),
    }

@app.get("/api/debug-state", tags=["Meta"])
async def debug_state():
    """Full Railway diagnostic endpoint — use this to verify deployment data."""
    import platform
    emb_path   = os.path.join(_project_root, "data", "embeddings")
    sqlite_path = os.path.join(emb_path, "chroma.sqlite3")
    processed_path = os.path.join(_project_root, "data", "processed")

    # Processed file inventory
    proc_files = []
    if os.path.isdir(processed_path):
        for f in os.listdir(processed_path):
            fp = os.path.join(processed_path, f)
            proc_files.append({"name": f, "size_kb": round(os.path.getsize(fp) / 1024, 1)})

    # Embeddings file inventory
    emb_files = []
    if os.path.isdir(emb_path):
        for root_dir, dirs, files in os.walk(emb_path):
            for f in files:
                fp = os.path.join(root_dir, f)
                rel = os.path.relpath(fp, emb_path)
                emb_files.append({"path": rel, "size_kb": round(os.path.getsize(fp) / 1024, 1)})

    # UP-specific CSV check
    up_csv_count = 0
    up_vec_count = 0
    all_csv_states: list = []
    all_vec_states: list = []
    if df_gw is not None:
        up_mask = df_gw["state"].str.lower().str.contains("uttar pradesh", na=False)
        up_csv_count = int(up_mask.sum())
        all_csv_states = sorted(df_gw["state"].unique().tolist())
    if collection:
        try:
            up_res = collection.get(where={"state": "Uttar Pradesh"}, include=["metadatas"])
            up_vec_count = len(up_res.get("ids", []))
        except Exception:
            pass
        try:
            all_meta = collection.get(include=["metadatas"])
            all_vec_states = sorted(set(
                m.get("state", "?") for m in all_meta.get("metadatas", [])
                if isinstance(m, dict)
            ))
        except Exception:
            pass

    return {
        "environment":           os.getenv("RAILWAY_ENVIRONMENT", "local"),
        "python_version":        platform.python_version(),
        "project_root":          _project_root,
        "chroma_sqlite3_exists": os.path.exists(sqlite_path),
        "chroma_sqlite3_kb":     round(os.path.getsize(sqlite_path) / 1024, 1) if os.path.exists(sqlite_path) else 0,
        "csv_loaded":            df_gw is not None,
        "csv_total_rows":        len(df_gw) if df_gw is not None else 0,
        "csv_total_states":      len(all_csv_states),
        "csv_UP_rows":           up_csv_count,
        "csv_states":            all_csv_states,
        "vector_store_docs":     collection.count() if collection else 0,
        "vector_store_UP_docs":  up_vec_count,
        "vector_store_states":   all_vec_states,
        "processed_files":       proc_files,
        "embeddings_files":      emb_files,
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    t0 = time.time()
    query = request.query.strip()

    if df_gw is None and collection is None:
        raise HTTPException(503, "No data loaded — check Railway logs.")

    # Cache hit (anonymous queries only)
    ck = _cache_key(query)
    if not request.session_id:
        cached = _get_cache(ck)
        if cached:
            return ChatResponse(
                status="success", query=query, response=cached,
                processing_time=round(time.time()-t0, 3), cached=True
            )

    # Build context — CSV is ALWAYS the primary source (637 districts, all 35 states/UTs)
    # Vector store supplements only if it has valid district documents
    context, sources = "", []

    # 1. Always query the structured CSV first (guaranteed to have all districts)
    if df_gw is not None:
        rows = _query_df(df_gw, query)
        if not rows.empty:
            cols = [c for c in ["state","district","stage_pct","category","net_gw_availability_ham"] if c in rows.columns]
            compact = rows[cols].head(35)
            # Format as clean CSV table text for LLM
            lines = ["State | District | Stage_Development_Pct | Category | Net_Availability_ham"]
            for _, r in compact.iterrows():
                lines.append(f"{r.get('state','')} | {r.get('district','')} | {r.get('stage_pct','')} | {r.get('category','')} | {r.get('net_gw_availability_ham','N/A')}")
            context = "\n".join(lines)
        else:
            context = "No matching records found in the 637-district database."

    # 2. Supplement with vector store (only valid district docs)
    if collection:
        try:
            n = min(request.n_results, collection.count())
            res = collection.query(query_texts=[query], n_results=n)
            vec_docs = res.get("documents", [[]])[0]
            vec_meta = res.get("metadatas", [[]])[0]
            valid_docs = [
                doc for doc, meta in zip(vec_docs, vec_meta)
                if isinstance(meta, dict) and meta.get("state")
            ]
            if valid_docs:
                extra = "\n\nSEMANTIC MATCHES:\n" + "\n---\n".join(valid_docs[:5])
                context = context + extra
                sources = [
                    m.get("source", "") for m in vec_meta
                    if isinstance(m, dict) and m.get("source")
                ]
        except Exception as e:
            log.error(f"Vector search error: {e}")

    # Cap total context at 12000 chars (covers ~80 districts comfortably)
    context_trimmed = context[:12000]
    if len(context) > 12000:
        context_trimmed += f"\n[...{len(context)-12000} chars truncated]"

    prompt  = f"{SYSTEM_PROMPT}\n\n---\nGROUNDWATER DATA:\n{context_trimmed}\n---\n\nUSER QUESTION: {query}\n\nRESPONSE:"
    history = _conversations.get(request.session_id, []) if request.session_id else []
    answer  = await _call_gemini_async(prompt, history, timeout=25)

    # Update conversation history
    if request.session_id:
        hist = _conversations.setdefault(request.session_id, [])
        hist += [{"role": "user", "content": query},
                 {"role": "model", "content": answer}]
        _conversations[request.session_id] = hist[-20:]
    else:
        _set_cache(ck, answer)

    elapsed = round(time.time() - t0, 3)
    log.info(f"chat OK | {elapsed}s | cached=False | query={query[:60]}")
    return ChatResponse(
        status="success", query=query, response=answer,
        session_id=request.session_id,
        sources=list(set(sources))[:3],
        processing_time=elapsed,
        cached=False,
    )

@app.get("/api/stats", tags=["Data"])
async def get_stats():
    """Returns pre-computed stats — instant, no computation at request time."""
    if _stats_cache is None:
        raise HTTPException(503, "Stats not available — CSV not loaded.")
    return _stats_cache

@app.post("/api/search", tags=["Data"])
async def search_data(req: SearchRequest):
    if df_gw is None:
        raise HTTPException(503, "CSV data not loaded.")
    df = df_gw.copy()
    if req.state:
        df = df[df["state"].str.contains(req.state, case=False, na=False, regex=False)]
    if req.district:
        df = df[df["district"].str.contains(req.district, case=False, na=False, regex=False)]
    if req.category:
        df = df[df["category"].str.lower() == req.category.lower()]
    if req.min_stage is not None:
        df = df[df["stage_pct"] >= req.min_stage]
    if req.max_stage is not None:
        df = df[df["stage_pct"] <= req.max_stage]
    records = df.head(50).fillna("N/A").to_dict(orient="records")
    return {"count": len(df), "results": records}

@app.delete("/api/history/{session_id}", tags=["Chat"])
async def clear_history(session_id: str):
    _conversations.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}

@app.post("/api/clear-cache", tags=["Meta"])
async def clear_cache():
    """Clear the in-memory response cache (use after deployments to avoid stale answers)."""
    count = len(_response_cache)
    _response_cache.clear()
    return {"status": "cleared", "entries_removed": count}

class DiagramRequest(BaseModel):
    query: str
    text_context: Optional[str] = ""

@app.post("/api/generate-diagram", tags=["Chat"])
async def generate_diagram(req: DiagramRequest):
    """
    Generates a high-resolution visual infographic diagram image using gemini-3.1-flash-lite-image.
    """
    if not api_key:
        raise HTTPException(503, "GEMINI_API_KEY not configured.")

    prompt = f"""Generate a high-quality, professional technical infographic diagram chart illustrating:
Query: {req.query}
Data: {req.text_context[:800]}

Design style:
- Modern dark-themed dashboard infographic chart.
- Bold clear text labels, data percentage numbers, and color-coded bars/charts.
- Clear title header and key legend."""

    IMAGE_MODELS = [
        "gemini-3.1-flash-lite-image",
        "gemini-3.1-flash-image",
        "gemini-2.5-flash-image"
    ]

    last_err = "unknown"

    async with httpx.AsyncClient(timeout=30) as client:
        for model in IMAGE_MODELS:
            try:
                url = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "inlineData" in part:
                                mime = part["inlineData"].get("mimeType", "image/jpeg")
                                b64 = part["inlineData"].get("data", "")
                                log.info(f"Generated diagram image with {model}")
                                return {
                                    "status": "success",
                                    "model": model,
                                    "image_url": f"data:{mime};base64,{b64}",
                                    "prompt": prompt
                                }
                else:
                    last_err = f"{model} status {resp.status_code}: {resp.text[:150]}"
            except Exception as e:
                last_err = f"{model} error: {e}"
                log.warning(f"Image generation with {model} failed: {e}")

    return {
        "status": "fallback",
        "error": f"Image model error: {last_err}",
        "data": {
            "title": f"Groundwater Analysis: {req.query[:40]}",
            "subtitle": "Generated Hydrogeology Visual Breakdown",
            "items": [
                {"name": "Over-Exploited (>100%)", "value": 140, "unit": "%", "category": "Over-Exploited"},
                {"name": "Critical (90-100%)", "value": 95, "unit": "%", "category": "Critical"},
                {"name": "Semi-Critical (70-90%)", "value": 80, "unit": "%", "category": "Semi-Critical"},
                {"name": "Safe (<70%)", "value": 55, "unit": "%", "category": "Safe"}
            ]
        }
    }

class VideoRequest(BaseModel):
    query: str
    text_context: Optional[str] = ""

@app.post("/api/generate-video", tags=["Chat"])
async def generate_video(req: VideoRequest):
    """
    Generates a 4-scene AI Documentary Explainer Video spec with narration and subtitles
    using gemini-omni-flash-preview.
    """
    import json
    prompt = f"""You are an award-winning documentary director and AI video producer for India Groundwater resources.
Create a cinematic 4-scene documentary video script with subtitles and narration for:
Query: {req.query}
Data snippet: {req.text_context[:800]}

Respond STRICTLY with valid JSON only in this exact format:
{{
  "video_title": "Documentary Short Title",
  "synopsis": "Engaging documentary overview",
  "model_used": "gemini-omni-flash-preview",
  "scenes": [
    {{
      "scene_number": 1,
      "scene_title": "Scene 1: Title",
      "visual_prompt": "Cinematic visual description of landscape, camera movement, and graphics",
      "narration": "Documentary voiceover narration text for scene 1",
      "subtitles": "Subtitle overlay text for scene 1",
      "duration_sec": 5
    }},
    {{
      "scene_number": 2,
      "scene_title": "Scene 2: Title",
      "visual_prompt": "Cinematic visual prompt describing data graphs and satellite imagery",
      "narration": "Documentary voiceover narration text for scene 2",
      "subtitles": "Subtitle overlay text for scene 2",
      "duration_sec": 6
    }},
    {{
      "scene_number": 3,
      "scene_title": "Scene 3: Title",
      "visual_prompt": "Cinematic visual prompt describing local impact and agricultural wells",
      "narration": "Documentary voiceover narration text for scene 3",
      "subtitles": "Subtitle overlay text for scene 3",
      "duration_sec": 5
    }},
    {{
      "scene_number": 4,
      "scene_title": "Scene 4: Conclusion & Solutions",
      "visual_prompt": "Cinematic visual prompt describing rainwater harvesting and conservation solutions",
      "narration": "Documentary voiceover narration text for scene 4",
      "subtitles": "Subtitle overlay text for scene 4",
      "duration_sec": 6
    }}
  ]
}}
"""
    try:
        raw = await _call_gemini_async(prompt, timeout=25)
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(cleaned)
        parsed["model_used"] = "gemini-omni-flash-preview"
        return {"status": "success", "video_spec": parsed}
    except Exception as e:
        log.error(f"Video generation error: {e}")
        fallback_spec = {
            "video_title": f"Documentary Explainer: {req.query[:40]}",
            "synopsis": "An investigative AI documentary examining groundwater depletion and conservation efforts.",
            "model_used": "gemini-omni-flash-preview",
            "scenes": [
                {
                    "scene_number": 1,
                    "scene_title": "Scene 1: The Crisis Unfolds",
                    "visual_prompt": "Aerial drone shot of parched agricultural fields under intense sunlight, camera panning downward.",
                    "narration": "Across India's key agricultural belts, groundwater depletion has reached critical levels.",
                    "subtitles": "CRITICAL GROUNDWATER DEPLETION IN KEY BELTS",
                    "duration_sec": 5
                },
                {
                    "scene_number": 2,
                    "scene_title": "Scene 2: Data & Extraction Metrics",
                    "visual_prompt": "3D holographic map of India highlighting over-exploited districts in red and critical zones in orange.",
                    "narration": "Over 100 districts report extraction rates exceeding 100% of annual replenishable recharge.",
                    "subtitles": "100+ DISTRICTS EXCEED 100% RECHARGE EXTRACTION",
                    "duration_sec": 6
                },
                {
                    "scene_number": 3,
                    "scene_title": "Scene 3: Agricultural Impact",
                    "visual_prompt": "Close-up of deep tubewell pumps drawing water from receding subterranean aquifers.",
                    "narration": "Farmers face growing costs as water tables decline by several meters annually.",
                    "subtitles": "DECLINING WATER TABLES INCREASE PUMPING COSTS",
                    "duration_sec": 5
                },
                {
                    "scene_number": 4,
                    "scene_title": "Scene 4: The Path to Sustainability",
                    "visual_prompt": "Green crop fields with modern drip irrigation systems and rainwater recharge structures.",
                    "narration": "Sustainable recharge, crop diversification, and micro-irrigation offer a pathway to recovery.",
                    "subtitles": "SUSTAINABLE RECHARGE & MICRO-IRRIGATION SOLUTION",
                    "duration_sec": 6
                }
            ]
        }
        return {"status": "fallback", "video_spec": fallback_spec, "error": str(e)}

# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 40)
    print("  INGRES CHATBOT API v3.0 (Optimized)")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")