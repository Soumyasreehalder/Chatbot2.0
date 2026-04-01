#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StoreOrderPro RAG — v5.1

Fixes & Enhancements:
  • Syntax fix: replaced '&&' with 'and' and rechecked for other syntax issues.
  • Password Reset SOP isolation:
      - For non-password issues, exclude Password Reset SOP from retrieval and fallbacks.
      - Password branch still reads Acct_details.csv (Account status + Account type) and uses Password Reset SOP only there.
  • Log evidence fallback correctness:
      - Sanitizes LLM log JSON: if no evidence_lines, force matched=false and write "no evidence found" summaries.
      - Checked-table never claims evidence if none is present.

Other behavior preserved:
  • Stage‑1 SOP extraction and cause list.
  • Stage‑2 logs analysis with checked table.
  • Final recommended resolution.
  • Prior incidents.
  • Version/DNS Q&A.
  • Contextual Q&A.

Environment variables:
  • SOP_DATA_DIR (default: ./backend/sop/data)
  • CHROMA_PERSIST_DIR (default: ./backend/sop/chroma_db)
  • KEDB_COLLECTION (default: kedb_pdf_collection)
  • HIST_COLLECTION (default: historical_csv_collection)
  • EMBED_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
  • GROQ_MODEL (default: llama-3.3-70b-versatile)
  • APP_LOG_FILE, SYS_LOG_FILE, DEPLOY_LOG_FILE, DNS_LOG_FILE, NET_LOG_FILE
  • SERVICENOW_URL
  • ACCOUNT_CSV_FILE (default: Acct_details.csv — read from SOP_DATA_DIR)
  • PASSWORD_SOP_FILENAME (default: Password Reset_SOP.PDF)

CSV expectations:
  - ID columns: ["account id","accountid","acct_id","user id","userid","user_id","login","username","uid"] (case-insensitive).
  - Status column: "status"/"state" or boolean-ish "active"/"enabled".
  - Account type: "account type"/"type"/"acct_type"… normalized to: cloud | hybrid | onprem | inactive | unknown.
"""

from __future__ import annotations

import os, uuid, glob, json, logging, re, unicodedata, hashlib
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------------------
# Boot & Config
# -------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_llm_staged_robust_v5_1")

app = FastAPI(title="StoreOrderPro RAG — v5.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Data & Models ---
DATA_DIR        = os.getenv("SOP_DATA_DIR", "./backend/sop/data")
PERSIST_DIR     = os.getenv("CHROMA_PERSIST_DIR", "./backend/sop/chroma_db")
KEDB_COLLECTION = os.getenv("KEDB_COLLECTION", "kedb_pdf_collection")
HIST_COLLECTION = os.getenv("HIST_COLLECTION", "historical_csv_collection")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Log file paths ---
APP_LOG_PATH    = os.path.join(DATA_DIR, os.getenv("APP_LOG_FILE",    "storeorderpro_app.log"))
SYS_LOG_PATH    = os.path.join(DATA_DIR, os.getenv("SYS_LOG_FILE",    "system_resource.log"))
DEPLOY_LOG_PATH = os.path.join(DATA_DIR, os.getenv("DEPLOY_LOG_FILE", "deployment_status.log"))
DNS_LOG_PATH    = os.path.join(DATA_DIR, os.getenv("DNS_LOG_FILE",    "dns.log"))
NET_LOG_PATH    = os.path.join(DATA_DIR, os.getenv("NET_LOG_FILE",    "network.log"))

SERVICENOW_URL  = os.getenv("SERVICENOW_URL", "https://www.servicenow.com/")

# --- Password flow data ---
ACCOUNT_CSV_FILE       = os.getenv("ACCOUNT_CSV_FILE", "Acct_details.csv")
ACCOUNT_CSV_PATH       = os.path.join(DATA_DIR, ACCOUNT_CSV_FILE)
PASSWORD_SOP_FILENAME  = os.getenv("PASSWORD_SOP_FILENAME", "Password Reset_SOP.PDF")

# --- Retrieval thresholds / limits ---
KEDB_SIM_THRESHOLD  = float(os.getenv("KEDB_SIM_THRESHOLD", "0.30"))
TOP_K_SOP           = int(os.getenv("TOP_K_SOP", "16"))
TOP_K_INCIDENTS     = int(os.getenv("TOP_K_INCIDENTS", "12"))
MAX_SOP_CHARS       = int(os.getenv("MAX_SOP_CHARS", "1800"))
MAX_TAIL_LINES      = int(os.getenv("MAX_TAIL_LINES", "300"))
MAX_MATCHED_STEPS   = int(os.getenv("MAX_MATCHED_STEPS", "6"))

# Global budgets
MAX_CONTEXTS                  = int(os.getenv("MAX_CONTEXTS", "10"))
TOTAL_CONTEXT_CHAR_BUDGET     = int(os.getenv("TOTAL_CONTEXT_CHAR_BUDGET", "60000"))
MIN_CAUSES_TRIGGER_FALLBACK   = int(os.getenv("MIN_CAUSES_TRIGGER_FALLBACK", "5"))

# Chunking
SOP_CHUNK_SIZE                = int(os.getenv("SOP_CHUNK_SIZE", "1600"))
SOP_CHUNK_OVERLAP             = int(os.getenv("SOP_CHUNK_OVERLAP", "450"))

# Flags
REQUIRE_RESOLUTION_NOTE       = os.getenv("REQUIRE_RESOLUTION_NOTE", "true").lower() == "true"
AUTO_APPEND_INCIDENTS         = os.getenv("AUTO_APPEND_INCIDENTS", "true").lower() == "true"
DEBUG_LLM                     = os.getenv("DEBUG_LLM", "false").lower() == "true"
MAX_CAUSES_TO_SHOW            = int(os.getenv("MAX_CAUSES_TO_SHOW", "5"))

# -------------------------------------------------------------------
# Dynamic out-of-scope detection — SOP similarity based
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Dynamic out-of-scope detection — SOP similarity based (FIXED)
# -------------------------------------------------------------------
SCOPE_SIM_THRESHOLD = float(os.getenv("SCOPE_SIM_THRESHOLD", "0.35"))

def _is_in_scope_sync(user_problem: str) -> bool:
    """
    Synchronous similarity check against loaded SOP chunks.
    similarity_search_with_score is a blocking call — do NOT wrap in async.
    Returns True (in-scope) if best similarity >= threshold.
    Fail-open on any error.
    """
    if not kedb_store:
        return True
    try:
        hits = kedb_store.similarity_search_with_score(user_problem, k=5)
        if not hits:
            logger.info(f"[ScopeCheck] no hits returned — defaulting to in-scope")
            return True
        best_sim = max(norm_distance_to_sim(dist) for _, dist in hits)
        logger.info(f"[ScopeCheck] best_sim={best_sim:.4f} threshold={SCOPE_SIM_THRESHOLD}")
        return best_sim >= SCOPE_SIM_THRESHOLD
    except Exception as e:
        logger.warning(f"[ScopeCheck] failed, defaulting in-scope: {e}")
        return True
    
# -------------------------------------------------------------------
# Session State
# -------------------------------------------------------------------
@dataclass
class SessionState:
    # Stage‑1 artifacts
    problem: str = ""
    sop_extraction_json: dict | None = None
    awaiting_log_confirm: bool = False

    # Stage‑2 artifacts
    _log_json_cache: dict | None = None
    _selection_cache: dict | None = None
    selected_cause: str = ""
    resolution_line: str = ""
    resolution_bullets: List[str] | None = None
    sop_anchor_excerpt: str = ""
    evidence_note: str = ""
    awaiting_resolution_confirm: bool = False

    # Password branch artifacts
    password_flow_active: bool = False
    awaiting_account_id: bool = False
    provided_account_id: str = ""
    account_status_label: str = ""   # normalized: active|locked|disabled|inactive|unknown
    account_type_label: str = ""     # normalized: cloud|hybrid|onprem|inactive|unknown
    password_sop_reco: Dict[str, Any] | None = None

_sessions: Dict[str, SessionState] = {}
def S(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]

class CauseItemModel(BaseModel):
    title: str = Field(..., description="Specific root cause header")
    short_description: str = Field("", description="1–2 sentence summary from SOP")
    has_steps: bool = False
    steps: List[str] = []
    resolution: str = Field("", description="The Resolution section text from SOP (not recovery steps)")
    resolution_bullets: List[str] = Field([], description="Resolution as bullet points")
    anchor_excerpt: str = ""
    source_hint: str = ""

# -------------------------------------------------------------------
# LLM + Embeddings + Stores
# -------------------------------------------------------------------
llm = ChatGroq(model=GROQ_MODEL, temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})

def norm_distance_to_sim(distance: float) -> float:
    try:
        return max(0.0, min(1.0, 1.0 - (float(distance)/2.0)))
    except Exception:
        return 0.0

def tail(path: str, max_lines: int = MAX_TAIL_LINES) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except FileNotFoundError:
        return f"# {os.path.basename(path)} not found\n"
    except Exception as e:
        return f"# Failed to read {os.path.basename(path)}: {e}\n"

# Load SOP PDFs → chunks → KEDB
sop_docs: List[Document] = []
try:
    sop_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    sop_docs = sop_loader.load()
except Exception as e:
    logger.warning(f"SOP load error: {e}")

def enrich_pdf_metadata(d: Document) -> Document:
    if "source" not in d.metadata:
        d.metadata["source"] = d.metadata.get("file_path", d.metadata.get("source", "unknown"))
    if "page" not in d.metadata and "page_number" in d.metadata:
        d.metadata["page"] = d.metadata["page_number"]
    d.metadata["filename"] = os.path.basename(d.metadata.get("source", ""))
    return d

sop_docs = [enrich_pdf_metadata(d) for d in sop_docs]

sop_chunks: List[Document] = []
if sop_docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=SOP_CHUNK_SIZE, chunk_overlap=SOP_CHUNK_OVERLAP)
    for d in sop_docs:
        for ch in splitter.split_documents([d]):
            sop_chunks.append(ch)

kedb_store: Optional[Chroma] = None
if sop_chunks:
    kedb_store = Chroma.from_documents(
        documents=sop_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=KEDB_COLLECTION,
    )

# Load CSV incidents → HIST
hist_docs: List[Document] = []
csv_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
for p in csv_paths:
    try:
        df = pd.read_csv(p).fillna("")
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="latin-1").fillna("")

    for _, row in df.iterrows():
        inc_id   = str(row.get(os.getenv("CSV_ID","Incident ID"), "")).strip()
        short    = str(row.get(os.getenv("CSV_TITLE","Short Description"), row.get("Issue",""))).strip()
        desc     = str(row.get(os.getenv("CSV_DESC","Description"), "")).strip()
        resol    = str(row.get(os.getenv("CSV_RES","Resolution Note"), "")).strip()
        work     = str(row.get(os.getenv("CSV_WORK","Worknote"), "")).strip()
        created  = str(row.get(os.getenv("CSV_Date","Create_Date"), "")).strip()
        combined = f"Short: {short}\nDesc: {desc}\nResolution: {resol}\nWorknote: {work}"
        hist_docs.append(Document(page_content=combined, metadata={
            "incident_id": inc_id,
            "short_description": short, "description": desc,
            "resolution_note": resol, "worknote": work,
            "created_at": created,
            "source": p
        }))

hist_store: Optional[Chroma] = None
if hist_docs:
    hist_store = Chroma.from_documents(
        documents=hist_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=HIST_COLLECTION,
    )

# ---- Account CSV load (for password flow) ----
_account_df: Optional[pd.DataFrame] = None
def _load_account_df() -> Optional[pd.DataFrame]:
    global _account_df
    if _account_df is not None:
        return _account_df
    try:
        if os.path.exists(ACCOUNT_CSV_PATH):
            try:
                _account_df = pd.read_csv(ACCOUNT_CSV_PATH).fillna("")
            except UnicodeDecodeError:
                _account_df = pd.read_csv(ACCOUNT_CSV_PATH, encoding="latin-1").fillna("")
            logger.info(f"Loaded account CSV: {ACCOUNT_CSV_PATH} with {len(_account_df)} rows")
        else:
            logger.warning(f"Account CSV not found at: {ACCOUNT_CSV_PATH}")
    except Exception as e:
        logger.error(f"Failed to load account CSV: {e}")
        _account_df = None
    return _account_df

# -------------------------------------------------------------------
# JSON / Structured Output Utilities
# -------------------------------------------------------------------
def _strip_code_fences(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    return s

def _extract_outer_json(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = _strip_code_fences(raw)
    try:
        start = s.index("{")
    except ValueError:
        return None
    stack = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                return s[start:i+1]
    return None

def parse_json_or_none(raw: str) -> Optional[Any]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    obj = _extract_outer_json(raw)
    if obj:
        try:
            return json.loads(obj)
        except Exception:
            repaired = re.sub(r",(\s*[}\]])", r"\1", obj)
            try:
                return json.loads(repaired)
            except Exception:
                return None
    return None

_JSON_START = "<<<JSON_START>>>"
_JSON_END   = "<<<JSON_END>>>"

def extract_between_markers(s: str, start=_JSON_START, end=_JSON_END) -> Optional[str]:
    if not s:
        return None
    i = s.find(start)
    if i == -1:
        return None
    j = s.find(end, i + len(start))
    if j == -1:
        return None
    return s[i+len(start):j].strip()

async def ainvoke_json(llm: ChatGroq, system_prompt: str, user_prompt: str, *, max_retries: int = 4) -> Tuple[Optional[Any], str]:
    raw = ""
    prompts = [
        (system_prompt,
         user_prompt + f"\n\nRespond with STRICT JSON ONLY between markers:\n{_JSON_START}\n{{...}}\n{_JSON_END}"),
        (system_prompt + "\n\nIMPORTANT: Return VALID JSON ONLY. No prose, no markdown, no code fences.",
         user_prompt   + f"\nReturn strictly valid JSON between markers.\n{_JSON_START}\n{{...}}\n{_JSON_END}"),
        (system_prompt + "\n\nCRITICAL: Output MUST be JSON. Anything else will break the system.",
         user_prompt   + f"\nReply ONLY with JSON inside these markers.\n{_JSON_START}\n{{...}}\n{_JSON_END}")
    ]
    for attempt in range(1, max_retries + 1):
        idx = min(attempt-1, len(prompts)-1)
        sys = prompts[idx][0]
        usr = prompts[idx][1]
        try:
            resp = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
            raw = (resp.content or "").strip()
        except Exception as e:
            logger.warning(f"[ainvoke_json] LLM call failed on attempt {attempt}: {e}")
            continue
        candidate = extract_between_markers(raw) or raw
        parsed = parse_json_or_none(candidate)
        if parsed is not None:
            return parsed, raw
        logger.warning(f"[ainvoke_json] JSON parse failed on attempt {attempt}. Raw head: {raw[:220]!r}")
        if DEBUG_LLM:
            logger.error(f"[DEBUG_LLM] RAW OUTPUT attempt {attempt}:\n{raw}")
    return None, raw

# -------------------------------------------------------------------
# Pydantic schema (optional structured output)
# -------------------------------------------------------------------
class SopExtractionModel(BaseModel):
    problem: str = ""
    causes: List[CauseItemModel] = []

structured_extractor = None
try:
    structured_extractor = llm.with_structured_output(SopExtractionModel)
except Exception:
    structured_extractor = None
    logger.info("Structured output not available; using JSON mode.")

async def sop_structured_extract(problem: str, context: str) -> Optional[SopExtractionModel]:
    if structured_extractor is None:
        return None
    try:
        resp: SopExtractionModel = await structured_extractor.ainvoke([
            SystemMessage(content="You are a meticulous SOP analyst. Output MUST match the provided schema."),
            HumanMessage(content=f"Problem (do not filter by it):\n{problem}\n\nSOP Context:\n{context}")
        ])
        return resp
    except Exception as e:
        logger.warning(f"[sop_structured_extract] failed: {e}")
        return None

# -------------------------------------------------------------------
# Prompts — SOP, logs, selection, contextual QA
# -------------------------------------------------------------------
SOP_EXTRACT_SYS = """You are a meticulous SOP analyst.

OUTPUT CONTRACT:
- You MUST return VALID JSON matching the exact schema below.
- Do NOT include any text before or after the JSON.
- If unsure, return an empty array for 'causes'.

JSON SCHEMA:
{
  "problem":"string",
  "causes":[
    {
      "title":"string",
      "short_description":"string",
      "has_steps":true|false,
      "steps":["string"],
      "resolution": "string",
      "resolution_bullets": ["string"],
      "anchor_excerpt":"string",
      "source_hint":"string"
    }
  ]
}

Extraction rules (strict):
- Do NOT invent content.
- Titles must be the SPECIFIC root-cause header (strip numbering).
- If steps are not explicitly present nearby, set has_steps=false and steps=[].
- `resolution`: Extract ONLY from the "Resolution" section (high-level fix; not recovery steps).
- `resolution_bullets`: Split the resolution into sentence bullets. If empty, [].
"""
SOP_EXTRACT_USER = """Problem (for reference only):
{problem}

SOP Context (truncated):
<<<SOP>>>
{context}
<<<END>>>
"""

LOG_PLAN_SYS = """You are a reliability SRE assistant.
Given a selected cause and SOP steps, read the provided log tails for evidence.
Domains/files: ["DEPLOYMENT","NETWORK","MEMORY","DNS","APPLICATION"].
Return JSON:
{
  "domains": [
    {"domain":"NETWORK","matched":true|false,"evidence_lines":["...","..."],"summary":"..."},
    {"domain":"DNS","matched":true|false,"evidence_lines":["...","..."],"summary":"..."},
    {"domain":"DEPLOYMENT","matched":...,"evidence_lines":[...],"summary":"..."},
    {"domain":"MEMORY",...},
    {"domain":"APPLICATION",...}
  ],
  "recommended_domains": ["DNS","NETWORK","APPLICATION"]
}
JSON only.
"""
LOG_PLAN_USER = """Problem:
{problem}

Selected cause:
{cause}

SOP quoted steps:
{steps}

Log tails:
[DEPLOYMENT]
{deploy}

[NETWORK]
{network}

[DNS]
{dns}

[MEMORY]
{memory}

[APPLICATION]
{app}
"""

SELECT_CAUSE_SYS = """You pick ONE best root cause using SOP extraction + log evidence.
Rules:
- Prefer causes with has_steps=true and quoted steps.
- If no steps, can_recommend=false.
- For `resolution_line`: Use the cause's `resolution` field verbatim (SOP Resolution section).
- For `resolution_bullets`: Use the cause's `resolution_bullets` verbatim; if empty but resolution exists, split into sentences.
- Include a short evidence_note from matched domains.
Return JSON ONLY:
{
  "selected_cause":"...",
  "resolution_line":"...",
  "sop_anchor_excerpt":"...",
  "resolution_bullets": ["...", "..."],
  "evidence_note":"...",
  "can_recommend":true|false,
  "refusal_reason":"..."
}
"""
SELECT_CAUSE_USER = """Problem:
{problem}

SOP extraction JSON:
{extraction_json}

Log evidence JSON:
{log_json}
"""

# Contextual QA — answers general questions
CONTEXT_QA_SYS = """You are a helpful SRE assistant.
Answer clearly and conversationally using ONLY the provided session context.
No lists of root causes unless explicitly requested.
If uncertain, say so briefly and suggest a next step."""
CONTEXT_QA_USER = """Session context:
- Problem: {problem}
- Selected cause: {selected_cause}
- Known causes: {cause_titles}
- Logs summary:
{logs_summary}

User question:
{question}
"""

YESNO_SYS = """Classify as YES/NO.
Return JSON only: {"answer":"YES"} or {"answer":"NO"}."""
YESNO_USER = """User reply:
{reply}"""

LOG_QA_SYS = """You read a deployment log tail to answer:
- "What is the current version?" / "latest version" / "active release"
Return JSON:
{
  "direct_answer":"...",    // e.g., "v3.8.4"
  "evidence_lines":["...","..."],
  "confidence":"high|medium|low"
}
If unclear, set direct_answer="Not found" and confidence="low".
JSON only.
"""
LOG_QA_USER = """Question:
{question}

Deployment log tail:
{deploy}
"""

# -------------------------------------------------------------------
# Title normalization / de-dup helpers
# -------------------------------------------------------------------
GENERIC_TITLES = {"root cause","root-cause","issue","problem","unknown","n/a","na","general"}
def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "")).strip().lower()
    s = re.sub(r"\s+"," ", s)
    s = re.sub(r"^[#:*\-\d\.\s]+","", s)
    s = re.sub(r"\b(root\s*cause\s*\d*:?)\b","", s)
    return s.strip(":-–—•| ")

def _looks_generic_title(s: str) -> bool:
    t = _norm_text(s)
    return (not t) or (t in GENERIC_TITLES) or (len(t) < 3)

def dedup_and_limit_causes(raw_causes: list, max_causes: int = 5) -> list:
    seen = set(); cleaned = []
    for c in raw_causes:
        title = (c.get("title") or c.get("cause") or "").strip()
        desc  = (c.get("short_description") or "").strip()
        if _looks_generic_title(title): continue
        key = _norm_text(title)
        if key in seen: continue
        seen.add(key)
        cleaned.append({
            "title": title, "short_description": desc,
            "has_steps": bool(c.get("has_steps")), "steps": c.get("steps") or [],
            "anchor_excerpt": c.get("anchor_excerpt") or "", "source_hint": c.get("source_hint") or ""
        })
        if len(cleaned) >= max_causes: break
    return cleaned

# -------------------------------------------------------------------
# Evidence sanitization helpers
# -------------------------------------------------------------------
_TS_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?\b"),
    re.compile(r"\b\d{4}[/-]\d{2}[/-]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    re.compile(r"\[\d{2}:\d{2}:\d{2}(?:\.\d+)?\]"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"),
    re.compile(r"^\d{10,13}\b"),
]
def strip_timestamps(s: str) -> str:
    if not s: return s
    out = s
    for p in _TS_PATTERNS: out = p.sub("", out)
    out = re.sub(r"^\s*[-–—\[\]\(\)]\s*","", out).strip()
    return re.sub(r"\s{2,}"," ", out)

def compress_phrase(text: str) -> str:
    s = strip_timestamps(text or "")
    s = s.replace("DNS request timed out", "DNS query timed out")
    s = re.sub(r"\btimeout(?: was)?\s*=?\s*\d+\s*(?:ms|milliseconds|s|seconds)?","timeout", s, flags=re.I)
    s = s.replace("Action=BLOCK","blocked").replace("DestPort=53","UDP/53").replace("Proto=UDP","UDP")
    return s[:160].rstrip(" ;,") if len(s) > 160 else s

def file_nice(file: str) -> str:
    return (file or "").strip() or "logs"

def build_fluent_sentence(domain: str, evidence_lines: List[str], file: str) -> str:
    dom = (domain or "").upper()
    # if no evidence, return explicit "no evidence" sentence
    if not evidence_lines:
        if dom == "DNS":
            return f"Checked the DNS log (**{file}**) — no evidence of DNS errors."
        if dom == "NETWORK":
            return f"Checked the network log (**{file}**) — no evidence of connectivity issues."
        if dom == "DEPLOYMENT":
            return f"Checked the deployment log (**{file}**) — no evidence of deployment issues."
        if dom == "MEMORY":
            return f"Checked the system log (**{file}**) — no evidence of resource problems."
        if dom == "APPLICATION":
            return f"Checked the application log (**{file}**) — no evidence of application errors."
        return f"Checked **{file}** — no evidence found."

    # Otherwise use first clean evidence line
    detail = ""
    for line in (evidence_lines or []):
        ph = compress_phrase(line)
        if ph:
            detail = ph
            break

    if dom == "DNS":
        if "timeout" in detail.lower():
            return f"Checked the DNS log — DNS lookups are timing out ({detail})."
        if "nxdomain" in detail.lower() or "not found" in detail.lower():
            return f"Checked the DNS log — hostnames are not resolving ({detail})."
        if "servfail" in detail.lower():
            return f"Checked the DNS log — DNS server returned a failure ({detail})."
        return f"Checked the DNS log — found: {detail}."

    if dom == "NETWORK":
        if "unreachable" in detail.lower() or "timeout" in detail.lower():
            return f"Checked the network log — target appears unreachable ({detail})."
        if "block" in detail.lower():
            return f"Checked the network log — traffic is being blocked ({detail})."
        return f"Checked the network log — found: {detail}."

    if dom == "DEPLOYMENT":
        if any(k in detail.lower() for k in ["complete", "success", "active", "stable"]):
            return f"Checked the deployment log — latest release is healthy ({detail})."
        if any(k in detail.lower() for k in ["fail", "rollback", "error"]):
            return f"Checked the deployment log — deployment failure or rollback detected ({detail})."
        return f"Checked the deployment log — found: {detail}."

    if dom == "MEMORY":
        if any(k in detail.lower() for k in ["high", "pressure", "critical", "84%", "90%", "95%"]):
            return f"Checked the system log — memory usage is high ({detail})."
        return f"Checked the system log — found: {detail}."

    if dom == "APPLICATION":
        if any(k in detail.lower() for k in ["error", "exception", "crash", "fail"]):
            return f"Checked the application log — application errors observed ({detail})."
        if any(k in detail.lower() for k in ["timeout", "slow", "latency"]):
            return f"Checked the application log — timeouts or slowness detected ({detail})."
        return f"Checked the application log — found: {detail}."

    return f"Checked the logs — found: {detail}."

# -------------------------------------------------------------------
# Retrieval helpers & SOP cheap scan with Password-SOP exclusion
# -------------------------------------------------------------------
def _all_sop_chunks(exclude_password: bool) -> List[Document]:
    if not exclude_password:
        return sop_chunks
    return [d for d in sop_chunks if d.metadata.get("filename","").lower() != PASSWORD_SOP_FILENAME.lower()
            and "password" not in d.metadata.get("filename","").lower()]

def _build_context_blobs_from_docs(docs: List[Document]) -> List[str]:
    contexts: List[str] = []
    buf = ""
    budget = TOTAL_CONTEXT_CHAR_BUDGET

    def _flush():
        nonlocal buf, budget
        if buf:
            part = buf[:MAX_SOP_CHARS]
            contexts.append(part)
            budget -= len(part)
            buf = ""

    for d in docs:
        chunk = (d.page_content or "")
        if not chunk:
            continue
        if len(chunk) > budget or len(contexts) >= MAX_CONTEXTS or budget <= 0:
            break
        if len(buf) + len(chunk) <= MAX_SOP_CHARS:
            buf += ("\n\n---\n\n" + chunk) if buf else chunk
        else:
            _flush()
            if len(contexts) >= MAX_CONTEXTS or budget <= 0:
                break
            buf = chunk

    if buf and len(contexts) < MAX_CONTEXTS and len(buf) <= budget:
        _flush()
    return contexts

def _retrieve_sop_contexts(problem_text: str, *, exclude_password: bool = True) -> List[str]:
    if not kedb_store:
        return []
    sop_docs_local: List[Document] = []
    try:
        hits = kedb_store.max_marginal_relevance_search(
            problem_text, k=TOP_K_SOP, fetch_k=max(TOP_K_SOP*4, TOP_K_SOP+8), lambda_mult=0.5
        )
        sop_docs_local = hits or []
    except Exception as e:
        logger.warning(f"MMR failed; fallback to similarity. Err: {e}")
        try:
            sim_hits = kedb_store.similarity_search_with_score(problem_text, k=TOP_K_SOP)
            if sim_hits:
                filtered = []
                for (d, dist) in sim_hits:
                    if norm_distance_to_sim(dist) >= KEDB_SIM_THRESHOLD:
                        filtered.append(d)
                sop_docs_local = filtered or [d for (d, _) in sim_hits]
        except Exception as e2:
            logger.error(f"Similarity retrieval failed: {e2}")
            sop_docs_local = []

    if exclude_password:
        sop_docs_local = [
            d for d in sop_docs_local
            if d.metadata.get("filename","").lower() != PASSWORD_SOP_FILENAME.lower()
            and "password" not in d.metadata.get("filename","").lower()
        ]

    if not sop_docs_local:
        return []
    return _build_context_blobs_from_docs(sop_docs_local)

def _full_sop_fallback_contexts(exclude_password: bool = True) -> List[str]:
    return _build_context_blobs_from_docs(_all_sop_chunks(exclude_password))

def cheap_root_cause_scan(text: str) -> dict:
    causes = []
    if not text:
        return {"problem":"", "causes":[]}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title_pat = re.compile(r"root\s*cause\s*\d*\s*[:\-]\s*(.+)$", re.I)
    seen = set()
    for i, line in enumerate(lines):
        m = title_pat.search(line)
        if not m:
            continue
        title = m.group(1).strip(":-• ").strip()
        norm = _norm_text(title)
        if not title or len(norm) < 3 or _looks_generic_title(title) or norm in seen:
            continue
        seen.add(norm)
        desc = ""
        for j in range(i+1, min(i+6, len(lines))):
            nxt = lines[j]
            if re.search(r"^root\s*cause", nxt, re.I):
                break
            if len(nxt) > 30:
                desc = nxt
                break
        causes.append({
            "title": title[:120],
            "short_description": (desc[:300] if desc else ""),
            "has_steps": False, "steps": [],
            "anchor_excerpt": "", "source_hint": ""
        })
        if len(causes) >= MAX_CAUSES_TO_SHOW:
            break
    return {"problem":"", "causes":causes}

# -------------------------------------------------------------------
# Prior incidents — hybrid ranking, dedup, variable length (1–3)
# -------------------------------------------------------------------
_DNS_TOKENS = {
    "dns","nslookup","dig","servfail","nxdomain","resolver","resolve","resolution",
    "coredns","bind","named","forwarder","delegation","zone","record","a record","cname","aaaa",
    "hostname","host","fqdn","timeout","rdns","stub","conditional forward","conditional forwarder","recursor"
}
_HOST_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.I)

def _extract_hostnames_from_logs(log_json: dict) -> List[str]:
    hosts = set()
    if not log_json:
        return []
    for item in (log_json.get("domains") or []):
        for line in (item.get("evidence_lines") or []):
            s = strip_timestamps(line or "")
            for m in _HOST_RE.finditer(s):
                hosts.add(m.group(0).lower())
    return list(hosts)

def _is_dns_cause(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["dns","resolve","resolver","hostname","host name"])

def _keyword_boost(text: str, tokens: set[str], hosts: List[str]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    token_hits = sum(1 for tok in tokens if tok in t)
    score += min(0.15, 0.03 * token_hits)
    host_hits = sum(1 for h in hosts if h and h in t)
    score += min(0.15, 0.05 * host_hits)
    return min(0.30, score)

def _content_hash(d: Document) -> str:
    base = (d.metadata.get("incident_id") or "") + "||" + (d.page_content or "")
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()

# -------------------------------------------------------------------
# Render SOP causes bullets
# -------------------------------------------------------------------
def render_sop_causes_bullets(sop_json: dict) -> str:
    causes = (sop_json or {}).get("causes") or []
    if not causes:
        return "I couldn’t extract clear root causes from the SOP yet."
    bullet = "\n".join(
        (f"- **{c.get('title') or c.get('cause') or ''}**: {c.get('short_description') or ''}").rstrip(": ")
        for c in causes
    )
    return (
        "🔍 I understand your concern. When I look for knowledge article, I see below are possible causes for which the issue may occur \n\n"
        f"{bullet}\n\n"
        "Would you like me to **further analysis** to identify the problem cause for this issue? (yes/no)"
    )

def format_resolution_bullets(bullets: List[str], fallback_line: str = "") -> str:
    clean = [b.strip(" •-") for b in (bullets or []) if b.strip(" •-")]
    if not clean and fallback_line:
        clean = [s.strip() for s in re.split(r'(?<=[.!?])\s+', fallback_line.strip()) if s.strip()]
    if not clean:
        return fallback_line or "Please refer to the SOP for resolution steps."
    return "\n".join(f"- {b}" for b in clean)

# -------------------------------------------------------------------
# Stage 2B — Final resolution (standard flow)
# -------------------------------------------------------------------
class ChatResponse(BaseModel):
    session_id: str
    answer: str

async def finish_with_resolution_and_incident(session_id: str) -> ChatResponse:
    ss = S(session_id)
    selection = ss._selection_cache or {}
    if not selection.get("can_recommend"):
        reason = selection.get("refusal_reason", "SOP does not provide explicit steps near the cause.")
        return ChatResponse(session_id=session_id,
            answer=(f"I get the urgency, and I wish I had clearer steps from the SOP this time. "
                    f"**Unable to recommend a fix**: {reason}\n\n"
                    f"If you want, we can follow the rollback path or open a ticket: {SERVICENOW_URL}"))
    bullets = selection.get("resolution_bullets") or ss.resolution_bullets or []
    fallback = selection.get("resolution_line") or ss.resolution_line or ""
    resolution_md = format_resolution_bullets(bullets, fallback)

    final = (
        f"##### ✅ To effectively address the observed issues, I propose the following recommended solutions: \n"
        f"**Identified Cause:** {ss.selected_cause}\n\n"
        f"**Resolution:**\n{resolution_md}\n"
    )
    final += "\n\n---\n😊 **Are you happy with the resolution?**"
    return ChatResponse(session_id=session_id, answer=final)

# -------------------------------------------------------------------
# LOG JSON Sanitizer (ensures correctness when evidence is missing)
# -------------------------------------------------------------------
_ALLOWED_DOMAINS = ["DEPLOYMENT","NETWORK","DNS","MEMORY","APPLICATION"]

def _sanitize_log_json(log_json: dict) -> dict:
    if not isinstance(log_json, dict):
        return {"domains": [], "recommended_domains": []}
    domains = []
    for wanted in _ALLOWED_DOMAINS:
        # find matching entry (case-insensitive)
        entry = None
        for itm in (log_json.get("domains") or []):
            d = (itm.get("domain") or "").upper()
            if d == wanted:
                entry = itm
                break
        if entry is None:
            entry = {"domain": wanted, "matched": False, "evidence_lines": [], "summary": ""}

        ev = [strip_timestamps(x) for x in (entry.get("evidence_lines") or []) if strip_timestamps(x)]
        has_ev = len(ev) > 0
        # Force correctness
        entry["domain"] = wanted
        entry["evidence_lines"] = ev
        entry["matched"] = bool(entry.get("matched") and has_ev)
        # Normalize summary
        if not has_ev:
            entry["summary"] = f"No evidence found in {wanted} logs."
        domains.append(entry)

    return {"domains": domains, "recommended_domains": list(set((log_json.get("recommended_domains") or [])))}

# -------------------------------------------------------------------
# Stage 2A — Logs → select cause → checked table (fluent sentences)
# -------------------------------------------------------------------
async def run_stage2_show_checked_table(session_id: str) -> ChatResponse:
    ss = S(session_id)
    problem = ss.problem or "(no problem provided)"
    sop_json = ss.sop_extraction_json or {"causes":[]}
    causes = sop_json.get("causes",[])
    if not causes:
        return ChatResponse(session_id=session_id, answer="I couldn't find causes from SOP. Please re-state the problem.")

    candidate = next((c for c in causes if c.get("has_steps")), causes[0])
    cause_text = (candidate.get("title") or candidate.get("cause") or "")
    steps_text = "\n".join((candidate.get("steps") or [])[:MAX_MATCHED_STEPS])

    deploy_tail = tail(DEPLOY_LOG_PATH)
    net_tail    = tail(NET_LOG_PATH)
    dns_tail    = tail(DNS_LOG_PATH)
    mem_tail    = tail(SYS_LOG_PATH)
    app_tail    = tail(APP_LOG_PATH)

    log_json, raw_log = await ainvoke_json(
        llm, LOG_PLAN_SYS,
        LOG_PLAN_USER.format(problem=problem, cause=cause_text, steps=steps_text or "(no steps)",
                             deploy=deploy_tail, network=net_tail, memory=mem_tail, dns=dns_tail, app=app_tail))
    if log_json is None:
        if DEBUG_LLM: logger.error(f"[DEBUG_LLM] LOG PLAN RAW:\n{raw_log}")
        log_json = {"domains":[], "recommended_domains":[]}

    # --- Sanitize LLM output so empty evidence never appears as matched ---
    log_json = _sanitize_log_json(log_json)

    # Reselect with sanitized logs
    selection, raw_sel = await ainvoke_json(
        llm, SELECT_CAUSE_SYS,
        SELECT_CAUSE_USER.format(problem=problem,
            extraction_json=json.dumps(sop_json, ensure_ascii=False),
            log_json=json.dumps(log_json, ensure_ascii=False)))
    if selection is None:
        if DEBUG_LLM: logger.error(f"[DEBUG_LLM] SELECT RAW:\n{raw_sel}")
        selection = {"can_recommend":False, "refusal_reason":"Parse error"}

    ss._log_json_cache   = log_json
    ss._selection_cache  = selection
    ss.selected_cause    = selection.get("selected_cause","") or cause_text
    ss.resolution_line   = selection.get("resolution_line","")
    ss.sop_anchor_excerpt= selection.get("sop_anchor_excerpt","")
    ss.evidence_note     = selection.get("evidence_note","")
    ss.resolution_bullets= selection.get("resolution_bullets", [])

    def file_for_domain(d: str) -> str:
        d = (d or "").upper()
        if d == "DEPLOYMENT": return os.path.basename(DEPLOY_LOG_PATH)
        if d == "NETWORK":    return os.path.basename(NET_LOG_PATH)
        if d == "DNS":        return os.path.basename(DNS_LOG_PATH)
        if d == "MEMORY":     return os.path.basename(SYS_LOG_PATH)
        return os.path.basename(APP_LOG_PATH)

    logs_info = []
    for item in (log_json.get("domains") or []):
        dom = (item.get("domain") or "").upper()
        file = file_for_domain(dom)
        evidence = (item.get("evidence_lines") or [])
        has_evidence = len(evidence) > 0
        summary_sentence = build_fluent_sentence(dom, evidence, file)
        logs_info.append({
            "domain": dom,
            "matched": bool(item.get("matched", False)),
            "summary_sentence": summary_sentence,
            "file": file,
            "has_evidence": has_evidence
        })

    cause_titles = [(c.get("title") or c.get("cause") or "") for c in causes if (c.get("title") or c.get("cause"))]

    def ruled_out_sentence(domain: str, file: str, cause: str) -> str:
        where = file_nice(file)
        d = (domain or "").upper()
        if d == "DNS":
            return f"Checked **{where}** and didn’t find DNS anomalies, so **{cause}** is likely ruled out."
        if d == "NETWORK":
            return f"Checked **{where}** and didn’t find network anomalies, so **{cause}** is likely ruled out."
        if d == "APPLICATION":
            return f"Checked **{where}** and didn’t see notable application errors, so **{cause}** is likely ruled out."
        if d == "DEPLOYMENT":
            return f"Checked **{where}** and didn’t find deployment issues, so **{cause}** is likely ruled out."
        if d == "MEMORY":
            return f"Checked **{where}** and memory/uptime look normal, so **{cause}** is likely ruled out."
        return f"Checked **{where}** and didn’t find anomalies, so **{cause}** is likely ruled out."

    def match_domain_for_cause(cause_title: str) -> str:
        ct = (cause_title or "").lower()
        if "dns" in ct:
            return "DNS"
        if "network" in ct or "reachability" in ct or "connectivity" in ct:
            return "NETWORK"
        if "deploy" in ct or "version" in ct or "release" in ct:
            return "DEPLOYMENT"
        if "uptime" in ct or "memory" in ct or "resource" in ct:
            return "MEMORY"
        return "APPLICATION"

    logs_by_domain = { item["domain"]: item for item in logs_info }

    rows = []
    for i, ct in enumerate(cause_titles, start=1):
        dom = match_domain_for_cause(ct)
        li = logs_by_domain.get(dom)

        if li:
            summary = li["summary_sentence"]
            file = li["file"]
            matched = bool(li.get("matched", False))
            has_evidence = bool(li.get("has_evidence", False))
            if (not matched) or (not has_evidence):
                summary = ruled_out_sentence(dom, file, ct)
        else:
            guessed_file = (
                os.path.basename(NET_LOG_PATH) if dom == "NETWORK" else
                os.path.basename(DNS_LOG_PATH) if dom == "DNS" else
                os.path.basename(DEPLOY_LOG_PATH) if dom == "DEPLOYMENT" else
                os.path.basename(SYS_LOG_PATH) if dom == "MEMORY" else
                os.path.basename(APP_LOG_PATH)
            )
            summary = ruled_out_sentence(dom, guessed_file, ct)

        status = "✓" if ct == ss.selected_cause else "✗"
        rows.append(f"| {i} | {ct} | {summary} | {status} |")

    table_md = (
        f"##### 🔍 As a result of my analysis with the system, I have observed the following:\n"
        f"**Possible cause:** {ss.selected_cause}\n\n"
        f"| # | Checked Root Cause | Result (summary) | ✓/✗ |\n"
        f"|:-:|---|---|:-:|\n" + "\n".join(rows)
    )

    final_section = f"{table_md}\n\nWould you like the recommended resolution for the selected root cause? (yes/no)"
    ss.awaiting_resolution_confirm = True
    return ChatResponse(session_id=session_id, answer=final_section)

# -------------------------------------------------------------------
# Stage 1 — SOP-first (exclude Password SOP unless it’s a password issue)
# -------------------------------------------------------------------
PASSWORD_KEYWORDS = {
    "password", "pwd", "pass code", "passcode", "forgot password", "reset password",
    "login failed", "login issue", "account locked", "lockout", "unlock account",
    "user locked", "credential", "cannot signin", "can't sign in", "sign-in issue"
}
def looks_like_password_issue(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in PASSWORD_KEYWORDS)

async def stage1_extract_all_causes(session_id: str, user_problem: str) -> ChatResponse:
    # ── Dynamic SOP-similarity scope guard ──────────────────────────────────
    try:
        in_scope = _is_in_scope_sync(user_problem)   # sync call, no await
    except Exception as e:
        logger.warning(f"[ScopeCheck] exception during check: {e}")
        in_scope = True  # fail-open

    if not in_scope:
        return ChatResponse(
            session_id=session_id,
            answer=(
                "Sorry, your question doesn't seem to match anything in my knowledge base. "
                "I can only assist with issues covered by the SOPs I have been trained on. 😊\n\n"
                "You can raise a ticket with details of your issue here: {https://www.servicenow.com/}"
            )
        )
    # ── END scope guard ──────────────────────────────────────────────────────

    if not kedb_store or not sop_chunks:
        return ChatResponse(
            session_id=session_id,
            answer=(
                f"I'm ready to dig in, but I can't access the SOP store right now. "
                f"If it's okay, please raise a ticket so we can proceed: {SERVICENOW_URL}"
            )
        )

    exclude_pw = not looks_like_password_issue(user_problem)
    contexts = _retrieve_sop_contexts(user_problem, exclude_password=exclude_pw)
    logger.info(f"[Stage1] First-pass contexts: {len(contexts)} (exclude_password={exclude_pw})")

    merged: List[Dict[str, Any]] = []

    async def _merge(ctx: str):
        se = await sop_structured_extract(user_problem, ctx)
        if se and se.causes:
            for c in se.causes:
                merged.append(c.dict())
            return
        sop_extraction, raw = await ainvoke_json(
            llm, SOP_EXTRACT_SYS, SOP_EXTRACT_USER.format(problem=user_problem, context=ctx))
        if sop_extraction is None:
            logger.warning(f"[SOP-EXTRACT] parse failed. Raw head: {raw[:220]!r}")
            if DEBUG_LLM:
                logger.error(f"[DEBUG_LLM] SOP-EXTRACT RAW:\n{raw}")
            return
        for c in (sop_extraction.get("causes") or []):
            merged.append(c)

    for ctx in contexts:
        await _merge(ctx)

    if len(merged) < MIN_CAUSES_TRIGGER_FALLBACK:
        logger.info(f"[Stage1] causes below threshold ({len(merged)}). Fallback to full SOP.")
        for ctx in _full_sop_fallback_contexts(exclude_password=exclude_pw):
            if len(merged) >= MIN_CAUSES_TRIGGER_FALLBACK * 3:
                break
            await _merge(ctx)

    if not merged:
        filtered = _all_sop_chunks(exclude_pw)
        raw_blob = "\n\n---\n\n".join([d.page_content for d in filtered[:8]])
        cheap = cheap_root_cause_scan(raw_blob)
        merged = cheap.get("causes", [])

    if not merged:
        return ChatResponse(
            session_id=session_id,
            answer=(
                "I couldn't extract clear root causes from the SOP just yet.\n"
                "If you can describe the issue in a line or two, I'll try again immediately."
            )
        )

    merged = dedup_and_limit_causes(merged, max_causes=MAX_CAUSES_TO_SHOW)

    ss = S(session_id)
    ss.problem = user_problem
    ss.sop_extraction_json = {"problem": user_problem, "causes": merged}
    ss.awaiting_log_confirm = True

    return ChatResponse(session_id=session_id, answer=render_sop_causes_bullets(ss.sop_extraction_json))



# -------------------------------------------------------------------
# Prior incidents — hybrid ranking, dedup, variable length (1–3)
# -------------------------------------------------------------------
def looks_like_prior_incident_request(q: str) -> bool:
    ql = (q or "").lower()
    for p in ["prior incident","previous incident","related incident","earlier incident",
              "past incident","any incidents like this","incident history","previous cases",
              "similar incidents","prior cases","seen this before"]:
        if p in ql:
            return True
    return False

def _parse_when(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None

def _within_last_six_months(dt: datetime, now_utc: datetime) -> bool:
    return dt >= (now_utc - timedelta(days=183))

async def summarize_prior_incidents(session_id: str) -> ChatResponse:
    ss = S(session_id)
    cause = (ss.selected_cause or "").strip()
    if not cause:
        sop_json = ss.sop_extraction_json or {}
        allc = sop_json.get("causes") or []
        if allc:
            cause = (allc[0].get("title") or allc[0].get("cause") or "").strip()

    if not cause:
        return ChatResponse(session_id=session_id,
            answer=("I want to help with similar cases, but I don't have a selected root cause yet.\n"
                    "If you'd like, tell me the issue in a line, and I'll look up matching incidents right away."))

    if not hist_store:
        return ChatResponse(session_id=session_id,
            answer=("I'm with you—it's helpful to know if others hit this recently. "
                    "Right now, I don't have access to the historical incident store. "
                    "If you can share or mount the CSV data, I'll fetch the latest matches for you."))

    hosts = _extract_hostnames_from_logs(ss._log_json_cache or {})
    dns_cause = _is_dns_cause(cause)
    enriched = cause + (" " + " ".join(hosts) if dns_cause and hosts else "")
    try:
        hits = hist_store.similarity_search_with_score(enriched, k=TOP_K_INCIDENTS)
    except Exception as e:
        logger.warning(f"Historical search failed: {e}")
        hits = []

    now_utc = datetime.now(timezone.utc)
    scored: List[Tuple[Document,float,Optional[datetime]]] = []
    for d, dist in hits:
        sim = norm_distance_to_sim(dist)
        blob = (d.page_content or "") + " " + " ".join([
            d.metadata.get("short_description",""),
            d.metadata.get("description",""),
            d.metadata.get("resolution_note","")
        ])
        boost = _keyword_boost(blob, (_DNS_TOKENS if dns_cause else set()), hosts)
        final = max(0.0, min(1.0, 0.7*sim + boost))
        dt = _parse_when((d.metadata.get("created_at") or d.metadata.get("opened_at") or "").strip())
        scored.append((d, final, dt))

    by_id: Dict[str, Tuple[Document,float,Optional[datetime]]] = {}
    by_hash: Dict[str, Tuple[Document,float,Optional[datetime]]] = {}
    for d, sc, dt in scored:
        iid = (d.metadata.get("incident_id") or "").strip()
        h = _content_hash(d)
        if iid:
            if (iid not in by_id) or (sc > by_id[iid][1]): by_id[iid] = (d,sc,dt)
        else:
            if (h not in by_hash) or (sc > by_hash[h][1]): by_hash[h] = (d,sc,dt)

    uniq = list(by_id.values()) + list(by_hash.values())
    last6 = [(d,sc,dt) for (d,sc,dt) in uniq if dt and _within_last_six_months(dt, now_utc)]
    last6.sort(key=lambda x: x[1], reverse=True)
    unknown = [(d,sc,dt) for (d,sc,dt) in uniq if dt is None]
    unknown.sort(key=lambda x: x[1], reverse=True)

    count_last6 = len(last6)
    top: List[Tuple[Document,float]] = [(d,sc) for (d,sc,_) in last6[:3]]
    if len(top) < 3:
        top.extend([(d,sc) for (d,sc,_) in unknown[:(3-len(top))]])
    top = top[:len(top)]

    if count_last6 > 0:
        intro = (f"I hear you—it's completely fair to check how often this has happened. "
                 f"In the last six months, I found **{count_last6}** incident(s) related to **{cause}**. "
                 f"Here {('is' if len(top)==1 else 'are')} the top {len(top)} most relevant:")
    else:
        if len(top) == 0:
            return ChatResponse(session_id=session_id,
                answer=(f"I looked for prior incidents related to **{cause}**, but I didn't find solid matches yet. "
                        f"If you can share the latest incident export, I'll run this again right away."))
        intro = (f"I checked for similar cases linked to **{cause}**. "
                 f"Here {('is' if len(top)==1 else 'are')} the top {len(top)} closest match{'' if len(top)==1 else 'es'}:")

    lines = [intro]
    for d, _ in top:
        inc_id = (d.metadata.get("incident_id") or "N/A").strip()
        desc   = (d.metadata.get("description") or d.page_content or "").strip()
        resol  = (d.metadata.get("resolution_note") or "").strip()
        create = (d.metadata.get("created_at") or "").strip()

        # compress to first useful line
        if "\n" in desc:
            for part in desc.splitlines():
                p = part.strip(); low = p.lower()
                if p and not (low.startswith("short:") or low.startswith("desc:")
                              or low.startswith("resolution:") or low.startswith("worknote:")):
                    desc = p; break

        if len(desc)  > 280: desc  = desc[:280] + "…"
        if len(resol) > 380: resol = resol[:380] + "…"

        lines.append(
            f"- **{inc_id}**" + (f" _(Reported: {create})_" if create else "")
            + f"\n  - **Description:** {desc or 'Not available'}"
            + f"\n  - **Resolution note:** {resol or 'Not captured'}"
        )

    lines.append("\nIf you'd like, I can open any of these or compare them with today's logs. I'm here to help.")
    return ChatResponse(session_id=session_id, answer="\n".join(lines))

# -------------------------------------------------------------------
# Intents: version Q&A, DNS blocking Q&A, contextual Q&A
# -------------------------------------------------------------------
def looks_like_deploy_question(q: str) -> bool:
    ql = (q or "").lower()
    keys = ["last deploy","last deployment","latest version","current version","version deployed",
            "latest build","release version","what version","deployment status","when deployed"]
    return any(k in ql for k in keys)

def looks_like_dns_block_question(q: str) -> bool:
    ql = (q or "").lower()
    return ("dns" in ql and ("block" in ql or "blocked" in ql or "why failing" in ql or "what dns is blocking" in ql))

def summarize_version_conversational(data: dict, deploy_tail: str) -> str:
    ver = (data or {}).get("direct_answer","").strip()
    ev  = (data or {}).get("evidence_lines",[])[:2]
    ev  = [compress_phrase(x) for x in ev if compress_phrase(x)]
    if not ver or ver.lower() == "not found":
        m = re.search(r"\bversion\s+(?:now\s+)?active\s+v?([0-9]+\.[0-9]+\.[0-9]+)", deploy_tail, re.I)
        if m:
            ver = m.group(1)
    if ver:
        detail = ""
        for e in ev:
            if any(k in e.lower() for k in ["active","stable","no rollback"]):
                detail = e; break
        if not detail and ev:
            detail = ev[0]
        if detail:
            return f"We’re currently running **v{ver}**. {detail}."
        return f"We’re currently running **v{ver}**."
    return "I wasn’t able to confirm the active version from the deployment logs just yet."

async def answer_deploy_question(session_id: str, question: str) -> ChatResponse:
    deploy_tail = tail(DEPLOY_LOG_PATH)
    data, raw = await ainvoke_json(llm, LOG_QA_SYS, LOG_QA_USER.format(question=question, deploy=deploy_tail))
    if data is None:
        if DEBUG_LLM: logger.error(f"[DEBUG_LLM] LOG QA RAW:\n{raw}")
        data = {"direct_answer":"Not found", "evidence_lines":[], "confidence":"low"}
    msg = summarize_version_conversational(data, deploy_tail)
    return ChatResponse(session_id=session_id, answer=msg)

async def answer_dns_blocking_question(session_id: str, question: str) -> ChatResponse:
    ss = S(session_id)
    log_json = ss._log_json_cache or {"domains":[]}

    network_hits = []
    dns_hits = []

    # Structured evidence (if any)
    for item in (log_json.get("domains") or []):
        dom = (item.get("domain") or "").upper()
        for ln in (item.get("evidence_lines") or []):
            s = strip_timestamps(ln)
            if any(k in s.upper() for k in ["FIREWALL", "BLOCK", "DNS.EXE", "DESTPORT=53", "PROTO=UDP", "UDP/53", "OUTBOUND-DNS-BLOCK"]):
                if dom == "DNS":
                    dns_hits.append(s)
                elif dom in {"NETWORK","APPLICATION"}:
                    network_hits.append(s)

    # Raw tails fallback
    if not dns_hits:
        dns_tail = tail(DNS_LOG_PATH)
        for ln in dns_tail.splitlines()[-200:]:
            s = strip_timestamps(ln)
            if any(k in s.upper() for k in ["FIREWALL", "BLOCK", "DNS.EXE", "DESTPORT=53", "PROTO=UDP", "UDP/53", "OUTBOUND-DNS-BLOCK"]):
                dns_hits.append(s)

    if not network_hits:
        net_tail = tail(NET_LOG_PATH)
        for ln in net_tail.splitlines()[-200:]:
            s = strip_timestamps(ln)
            if any(k in s.upper() for k in ["FIREWALL", "BLOCK", "DNS.EXE", "DESTPORT=53", "PROTO=UDP", "UDP/53", "OUTBOUND-DNS-BLOCK"]):
                network_hits.append(s)

    parts = []
    if dns_hits:
        parts.append(f"In **{os.path.basename(DNS_LOG_PATH)}**, DNS-related blocking is indicated.")
    if network_hits:
        parts.append(f"In **{os.path.basename(NET_LOG_PATH)}**, network controls indicate DNS traffic interference.")

    if not parts:
        return ChatResponse(
            session_id=session_id,
            answer=(
                f"I checked **{os.path.basename(DNS_LOG_PATH)}** and **{os.path.basename(NET_LOG_PATH)}**, "
                f"but didn’t see clear signs of DNS blocking. If you’d like, I can run live queries and monitor."
            )
        )

    return ChatResponse(session_id=session_id, answer=" ".join(parts))

# -------------------------------------------------------------------
# Contextual Q&A
# -------------------------------------------------------------------
def looks_like_general_question(q: str) -> bool:
    ql = (q or "").lower()
    special = [
        "root causes","show root causes","list root causes","show causes",
        "check logs","analyze logs","show resolution","resolution","fix","how to fix",
        "prior incident","previous incident","related incident","similar incidents",
        "last deploy","latest version","deployment status","what version","version deployed",
        "what dns is blocking","dns is blocking","why dns failing"
    ]
    return not any(p in ql for p in special)

async def answer_contextual_question(session_id: str, question: str) -> ChatResponse:
    ss = S(session_id)
    logs = ss._log_json_cache or {"domains":[]}
    summaries = []
    for item in (logs.get("domains") or []):
        dom = (item.get("domain") or "").upper()
        evs = [strip_timestamps(x) for x in (item.get("evidence_lines") or [])[:2]]
        evs = [x for x in evs if x]
        if evs: summaries.append(f"- {dom}: " + "; ".join(evs))
        else:   summaries.append(f"- {dom}: no notable signals")
    sop_json = ss.sop_extraction_json or {"causes":[]}
    cause_titles = [c.get("title") or c.get("cause") or "" for c in (sop_json.get("causes") or [])]

    sys = SystemMessage(content=CONTEXT_QA_SYS)
    usr = HumanMessage(content=CONTEXT_QA_USER.format(
        problem=ss.problem or "(not provided)",
        selected_cause=ss.selected_cause or "(none selected yet)",
        cause_titles=", ".join([t for t in cause_titles if t]) or "(not extracted yet)",
        logs_summary="\n".join(summaries) or "(no logs analyzed yet)",
        question=question
    ))
    try:
        resp = await llm.ainvoke([sys, usr]); answer = (resp.content or "").strip()
        if not answer:
            answer = ("I don’t have enough context to answer that confidently yet. "
                      "If you share a bit more detail, I’ll dig right in.")
    except Exception:
        answer = ("I’m running into an issue answering that right now. "
                  "If you can rephrase or share a bit more context, I’ll try again.")
    return ChatResponse(session_id=session_id, answer=answer)

# -------------------------------------------------------------------
# Password-issue branch (reads Status + Account type)
# -------------------------------------------------------------------
def _detect_id_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["account id","accountid","acct_id","acctid","user id","userid","user_id","login","username","uid"]
    cols = []
    for c in df.columns:
        if c.lower().strip() in candidates:
            cols.append(c)
    if not cols:
        for c in df.columns:
            low = c.lower()
            if any(k in low for k in ["account","user","login","name","uid"]):
                cols.append(c)
    return cols

def _detect_status_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower().strip() in ("status","state"):
            return c
    for c in df.columns:
        if c.lower().strip() in ("active","isactive","enabled","isenabled"):
            return c
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in ["status","active","enable","lock"]):
            return c
    return None

def _detect_account_type_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower().strip() in ("account type","acct_type","type","accttype","user type","usertype"):
            return c
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in ["type","cloud","hybrid","on-prem","onprem","aad","azure","ad connect","synced"]):
            return c
    return None

def _normalize_status(val: Any) -> str:
    s = str(val).strip().lower()
    if s in {"active","true","yes","enabled","enable","1"}:
        return "active"
    if s in {"locked","lock","lockout"}:
        return "locked"
    if s in {"disabled","disable","deactivated","deactivate"}:
        return "disabled"
    if s in {"inactive","false","no","0"}:
        return "inactive"
    return s if s else "unknown"

def _normalize_acct_type(val: Any) -> str:
    s = str(val).strip().lower()
    if any(k in s for k in ["inactive","deactivated","disabled"]):
        return "inactive"
    if any(k in s for k in ["cloud","azure ad","aad","microsoft entra","entra"]):
        return "cloud"
    if any(k in s for k in ["hybrid","ad connect","connect","sync","synchronized","synced","hybrid-joined"]):
        return "hybrid"
    if any(k in s for k in ["onprem","on-prem","on premise","on-premise","ad only","local ad","domain only"]):
        return "onprem"
    if s in {"", "unknown", "n/a", "na"}:
        return "unknown"
    if "cloud" in s: return "cloud"
    if "hybrid" in s or "connect" in s or "sync" in s: return "hybrid"
    if "prem" in s or "local" in s or "domain" in s: return "onprem"
    return "unknown"

def _find_account_record(account_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    df = _load_account_df()
    if df is None or df.empty:
        return "no_csv", None
    id_cols     = _detect_id_columns(df)
    status_col  = _detect_status_column(df)
    type_col    = _detect_account_type_column(df)

    if not id_cols:
        return "no_id_columns", None

    mask = None
    for col in id_cols:
        colmask = df[col].astype(str).str.strip().str.lower() == account_id.strip().lower()
        mask = colmask if mask is None else (mask | colmask)
    matches = df[mask] if mask is not None else pd.DataFrame()

    if matches.empty:
        for col in id_cols:
            colmask = df[col].astype(str).str.strip().str.lower().str.contains(re.escape(account_id.strip().lower()), na=False)
            matches = df[colmask]
            if not matches.empty:
                break

    if matches.empty:
        return "not_found", None

    row = matches.iloc[0].to_dict()
    status_val = _normalize_status(row.get(status_col, "")) if status_col else "unknown"
    type_val   = _normalize_acct_type(row.get(type_col, "")) if type_col else "unknown"

    row["_normalized_status"]   = status_val
    row["_status_column"]       = status_col or ""
    row["_matched_id_columns"]  = id_cols
    row["_normalized_acct_type"]= type_val
    row["_acct_type_column"]    = type_col or ""
    return "ok", row

def _filter_chunks_for_password_sop() -> List[Document]:
    pw_chunks = [d for d in sop_chunks if d.metadata.get("filename","").lower() == PASSWORD_SOP_FILENAME.lower()]
    if pw_chunks:
        return pw_chunks
    pw_chunks = [d for d in sop_chunks if "password" in d.metadata.get("filename","").lower()]
    return pw_chunks

def _contexts_from_specific_docs(docs: List[Document]) -> List[str]:
    return _build_context_blobs_from_docs(docs)

async def _extract_password_sop(problem_hint: str) -> Dict[str, Any]:
    contexts: List[str] = []
    pw_chunks = _filter_chunks_for_password_sop()
    if pw_chunks:
        contexts = _contexts_from_specific_docs(pw_chunks)
    else:
        contexts = _retrieve_sop_contexts(problem_hint or "password reset", exclude_password=False)

    if not contexts:
        return {"causes": []}

    merged: List[Dict[str,Any]] = []
    async def _merge(ctx: str):
        se = await sop_structured_extract(problem_hint, ctx)
        if se and se.causes:
            for c in se.causes:
                merged.append(c.dict())
            return
        sop_extraction, raw = await ainvoke_json(
            llm, SOP_EXTRACT_SYS, SOP_EXTRACT_USER.format(problem=problem_hint, context=ctx))
        if sop_extraction is None:
            if DEBUG_LLM: logger.error(f"[DEBUG_LLM] PW-SOP RAW:\n{raw}")
            return
        for c in (sop_extraction.get("causes") or []):
            merged.append(c)

    for ctx in contexts:
        await _merge(ctx)

    return {"causes": merged}

def _choose_password_cause(causes: List[Dict[str, Any]], status: str, acct_type: str) -> Dict[str, Any] | None:
    if not causes:
        return None
    status = (status or "unknown").lower()
    acct_type = (acct_type or "unknown").lower()

    prefs: List[str] = []
    if acct_type == "cloud":
        prefs += ["cloud", "azure ad", "microsoft entra", "aad", "sspr", "self-service", "mfa", "password writeback"]
    elif acct_type == "hybrid":
        prefs += ["hybrid", "ad connect", "azure ad connect", "sync", "synchronized", "password writeback", "federation", "pass-through authentication"]
    elif acct_type == "onprem":
        prefs += ["on-prem", "onprem", "active directory", "domain", "local ad", "group policy"]
    elif acct_type == "inactive":
        prefs += ["inactive", "reactivate", "enable", "re-enable"]

    if status == "locked":
        prefs += ["locked", "lockout", "too many attempts", "unlock"]
    elif status in {"disabled","inactive"}:
        prefs += ["disabled", "deactivated", "inactive", "reactivate", "enable account", "re-enable"]
    elif status == "active":
        prefs += ["forgot password", "password reset", "reset password", "user forgot", "credential reset"]

    if not prefs:
        prefs += ["password", "reset", "unlock", "enable", "activate", "reactivate", "writeback", "sspr"]

    candidate_pool = [c for c in causes if (c.get("resolution") or "").strip()] or causes

    def score(c: Dict[str,Any]) -> int:
        txt = (c.get("title","") + " " + c.get("short_description","")).lower()
        sc = 0
        for w in prefs:
            if w in txt:
                sc += 2
        if c.get("has_steps"): sc += 1
        return sc

    best = sorted(candidate_pool, key=score, reverse=True)[0]
    return best

def _format_password_recommendation(account_id: str, status_label: str, acct_type: str, cause: Dict[str, Any]) -> str:
    cause_title = cause.get("title","(Password SOP)")
    resolution_bullets = cause.get("resolution_bullets") or []
    resolution_line = cause.get("resolution","")
    res_md = format_resolution_bullets(resolution_bullets, resolution_line)

    status_human = status_label.capitalize() if status_label else "Unknown"
    type_map = {"cloud":"Cloud-only", "hybrid":"Hybrid", "onprem":"On‑prem", "inactive":"Inactive", "unknown":"Unknown"}
    type_human = type_map.get((acct_type or "unknown").lower(), "Unknown")

    header = "##### 🔐 Password assistance"
    acct_block = (
        f"**Account/User ID:** `{account_id}`  \n"
        f"**Account status:** **{status_human}**  \n"
        f"**Account type:** **{type_human}**"
    )
    cause_line = f"**Relevant SOP cause:** {cause_title}"
    steps = f"**Recommended next steps (from Password Reset SOP):**\n{res_md}"
    return f"{header}\n\n{acct_block}\n\n{cause_line}\n\n{steps}"

async def handle_password_issue(session_id: str, user_input: str) -> ChatResponse:
    ss = S(session_id)

    # Awaiting the account id
    if ss.awaiting_account_id and not ss.provided_account_id:
        acct = user_input.strip()
        if not acct:
            return ChatResponse(session_id=session_id, answer="Please share the **Account ID/User ID** to proceed with password assistance.")
        ss.provided_account_id = acct

        status, record = _find_account_record(acct)
        if status == "no_csv":
            pw = await _extract_password_sop("password reset")
            cause = _choose_password_cause(pw.get("causes", []), "unknown", "unknown")
            if not cause:
                msg = ("I couldn’t locate the Password Reset SOP content right now. "
                       "Please raise a ticket and we’ll assist immediately: " + SERVICENOW_URL)
                return ChatResponse(session_id=session_id, answer=msg)
            msg = (
                "I couldn't open the account list, but here are the general steps:\n\n" +
                _format_password_recommendation(acct, "unknown", "unknown", cause) +
                "\n\nIf this doesn’t help, I can route this to the service desk."
            )
            ss.password_flow_active = False
            ss.awaiting_account_id = False
            return ChatResponse(session_id=session_id, answer=msg)

        if status in {"no_id_columns","not_found"}:
            ask = (
                "I couldn’t find that ID in **Acct_details.csv**. "
                "Please re-check the ID or share an alternate identifier.\n\n"
                "If you prefer, I can share general Password Reset steps without account verification. Would you like that? (yes/no)"
            )
            return ChatResponse(session_id=session_id, answer=ask)

        # Found record: get status + type
        ss.account_status_label = (record or {}).get("_normalized_status","unknown")
        ss.account_type_label   = (record or {}).get("_normalized_acct_type","unknown")

        pw = await _extract_password_sop(ss.account_type_label or "password reset")
        cause = _choose_password_cause(pw.get("causes", []), ss.account_status_label, ss.account_type_label)
        if not cause:
            msg = ("I couldn’t extract clear steps from the Password Reset SOP right now. "
                   "Please raise a ticket and we’ll assist immediately: " + SERVICENOW_URL)
            ss.password_flow_active = False
            ss.awaiting_account_id = False
            return ChatResponse(session_id=session_id, answer=msg)

        answer = _format_password_recommendation(acct, ss.account_status_label, ss.account_type_label, cause)
        ss.password_flow_active = False
        ss.awaiting_account_id = False
        return ChatResponse(session_id=session_id, answer=answer)

    # Start password flow
    ss.password_flow_active = True
    ss.awaiting_account_id = True
    return ChatResponse(
        session_id=session_id,
        answer="It looks like a **password-related** issue. Please provide the **Account ID/User ID** to proceed."
    )

# -------------------------------------------------------------------
# API models
# -------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str]
    message: str

# -------------------------------------------------------------------
# Chat Router — with password flow precedence & SOP isolation
# -------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    text = (req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message.")
    ss = S(session_id)

    try:
        # Password flow — awaiting account ID
        if ss.password_flow_active and ss.awaiting_account_id:
            yn, _ = await ainvoke_json(llm, YESNO_SYS, YESNO_USER.format(reply=text))
            ans = ((yn or {}).get("answer") or "").upper()
            if ans == "YES" and not ss.provided_account_id:
                pw = await _extract_password_sop("password reset")
                cause = _choose_password_cause(pw.get("causes", []), "unknown", "unknown")
                if not cause:
                    ss.password_flow_active = False
                    ss.awaiting_account_id = False
                    return ChatResponse(
                        session_id=session_id,
                        answer=(
                            "I couldn't locate the Password Reset SOP content right now. "
                            f"Please raise a ticket and we'll assist immediately: {SERVICENOW_URL}"
                        )
                    )
                msg = _format_password_recommendation("(not verified)", "unknown", "unknown", cause)
                ss.password_flow_active = False
                ss.awaiting_account_id = False
                return ChatResponse(session_id=session_id, answer=msg)
            return await handle_password_issue(session_id, text)

        # Show causes command
        if text.lower() in {"root causes", "show root causes", "list root causes", "show causes"}:
            if ss.sop_extraction_json:
                return ChatResponse(
                    session_id=session_id,
                    answer=render_sop_causes_bullets(ss.sop_extraction_json)
                )
            return await stage1_extract_all_causes(session_id, text)

        # Awaiting resolution confirm
        if ss.awaiting_resolution_confirm:
            yn, _ = await ainvoke_json(llm, YESNO_SYS, YESNO_USER.format(reply=text))
            ans = ((yn or {}).get("answer") or "").upper()
            if ans == "NO":
                ss.awaiting_resolution_confirm = False
                return ChatResponse(
                    session_id=session_id,
                    answer="No worries. If you need the recommended resolution later, just say **resolution**."
                )
            if ans == "YES":
                ss.awaiting_resolution_confirm = False
                return await finish_with_resolution_and_incident(session_id)
            return ChatResponse(
                session_id=session_id,
                answer="Please reply **yes** or **no** to continue."
            )

        # Awaiting log confirm
        if ss.awaiting_log_confirm:
            yn, _ = await ainvoke_json(llm, YESNO_SYS, YESNO_USER.format(reply=text))
            ans = ((yn or {}).get("answer") or "").upper()
            if ans == "NO":
                ss.awaiting_log_confirm = False
                return ChatResponse(
                    session_id=session_id,
                    answer="Okay. If you want me to analyze logs later, just say **check logs**."
                )
            if ans == "YES":
                ss.awaiting_log_confirm = False
                if not ss.sop_extraction_json:
                    return ChatResponse(
                        session_id=session_id,
                        answer="Looks like I lost the SOP context. Could you describe the issue again?"
                    )
                return await run_stage2_show_checked_table(session_id)
            return ChatResponse(
                session_id=session_id,
                answer="Please reply **yes** or **no** to continue with analysis."
            )

        # Password intent
        if looks_like_password_issue(text):
            return await handle_password_issue(session_id, text)

        # Other specific intents
        if looks_like_prior_incident_request(text):
            return await summarize_prior_incidents(session_id)
        if looks_like_deploy_question(text):
            return await answer_deploy_question(session_id, text)
        if looks_like_dns_block_question(text):
            return await answer_dns_blocking_question(session_id, text)

        # First turn → Stage-1
        if ss.sop_extraction_json is None:
            return await stage1_extract_all_causes(session_id, text)

        # Contextual Q&A fallback
        return await answer_contextual_question(session_id, text)

    except Exception as e:
        # Safety net — never return None, always return a valid ChatResponse
        logger.error(f"[chat] Unhandled exception for session {session_id}: {e}", exc_info=True)
        return ChatResponse(
            session_id=session_id,
            answer=(
                "I ran into an unexpected issue processing your request. "
                "Please try again, or raise a ticket if this persists: "
                f"{SERVICENOW_URL}"
            )
        )
    
# Optional: deploy QA endpoint
@app.post("/ask-logs", response_model=ChatResponse)
async def ask_logs(req: ChatRequest):
    return await answer_deploy_question(req.session_id or str(uuid.uuid4()), req.message)

# Health
@app.get("/metrics")
def metrics():
    return {
        "status":"ok",
        "sop_chunks": len(sop_chunks),
        "hist_docs": len(hist_docs),
        "config": {
            "TOP_K_SOP": TOP_K_SOP, "MAX_SOP_CHARS": MAX_SOP_CHARS,
            "MAX_CONTEXTS": MAX_CONTEXTS, "TOTAL_CONTEXT_CHAR_BUDGET": TOTAL_CONTEXT_CHAR_BUDGET,
            "MIN_CAUSES_TRIGGER_FALLBACK": MIN_CAUSES_TRIGGER_FALLBACK, "MAX_CAUSES_TO_SHOW": MAX_CAUSES_TO_SHOW,
            "SOP_CHUNK_SIZE": SOP_CHUNK_SIZE, "SOP_CHUNK_OVERLAP": SOP_CHUNK_OVERLAP,
            "ACCOUNT_CSV_PATH": ACCOUNT_CSV_PATH,
            "PASSWORD_SOP_FILENAME": PASSWORD_SOP_FILENAME
        }
    }