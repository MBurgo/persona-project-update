import os
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Optional imports (fail gracefully so the app still opens)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Foolish Persona Portal", layout="centered", page_icon="F")


# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stButton>button{border:1px solid #485cc7;border-radius:8px;width:100%}
    .chat-bubble {
        padding: 15px; border-radius: 10px; margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-bubble { background-color: #f0f2f6; border-left: 5px solid #485cc7; }
    .bot-bubble { background-color: #e3f6d8; border-left: 5px solid #43B02A; }
    .small-muted { color: #53565A; font-size: 0.9rem; }
    code { white-space: pre-wrap !important; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"


def normalize_dashes(s: str) -> str:
    return re.sub(f"[{DASH_CHARS}]", "-", s or "")


def slugify(s: str) -> str:
    s = normalize_dashes(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def first_present(*values: Optional[str]) -> Optional[str]:
    for v in values:
        if v is not None and str(v).strip() != "":
            return v
    return None


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from model output."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (English): ~4 chars per token."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def truncate_to_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    if max_words <= 0:
        return ""
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "\n\n[...truncated...]"


def strip_common_email_footer(text: str) -> str:
    """Best-effort removal of common email boilerplate (unsubscribe/legal/footer)."""
    if not text:
        return ""
    lines = text.splitlines()
    patterns = [
        r"(?i)\bunsubscribe\b",
        r"(?i)\bmanage preferences\b",
        r"(?i)\bprivacy policy\b",
        r"(?i)\bterms\b",
        r"(?i)\bview in (a )?browser\b",
        r"(?i)^\s*copyright\b",
        r"(?i)^\s*\u00a9\b",  # may appear in text; safe in regex
        r"(?i)\bdisclaimer\b",
        r"(?i)\bthis email\b",
    ]

    for i, line in enumerate(lines):
        if any(re.search(p, line) for p in patterns):
            trimmed = "\n".join(lines[:i]).strip()
            return trimmed if trimmed else text.strip()
    return text.strip()


def claim_risk_flags(text: str) -> List[str]:
    """Very lightweight claim-risk heuristic for marketing copy."""
    if not text:
        return []
    t = text.lower()
    patterns = {
        "Guaranteed / certainty language": ["guaranteed", "can't lose", "sure thing", "no risk", "risk-free", "100%"],
        "Urgency pressure": ["urgent", "act now", "limited time", "today only", "last chance"],
        "Implied future performance": ["will double", "will triple", "can't miss", "next nvidia"],
        "Overly absolute claims": ["always", "never", "everyone", "no one"],
    }
    hits: List[str] = []
    for label, toks in patterns.items():
        if any(tok in t for tok in toks):
            hits.append(label)
    return hits


def format_brief_summary(brief: Dict[str, Any]) -> str:
    """Convert extracted brief JSON into a compact, human-readable summary."""
    if not isinstance(brief, dict):
        return ""

    def _join(xs: Any, n: int = 6) -> str:
        xs = _ensure_list(xs)
        xs = [str(x).strip() for x in xs if str(x).strip()]
        return "; ".join(xs[:n])

    primary_promise = brief.get("primary_promise") or brief.get("promise") or ""
    offer = brief.get("offer_summary") or brief.get("offer") or ""
    cta = brief.get("cta") or ""
    key_claims = _join(brief.get("key_claims"), 8)
    proof = _join(brief.get("proof_elements"), 8)
    missing = _join(brief.get("missing_proof"), 8)

    bits: List[str] = []
    if primary_promise:
        bits.append(f"Primary promise: {primary_promise}")
    if offer:
        bits.append(f"Offer: {offer}")
    if cta:
        bits.append(f"CTA: {cta}")
    if key_claims:
        bits.append(f"Key claims: {key_claims}")
    if proof:
        bits.append(f"Proof present: {proof}")
    if missing:
        bits.append(f"Missing proof: {missing}")

    return "\n".join(bits).strip()


def build_brief_prompt(copy_type: str, text: str, truncated_note: str) -> str:
    """Prompt to extract a structured creative brief from pasted copy."""
    copy_type = copy_type or "Other"
    return f"""
You are a senior conversion copy strategist.

TASK:
Extract a structured brief from the marketing creative provided below. This brief will be used to run a synthetic focus group.

RULES:
- Return ONLY a single JSON object. No markdown. No commentary.
- Do NOT invent facts. If something is missing, say it is missing.
- Be concise. Use short strings. Lists should be short.
- If the creative is long, try to detect sections and summarise them.

COPY TYPE (declared by user): {copy_type}
TRUNCATION NOTE: {truncated_note}

OUTPUT JSON SCHEMA:
{{
  "copy_type": "{copy_type}",
  "primary_promise": "...",
  "offer_summary": "...",
  "cta": "...",
  "audience_assumed": "...",
  "tone": ["..."],
  "key_claims": ["..."],
  "proof_elements": ["..."],
  "missing_proof": ["..."],
  "risk_flags": ["..."],
  "sections": [
    {{"section": "Hero/Subject/Hook", "summary": "...", "start_quote": "..."}},
    {{"section": "Offer", "summary": "...", "start_quote": "..."}}
  ]
}}

MARKETING CREATIVE:
{text}
""".strip()


def moderator_schema_example(copy_type: str) -> str:
    ct = (copy_type or "Other").strip()

    if ct == "Headline":
        return (
            "{\n"
            "  \"copy_type\": \"Headline\",\n"
            "  \"scope\": {\"mode\": \"...\", \"note\": \"...\"},\n"
            "  \"executive_summary\": \"...\",\n"
            "  \"real_why\": \"...\",\n"
            "  \"trust_gap\": \"...\",\n"
            "  \"key_objections\": [\"...\"],\n"
            "  \"proof_needed\": [\"...\"],\n"
            "  \"risk_flags\": [\"...\"],\n"
            "  \"actionable_fixes\": [\"...\"],\n"
            "  \"rewrite\": {\n"
            "    \"headlines\": [\"...\"],\n"
            "    \"supporting_lines\": [\"...\"],\n"
            "    \"angle_notes\": \"...\"\n"
            "  }\n"
            "}"
        )

    if ct == "Email":
        return (
            "{\n"
            "  \"copy_type\": \"Email\",\n"
            "  \"scope\": {\"mode\": \"...\", \"note\": \"...\"},\n"
            "  \"executive_summary\": \"...\",\n"
            "  \"real_why\": \"...\",\n"
            "  \"trust_gap\": \"...\",\n"
            "  \"key_objections\": [\"...\"],\n"
            "  \"proof_needed\": [\"...\"],\n"
            "  \"confusing_phrases\": [\"...\"],\n"
            "  \"risk_flags\": [\"...\"],\n"
            "  \"actionable_fixes\": [\"...\"],\n"
            "  \"rewrite\": {\n"
            "    \"subject\": \"...\",\n"
            "    \"preheader\": \"...\",\n"
            "    \"body\": \"...\",\n"
            "    \"cta\": \"...\",\n"
            "    \"ps\": \"...\"\n"
            "  }\n"
            "}"
        )

    if ct == "Sales Page":
        return (
            "{\n"
            "  \"copy_type\": \"Sales Page\",\n"
            "  \"scope\": {\"mode\": \"...\", \"note\": \"...\"},\n"
            "  \"executive_summary\": \"...\",\n"
            "  \"real_why\": \"...\",\n"
            "  \"trust_gap\": \"...\",\n"
            "  \"key_objections\": [\"...\"],\n"
            "  \"proof_needed\": [\"...\"],\n"
            "  \"risk_flags\": [\"...\"],\n"
            "  \"actionable_fixes\": [\"...\"],\n"
            "  \"section_feedback\": [\n"
            "    {\"section\": \"Hero\", \"what_works\": \"...\", \"what_fails\": \"...\", \"fix\": \"...\"}\n"
            "  ],\n"
            "  \"rewrite\": {\n"
            "    \"hero_headline\": \"...\",\n"
            "    \"hero_subheadline\": \"...\",\n"
            "    \"bullets\": [\"...\"],\n"
            "    \"proof_block\": \"...\",\n"
            "    \"offer_stack\": [\"...\"],\n"
            "    \"cta_button\": \"...\",\n"
            "    \"cta_line\": \"...\"\n"
            "  }\n"
            "}"
        )

    return (
        "{\n"
        "  \"copy_type\": \"Other\",\n"
        "  \"scope\": {\"mode\": \"...\", \"note\": \"...\"},\n"
        "  \"executive_summary\": \"...\",\n"
        "  \"real_why\": \"...\",\n"
        "  \"trust_gap\": \"...\",\n"
        "  \"key_objections\": [\"...\"],\n"
        "  \"proof_needed\": [\"...\"],\n"
        "  \"risk_flags\": [\"...\"],\n"
        "  \"actionable_fixes\": [\"...\"],\n"
        "  \"rewrite\": {\n"
        "    \"headline\": \"...\",\n"
        "    \"body\": \"...\"\n"
        "  }\n"
        "}"
    )


def format_rewrite_as_text(mj: Dict[str, Any]) -> str:
    if not isinstance(mj, dict):
        return ""

    copy_type = mj.get("copy_type") or st.session_state.get("copy_type") or "Other"
    rw = _ensure_dict(mj.get("rewrite"))

    if copy_type == "Headline":
        heads = _ensure_list(rw.get("headlines"))
        if heads:
            # Use the first headline as the active creative, but keep the rest visible in moderator panel.
            return str(heads[0]).strip()
        head = rw.get("headline")
        return str(head).strip() if head else ""

    if copy_type == "Email":
        subject = first_present(rw.get("subject"), rw.get("Subject"))
        preheader = first_present(rw.get("preheader"), rw.get("pre_header"), rw.get("preview"))
        body = first_present(rw.get("body"), rw.get("email_body"))
        cta = first_present(rw.get("cta"), rw.get("call_to_action"))
        ps = first_present(rw.get("ps"), rw.get("P.S."), rw.get("p.s."))

        parts: List[str] = []
        if subject:
            parts.append(f"Subject: {str(subject).strip()}")
        if preheader:
            parts.append(f"Preheader: {str(preheader).strip()}")
        if body:
            parts.append("\n" + str(body).strip())
        if cta:
            parts.append("\nCTA: " + str(cta).strip())
        if ps:
            parts.append("\nP.S. " + str(ps).strip())
        return "\n".join([p for p in parts if p]).strip()

    if copy_type == "Sales Page":
        hero_h1 = first_present(rw.get("hero_headline"), rw.get("h1"))
        hero_sub = first_present(rw.get("hero_subheadline"), rw.get("subheadline"), rw.get("subhead"))
        bullets = _ensure_list(rw.get("bullets"))
        proof = first_present(rw.get("proof_block"), rw.get("proof_section"), rw.get("proof"))
        offer_stack = _ensure_list(rw.get("offer_stack"))
        cta_button = first_present(rw.get("cta_button"), rw.get("button"))
        cta_line = first_present(rw.get("cta_line"), rw.get("cta"))

        out: List[str] = []
        if hero_h1:
            out.append(str(hero_h1).strip())
        if hero_sub:
            out.append(str(hero_sub).strip())

        if bullets:
            out.append("\nKey bullets:")
            for b in bullets[:8]:
                out.append(f"- {str(b).strip()}")

        if proof:
            out.append("\nProof block:")
            out.append(str(proof).strip())

        if offer_stack:
            out.append("\nOffer stack:")
            for o in offer_stack[:10]:
                out.append(f"- {str(o).strip()}")

        if cta_button or cta_line:
            out.append("\nCTA:")
            if cta_button:
                out.append(f"Button: {str(cta_button).strip()}")
            if cta_line:
                out.append(str(cta_line).strip())

        return "\n".join(out).strip()

    # Other/mixed
    subject = rw.get("subject")
    body = rw.get("body")
    headline = rw.get("headline")
    text = rw.get("text")

    if subject or body:
        parts: List[str] = []
        if subject:
            parts.append(f"Subject: {str(subject).strip()}")
        if body:
            parts.append(str(body).strip())
        return "\n\n".join([p for p in parts if p]).strip()

    if headline and text:
        return f"{str(headline).strip()}\n\n{str(text).strip()}".strip()

    if text:
        return str(text).strip()

    return ""


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    # key: persona_uid -> list[(q, a)]
    st.session_state.chat_history = {}
if "debate_history" not in st.session_state:
    # list[{name, uid, text}]
    st.session_state.debate_history = []
if "marketing_topic" not in st.session_state:
    st.session_state.marketing_topic = "Subject: 3 AI Stocks better than Nvidia. Urgent Buy Alert!"
if "moderator_raw" not in st.session_state:
    st.session_state.moderator_raw = ""
if "moderator_json" not in st.session_state:
    st.session_state.moderator_json = None
if "suggested_rewrite" not in st.session_state:
    st.session_state.suggested_rewrite = ""
if "campaign_assets" not in st.session_state:
    st.session_state.campaign_assets = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "selected_persona_uid" not in st.session_state:
    st.session_state.selected_persona_uid = None

# New: copy format + scope controls
if "copy_type" not in st.session_state:
    st.session_state.copy_type = "Email"
if "scope_mode" not in st.session_state:
    st.session_state.scope_mode = "Full text"
if "scope_words" not in st.session_state:
    st.session_state.scope_words = 350
if "custom_excerpt" not in st.session_state:
    st.session_state.custom_excerpt = ""
if "strip_boilerplate" not in st.session_state:
    st.session_state.strip_boilerplate = True
if "use_brief" not in st.session_state:
    st.session_state.use_brief = True
if "brief_cache" not in st.session_state:
    st.session_state.brief_cache = {}
if "creative_brief_raw" not in st.session_state:
    st.session_state.creative_brief_raw = ""
if "creative_brief_json" not in st.session_state:
    st.session_state.creative_brief_json = None
if "creative_excerpt" not in st.session_state:
    st.session_state.creative_excerpt = ""
if "creative_scope_note" not in st.session_state:
    st.session_state.creative_scope_note = ""
if "creative_processed" not in st.session_state:
    st.session_state.creative_processed = ""


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent


def find_personas_file() -> Optional[Path]:
    """Find personas JSON in a robust way.

    Priority:
    1) personas.json in same folder as this app
    2) personas.json in current working directory
    3) any personas*.json in same folder
    4) any personas*.json in cwd
    """
    candidates = [APP_DIR / "personas.json", Path.cwd() / "personas.json"]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    for base in (APP_DIR, Path.cwd()):
        globs = sorted(base.glob("personas*.json"))
        for p in globs:
            if p.exists() and p.is_file():
                return p

    return None


def _patch_core(core: Dict[str, Any]) -> Dict[str, Any]:
    core = dict(core or {})

    core.setdefault("future_confidence", 3)
    core.setdefault("family_support_received", False)
    core.setdefault("ideal_salary_for_comfort", 120_000)
    core.setdefault("budget_adjustments_6m", [])
    core.setdefault("super_engagement", "Unknown")
    core.setdefault("property_via_super_interest", "No")

    core.setdefault("income", 80_000)
    core.setdefault("goals", [])
    core.setdefault("values", [])
    core.setdefault("personality_traits", [])
    core.setdefault("concerns", [])
    core.setdefault("decision_making", "Unknown")

    bt = _ensure_dict(core.get("behavioural_traits"))
    bt.setdefault("risk_tolerance", "Moderate")
    bt.setdefault("investment_experience", "Unknown")
    bt.setdefault("information_sources", [])
    bt.setdefault("preferred_channels", [])
    core["behavioural_traits"] = bt

    cc = _ensure_dict(core.get("content_consumption"))
    cc.setdefault("preferred_formats", [])
    cc.setdefault("preferred_channels", [])
    cc.setdefault("additional_notes", "")
    core["content_consumption"] = cc

    core.setdefault("suggestions", [])
    return core


def _convert_old_schema(old: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the original {personas: [{segment, male, female}, ...]} format
    into the new schema used by this app.
    """
    groups = _ensure_list(old.get("personas"))

    default_summaries = {
        "Next Generation Investors (18-24 years)": "Tech-native, socially-conscious starters focused on building asset bases early.",
        "Emerging Wealth Builders (25-34 years)": "Balancing house deposits, careers and investing; optimistic but wage-squeezed.",
        "Established Accumulators (35-49 years)": "Juggling family, mortgages and wealth growth; value efficiency and advice.",
        "Pre-Retirees (50-64 years)": "Capital-preservers planning retirement income; keen super watchers.",
        "Retirees (65+ years)": "Stability-seekers prioritising income and low volatility.",
    }

    segments: List[Dict[str, Any]] = []

    for g in groups:
        label = g.get("segment", "Unknown")
        seg_id = slugify(label)

        label_norm = normalize_dashes(label)
        # also normalize en-dash ranges to hyphen range when matching defaults
        label_norm = label_norm.replace("\u2013", "-")
        summary = default_summaries.get(label_norm, "")

        people: List[Dict[str, Any]] = []
        for gender in ("male", "female"):
            if gender not in g:
                continue
            p = dict(g[gender] or {})

            # normalize spelling for enrichment if present
            if "behavioral_enrichment" in p and "behavioural_enrichment" not in p:
                p["behavioural_enrichment"] = p.pop("behavioral_enrichment")

            core = {
                k: v
                for k, v in p.items()
                if k
                not in {
                    "scenarios",
                    "peer_influence",
                    "risk_tolerance_differences",
                    "behavioural_enrichment",
                    "behavioral_enrichment",
                }
            }
            ext = {
                "behavioural_enrichment": p.get("behavioural_enrichment", {}),
                "risk_tolerance_differences": p.get("risk_tolerance_differences", ""),
                "scenarios": p.get("scenarios", {}),
                "peer_influence": p.get("peer_influence", {}),
            }

            core = _patch_core(core)

            people.append(
                {
                    "id": slugify(p.get("name", f"{gender}_{seg_id}")),
                    "gender": gender,
                    "core": core,
                    "extended": ext,
                }
            )

        segments.append({"id": seg_id, "label": label, "summary": summary, "personas": people})

    return {"schema_version": "1.0", "segments": segments}


@st.cache_data
def load_personas() -> Tuple[Optional[Path], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    path = find_personas_file()
    if path is None:
        return None, {"segments": []}, [], []

    raw = json.loads(path.read_text(encoding="utf-8"))

    if "segments" not in raw and "personas" in raw:
        raw = _convert_old_schema(raw)

    segments = _ensure_list(raw.get("segments"))

    # Flatten for UI
    flat: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = seg.get("id") or slugify(seg.get("label", ""))
        seg_label = seg.get("label", "Unknown")
        seg_summary = seg.get("summary", "")
        for persona in _ensure_list(seg.get("personas")):
            pid = persona.get("id") or slugify(_ensure_dict(persona.get("core")).get("name", ""))
            uid = f"{seg_id}:{pid}"

            core = _patch_core(_ensure_dict(persona.get("core")))
            ext = _ensure_dict(persona.get("extended"))

            flat.append(
                {
                    "uid": uid,
                    "segment_id": seg_id,
                    "segment_label": seg_label,
                    "segment_summary": seg_summary,
                    "persona_id": pid,
                    "gender": persona.get("gender", "unknown"),
                    "core": core,
                    "extended": ext,
                }
            )

    return path, raw, segments, flat


personas_path, personas_raw, segments_raw, all_personas_flat = load_personas()


# -----------------------------------------------------------------------------
# AI CLIENTS
# -----------------------------------------------------------------------------

def _get_secret_or_env(key: str) -> Optional[str]:
    try:
        if key in st.secrets:
            v = st.secrets.get(key)
            if v:
                return str(v)
    except Exception:
        pass
    return os.getenv(key)


OPENAI_API_KEY = _get_secret_or_env("OPENAI_API_KEY")
GOOGLE_API_KEY = _get_secret_or_env("GOOGLE_API_KEY")


openai_ready = bool(OPENAI_API_KEY) and OpenAI is not None
if openai_ready:
    try:
        client_openai = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client_openai = None
        openai_ready = False
else:
    client_openai = None


gemini_ready = bool(GOOGLE_API_KEY) and genai is not None
if gemini_ready:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        gemini_ready = False


def query_openai(messages: List[Dict[str, str]], model: str = "gpt-4o", temperature: float = 0.7) -> str:
    if not openai_ready or client_openai is None:
        return "Error: OpenAI client not configured (missing OPENAI_API_KEY or openai package)."
    try:
        completion = client_openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def query_gemini(prompt: str, model_name: str = "gemini-3-flash-preview") -> str:
    if not gemini_ready or genai is None:
        # Fallback to OpenAI if possible
        return "Gemini Error (not configured).\n\nFallback Analysis:\n" + query_openai(
            [{"role": "user", "content": prompt}],
            model=st.session_state.get("openai_model", "gpt-4o"),
            temperature=float(st.session_state.get("openai_temperature", 0.7)),
        )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip() or ""
    except Exception as e:
        return f"Gemini Error ({str(e)}).\n\nFallback Analysis:\n" + query_openai(
            [{"role": "user", "content": prompt}],
            model=st.session_state.get("openai_model", "gpt-4o"),
            temperature=float(st.session_state.get("openai_temperature", 0.7)),
        )


def build_persona_system_prompt(core: Dict[str, Any]) -> str:
    name = core.get("name", "Unknown")
    age = core.get("age", "")
    location = core.get("location", "")
    occupation = core.get("occupation", "")
    income = core.get("income", "")
    narrative = core.get("narrative", "")

    values = ", ".join(_ensure_list(core.get("values"))[:5])
    goals = "; ".join(_ensure_list(core.get("goals"))[:4])
    concerns = "; ".join(_ensure_list(core.get("concerns"))[:4])
    decision_making = core.get("decision_making", "")

    bt = _ensure_dict(core.get("behavioural_traits"))
    risk = bt.get("risk_tolerance", "Unknown")
    exp = bt.get("investment_experience", "Unknown")

    info_sources = ", ".join(_ensure_list(bt.get("information_sources"))[:6])
    preferred_channels = ", ".join(_ensure_list(bt.get("preferred_channels"))[:6])

    cc = _ensure_dict(core.get("content_consumption"))
    formats = ", ".join(_ensure_list(cc.get("preferred_formats"))[:6])

    return (
        f"You are {name}, a {age}-year-old {occupation} based in {location}. "
        f"Income: ${income}.\n"
        f"Bio: {narrative}\n"
        f"Values: {values}\n"
        f"Goals: {goals}\n"
        f"Concerns: {concerns}\n"
        f"Decision style: {decision_making}\n"
        f"Investing: {exp}. Risk tolerance: {risk}.\n"
        f"Information sources: {info_sources}\n"
        f"Preferred channels: {preferred_channels}\n"
        f"Preferred formats: {formats}\n\n"
        "Rules: Respond in first person, in-character, and grounded in your constraints. "
        "Be specific and concrete. Keep answers under ~120 words unless the user asks for depth."
    )


def extract_creative_brief(text_for_brief: str, copy_type: str, truncation_note: str) -> Tuple[str, Optional[dict]]:
    """Return (raw_model_output, parsed_json_or_none). Uses session cache by hash."""
    cache_key = sha256_text(f"{copy_type}||{truncation_note}||{text_for_brief}")
    cache = st.session_state.get("brief_cache") or {}

    if cache_key in cache:
        raw = cache[cache_key].get("raw", "")
        mj = cache[cache_key].get("json")
        return raw, mj

    prompt = build_brief_prompt(copy_type=copy_type, text=text_for_brief, truncated_note=truncation_note)
    raw = query_openai(
        [{"role": "user", "content": prompt}],
        model=st.session_state.get("brief_model", "gpt-4o-mini"),
        temperature=float(st.session_state.get("brief_temperature", 0.2)),
    )
    mj = extract_json_object(raw)

    cache[cache_key] = {"raw": raw, "json": mj}
    st.session_state.brief_cache = cache
    return raw, mj


def apply_rewrite_from_moderator() -> None:
    mj = st.session_state.get("moderator_json")
    if isinstance(mj, dict):
        rewritten_text = format_rewrite_as_text(mj)
        if rewritten_text:
            st.session_state.marketing_topic = rewritten_text.strip()
            # Keep copy type aligned if moderator specified
            if mj.get("copy_type"):
                st.session_state.copy_type = mj.get("copy_type")

            st.session_state.debate_history = []
            st.session_state.campaign_assets = None
            return

    # Fallback to raw heuristic if JSON isn't available
    raw = st.session_state.get("moderator_raw", "")
    if raw:
        st.session_state.marketing_topic = raw.strip()
        st.session_state.debate_history = []
        st.session_state.campaign_assets = None


# -----------------------------------------------------------------------------
# SIDEBAR CONFIG
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Settings")

    st.session_state.openai_model = st.selectbox(
        "OpenAI model (personas)",
        options=["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
        index=0,
    )
    st.session_state.openai_temperature = st.slider("OpenAI temperature", 0.0, 1.5, 0.7, 0.1)

    st.session_state.gemini_model = st.selectbox(
        "Gemini model (moderator)",
        options=["gemini-3-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Long Copy Controls")
    st.session_state.brief_model = st.selectbox(
        "OpenAI model (brief extraction)",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
        index=0,
    )
    st.session_state.brief_temperature = st.slider("Brief temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.markdown("### Batch Controls")
    st.session_state.max_batch = st.slider("Max personas per batch", 1, 25, 10, 1)

    st.markdown("---")
    st.markdown("### Data")
    if personas_path is None:
        st.error("No personas JSON found. Add a 'personas.json' next to this app.")
    else:
        st.caption(f"Loaded: {personas_path.name}")

    if not openai_ready:
        st.warning("OpenAI not configured (missing OPENAI_API_KEY).")
    if not gemini_ready:
        st.warning("Gemini not configured (missing GOOGLE_API_KEY).")


# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
st.title("The Foolish Synthetic Audience")

st.markdown(
    """
<div style="background:#f0f2f6;padding:20px;border-left:6px solid #485cc7;border-radius:10px;margin-bottom:25px">
    <h4 style="margin-top:0">About This Tool</h4>
    <p>This tool uses a hybrid setup: <strong>OpenAI</strong> for persona simulation and <strong>Gemini</strong> (with OpenAI fallback) for strategic analysis.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Quick claim-risk flags for the current marketing topic
risk_flags = claim_risk_flags(st.session_state.marketing_topic)
if risk_flags:
    st.warning("Claim-risk flags detected: " + ", ".join(risk_flags))


tab1, tab2 = st.tabs(["Individual Interview", "Focus Group Debate"])


# ==============================================================================
# TAB 1: INDIVIDUAL INTERVIEW
# ==============================================================================
with tab1:
    segment_options = [
        {
            "segment_id": s.get("id") or slugify(s.get("label", "")),
            "segment_label": s.get("label", "Unknown"),
            "summary": s.get("summary", ""),
        }
        for s in segments_raw
    ]

    segment_label_by_id = {x["segment_id"]: x["segment_label"] for x in segment_options}

    selected_segment_id = st.selectbox(
        "Filter by Segment",
        options=["All"] + [x["segment_id"] for x in segment_options],
        format_func=lambda sid: "All" if sid == "All" else segment_label_by_id.get(sid, sid),
    )

    if selected_segment_id == "All":
        with st.expander("Segment Cheat Sheet"):
            for s in segment_options:
                if s["summary"]:
                    st.markdown(f"**{s['segment_label']}**\n{s['summary']}\n")
    else:
        selected = next((s for s in segment_options if s["segment_id"] == selected_segment_id), None)
        if selected and selected.get("summary"):
            with st.expander("Segment Overview", expanded=True):
                st.write(selected["summary"])

    filtered_list = (
        all_personas_flat
        if selected_segment_id == "All"
        else [p for p in all_personas_flat if p["segment_id"] == selected_segment_id]
    )

    st.markdown("### Select a Persona")
    cols = st.columns(3)

    for i, entry in enumerate(filtered_list):
        core = entry["core"]
        with cols[i % 3]:
            with st.container():
                if core.get("image"):
                    st.image(core["image"], use_container_width=True)
                st.markdown(f"**{core.get('name','Unknown')}**")
                st.caption(entry["segment_label"])
                if st.button("Select", key=f"sel_{entry['uid']}"):
                    st.session_state.selected_persona_uid = entry["uid"]
                    st.rerun()

    selected_uid = st.session_state.get("selected_persona_uid")
    selected_entry = next((e for e in all_personas_flat if e["uid"] == selected_uid), None)

    if selected_entry:
        core = selected_entry["core"]
        seg_label = selected_entry["segment_label"]

        st.markdown("---")
        st.markdown(
            f"""
            <div style="background:#e3f6d8;padding:20px;border-left:6px solid #43B02A;border-radius:10px">
                <h4 style="margin-top:0">{core.get('name','Unknown')} <span style="font-weight:normal">({seg_label})</span></h4>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <p><strong>Age:</strong> {core.get('age')}</p>
                        <p><strong>Location:</strong> {core.get('location')}</p>
                        <p><strong>Occupation:</strong> {core.get('occupation')}</p>
                    </div>
                    <div>
                        <p><strong>Income:</strong> ${int(core.get('income', 0) or 0):,}</p>
                        <p><strong>Risk:</strong> {_ensure_dict(core.get('behavioural_traits')).get('risk_tolerance','Unknown')}</p>
                        <p><strong>Confidence:</strong> {core.get('future_confidence')}/5</p>
                    </div>
                </div>
                <hr style="margin:10px 0; border-top: 1px solid #ccc;">
                <p><strong>Values:</strong> {', '.join(_ensure_list(core.get('values')))}</p>
                <p><strong>Goals:</strong> {'; '.join(_ensure_list(core.get('goals')))}</p>
                <p><strong>Concerns:</strong> {'; '.join(_ensure_list(core.get('concerns')))}</p>
                <p><strong>Narrative:</strong> {core.get('narrative','')}</p>
                <p><strong>Recent Budget Cuts:</strong> {', '.join(_ensure_list(core.get('budget_adjustments_6m')) or ['None'])}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### Suggested Questions")
        suggestions = _ensure_list(core.get("suggestions"))
        if suggestions:
            cols_s = st.columns(min(len(suggestions), 3))
            for idx, s in enumerate(suggestions[:3]):
                if cols_s[idx % 3].button(f"Ask: {str(s)[:40]}...", key=f"sugg_{selected_entry['uid']}_{idx}"):
                    st.session_state.question_input = s
                    st.rerun()
        else:
            st.caption("No specific suggestions for this persona.")

        st.markdown("### Interaction")
        user_input = st.text_area("Enter your question:", value=st.session_state.question_input, key="q_input")
        ask_all = st.checkbox("Ask ALL visible personas (Batch Test)")

        if st.button("Ask Persona(s)", type="primary"):
            if not user_input:
                st.warning("Please enter a question.")
            else:
                target_list = filtered_list if ask_all else [selected_entry]

                if ask_all and len(target_list) > int(st.session_state.max_batch):
                    st.warning(
                        f"Batch capped at {st.session_state.max_batch} personas (selected segment contains {len(target_list)})."
                    )
                    target_list = target_list[: int(st.session_state.max_batch)]

                with st.spinner(f"Interviewing {len(target_list)} persona(s)..."):
                    for target in target_list:
                        tp_core = target["core"]
                        persona_uid = target["uid"]

                        sys_msg = build_persona_system_prompt(tp_core)

                        hist = st.session_state.chat_history.get(persona_uid, [])
                        messages = [{"role": "system", "content": sys_msg}]

                        for q, a in hist[-3:]:
                            messages.append({"role": "user", "content": q})
                            messages.append({"role": "assistant", "content": a})
                        messages.append({"role": "user", "content": user_input})

                        ans = query_openai(
                            messages,
                            model=st.session_state.openai_model,
                            temperature=float(st.session_state.openai_temperature),
                        )
                        st.session_state.chat_history.setdefault(persona_uid, []).append((user_input, ans))

                st.success("Responses received!")
                st.session_state.question_input = ""
                st.rerun()

        if ask_all:
            st.markdown("#### Batch Results")
            for target in filtered_list[: int(st.session_state.max_batch)]:
                persona_uid = target["uid"]
                if persona_uid in st.session_state.chat_history and st.session_state.chat_history[persona_uid]:
                    last_q, last_a = st.session_state.chat_history[persona_uid][-1]
                    if last_q == user_input:
                        st.markdown(f"**{target['core'].get('name','Unknown')}:** {last_a}")
                        st.divider()
        else:
            persona_uid = selected_entry["uid"]
            if persona_uid in st.session_state.chat_history:
                st.markdown("#### Conversation History")
                for q, a in reversed(st.session_state.chat_history[persona_uid]):
                    st.markdown(
                        f"<div class='chat-bubble user-bubble'><strong>You:</strong> {q}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='chat-bubble bot-bubble'><strong>{core.get('name','Persona')}:</strong> {a}</div>",
                        unsafe_allow_html=True,
                    )


# ==============================================================================
# TAB 2: FOCUS GROUP DEBATE
# ==============================================================================
with tab2:
    st.header("Marketing Focus Group")
    st.markdown("Pit two investors against each other to stress-test your creative.")

    persona_options = {p["uid"]: p for p in all_personas_flat}
    persona_labels = {
        uid: f"{p['core'].get('name','Unknown')} ({p['segment_label']})" for uid, p in persona_options.items()
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        p1_uid = st.selectbox(
            "Participant 1 (Believer)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=0 if persona_options else 0,
        )
    with c2:
        p2_uid = st.selectbox(
            "Participant 2 (Skeptic)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=1 if len(persona_options) > 1 else 0,
        )
    with c3:
        st.caption("Mode: Adversarial stress test")

    st.session_state.copy_type = st.selectbox(
        "Copy type",
        options=["Headline", "Email", "Sales Page", "Other"],
        index=["Headline", "Email", "Sales Page", "Other"].index(st.session_state.copy_type)
        if st.session_state.copy_type in ["Headline", "Email", "Sales Page", "Other"]
        else 1,
    )

    with st.expander("Long copy settings (recommended for sales pages)", expanded=False):
        st.session_state.strip_boilerplate = st.checkbox(
            "Strip common email footer/boilerplate",
            value=bool(st.session_state.strip_boilerplate),
            help="Removes common unsubscribe/privacy/footer blocks to reduce noise.",
        )

        st.session_state.scope_mode = st.radio(
            "Analysis scope for participants",
            options=["Full text", "First N words", "Custom excerpt"],
            index=["Full text", "First N words", "Custom excerpt"].index(st.session_state.scope_mode)
            if st.session_state.scope_mode in ["Full text", "First N words", "Custom excerpt"]
            else 0,
            help="Controls what the Believer/Skeptic are shown. Moderator can still see a brief.",
        )

        if st.session_state.scope_mode == "First N words":
            st.session_state.scope_words = st.slider("N words", 100, 2500, int(st.session_state.scope_words), 50)
        elif st.session_state.scope_mode == "Custom excerpt":
            st.session_state.custom_excerpt = st.text_area(
                "Paste an excerpt to analyse (used instead of full text)",
                value=st.session_state.custom_excerpt,
                height=140,
            )

        st.session_state.use_brief = st.checkbox(
            "Auto-extract a structured brief",
            value=bool(st.session_state.use_brief),
            help="Strongly recommended for long sales pages to keep prompts stable and outputs structured.",
        )

    height = 120 if st.session_state.copy_type == "Headline" else 220
    if st.session_state.copy_type == "Sales Page":
        height = 360

    marketing_topic = st.text_area("Paste creative", key="marketing_topic", height=height)

    # Live stats
    raw_text = marketing_topic or ""
    processed_text = raw_text
    if st.session_state.strip_boilerplate and st.session_state.copy_type == "Email":
        processed_text = strip_common_email_footer(processed_text)

    stats_words = word_count(processed_text)
    stats_tokens = estimate_tokens(processed_text)
    st.caption(f"Input size: {stats_words} words (approx {stats_tokens} tokens)")

    # Auto-suggest safe defaults for long copy
    long_copy = stats_words >= 900 or stats_tokens >= 2400
    if long_copy:
        st.info(
            "Long copy detected. Recommended: enable brief extraction and show participants a shorter excerpt (First N words)."
        )
    PARTICIPANT_MAX_WORDS = 1500

    def compute_excerpt(text: str) -> Tuple[str, str]:
        mode = st.session_state.scope_mode

        if mode == "Custom excerpt":
            custom = (st.session_state.custom_excerpt or "").strip()
            if custom:
                if word_count(custom) > PARTICIPANT_MAX_WORDS:
                    return (
                        truncate_to_words(custom, PARTICIPANT_MAX_WORDS),
                        f"Custom excerpt (truncated to first {PARTICIPANT_MAX_WORDS} words)",
                    )
                return custom, "Custom excerpt"

            # fall back to full text
            if word_count(text) > PARTICIPANT_MAX_WORDS:
                return (
                    truncate_to_words(text, PARTICIPANT_MAX_WORDS),
                    f"Full text (truncated to first {PARTICIPANT_MAX_WORDS} words)",
                )
            return text.strip(), "Full text"

        if mode == "First N words":
            n = min(int(st.session_state.scope_words), PARTICIPANT_MAX_WORDS)
            return truncate_to_words(text, n), f"First {n} words"

        # Full text
        if word_count(text) > PARTICIPANT_MAX_WORDS:
            return (
                truncate_to_words(text, PARTICIPANT_MAX_WORDS),
                f"Full text (truncated to first {PARTICIPANT_MAX_WORDS} words)",
            )
        return text.strip(), "Full text"

    excerpt_text, scope_note = compute_excerpt(processed_text)

    # Decide what to feed into the brief extractor
    BRIEF_MAX_WORDS = 2500
    brief_input_text = truncate_to_words(processed_text, BRIEF_MAX_WORDS)
    brief_trunc_note = "Not truncated"
    if word_count(processed_text) > BRIEF_MAX_WORDS:
        brief_trunc_note = f"Brief built from first {BRIEF_MAX_WORDS} words (input was longer)."

    if st.button("Start Focus Group", type="primary"):
        st.session_state.debate_history = []
        st.session_state.moderator_raw = ""
        st.session_state.moderator_json = None
        st.session_state.suggested_rewrite = ""
        st.session_state.campaign_assets = None

        st.session_state.creative_processed = processed_text
        st.session_state.creative_excerpt = excerpt_text
        st.session_state.creative_scope_note = scope_note

        p_a = persona_options.get(p1_uid)
        p_b = persona_options.get(p2_uid)

        if not p_a or not p_b:
            st.error("Please select two participants.")
            st.stop()

        # Extract brief if enabled or if long copy
        st.session_state.creative_brief_raw = ""
        st.session_state.creative_brief_json = None

        should_extract_brief = bool(st.session_state.use_brief) or long_copy
        brief_summary = ""

        if should_extract_brief:
            with st.spinner("Extracting structured brief..."):
                brief_raw, brief_json = extract_creative_brief(
                    text_for_brief=brief_input_text,
                    copy_type=st.session_state.copy_type,
                    truncation_note=brief_trunc_note,
                )
            st.session_state.creative_brief_raw = brief_raw
            st.session_state.creative_brief_json = brief_json
            if isinstance(brief_json, dict):
                brief_summary = format_brief_summary(brief_json)

        # Base instruction shared across roles
        base_instruction = (
            "IMPORTANT: This is a simulation for marketing research. "
            "You are roleplaying a specific persona. Do NOT sound like a generic AI. "
            "Use specific vocabulary, worldview, and constraints from the persona. "
            "Do not give financial advice; focus on reactions to marketing and credibility."
        )

        def role_prompt(entry: Dict[str, Any], stance: str) -> str:
            core = entry["core"]
            bt = _ensure_dict(core.get("behavioural_traits"))

            values = ", ".join(_ensure_list(core.get("values"))[:5])
            goals = "; ".join(_ensure_list(core.get("goals"))[:4])
            concerns = "; ".join(_ensure_list(core.get("concerns"))[:4])

            style_block = ""
            if st.session_state.copy_type == "Headline":
                style_block = (
                    "Respond in this structure:\n"
                    "- Hook: ...\n"
                    "- What I infer: ...\n"
                    "- What I'd do next: ...\n"
                )
            elif st.session_state.copy_type == "Email":
                style_block = (
                    "Respond in this structure:\n"
                    "- Open or ignore (and why): ...\n"
                    "- Trust/credibility reaction: ...\n"
                    "- Biggest question: ...\n"
                    "- One change that improves it: ...\n"
                )
            elif st.session_state.copy_type == "Sales Page":
                style_block = (
                    "Respond in this structure:\n"
                    "- Above-the-fold reaction: ...\n"
                    "- Offer clarity: ...\n"
                    "- Proof/trust: ...\n"
                    "- Objection: ...\n"
                    "- One fix: ...\n"
                )
            else:
                style_block = (
                    "Respond in this structure:\n"
                    "- What works: ...\n"
                    "- What worries me: ...\n"
                    "- Proof I'd need: ...\n"
                    "- One fix: ...\n"
                )

            return (
                f"ROLE: You are {core.get('name')}, a {core.get('age')}-year-old {core.get('occupation')}. "
                f"BIO: {core.get('narrative','')}\n"
                f"VALUES: {values}\n"
                f"GOALS: {goals}\n"
                f"CONCERNS: {concerns}\n"
                f"RISK TOLERANCE: {bt.get('risk_tolerance','Unknown')}\n\n"
                f"CONTEXT: In this focus group, you represent '{stance}'.\n"
                + (
                    "You WANT the marketing message to be true. You focus on upside, possibility, and emotional appeal. "
                    "You defend the message against skepticism, but you still sound like a real person.\n"
                    if stance == "Believer"
                    else "You are critical of marketing hype and naturally risk-aware. You look for missing details, credibility gaps, and implied claims. "
                    "You call out anything that sounds too good to be true.\n"
                )
                + "\n" + style_block
            )

        role_a = role_prompt(p_a, "Believer")
        role_b = role_prompt(p_b, "Skeptic")

        # Compose the creative context shown to participants
        creative_context = (
            f"COPY TYPE: {st.session_state.copy_type}\n"
            f"SCOPE SHOWN TO YOU: {scope_note}\n\n"
            "CREATIVE (what you saw):\n"
            f"{excerpt_text}\n"
        )
        if brief_summary:
            creative_context += "\nCREATIVE BRIEF (for context):\n" + brief_summary + "\n"

        st.markdown("---")
        st.markdown("### What the personas saw")
        with st.expander("Preview excerpt used in the debate", expanded=False):
            st.code(excerpt_text)
        if isinstance(st.session_state.creative_brief_json, dict):
            with st.expander("Preview extracted brief JSON", expanded=False):
                st.json(st.session_state.creative_brief_json)
        elif should_extract_brief and st.session_state.creative_brief_raw:
            with st.expander("Brief extraction output (raw)", expanded=False):
                st.code(st.session_state.creative_brief_raw)

        st.markdown("---")
        st.markdown("### Debate")

        # 1) Believer
        msg_a = query_openai(
            [
                {"role": "system", "content": base_instruction + "\n\n" + role_a},
                {
                    "role": "user",
                    "content": (
                        "React to the creative. Be specific about what pulls you in, and what you want to believe.\n\n"
                        + creative_context
                    ),
                },
            ],
            model=st.session_state.openai_model,
            temperature=float(st.session_state.openai_temperature),
        )
        st.session_state.debate_history.append({"name": p_a["core"].get("name"), "uid": p_a["uid"], "text": msg_a})
        st.markdown(f"**{p_a['core'].get('name')} (Believer)**: {msg_a}")
        time.sleep(0.3)

        # 2) Skeptic
        msg_b = query_openai(
            [
                {"role": "system", "content": base_instruction + "\n\n" + role_b},
                {
                    "role": "user",
                    "content": (
                        f"The Believer ({p_a['core'].get('name')}) just said:\n{msg_a}\n\n"
                        "Give them a reality check. Be specific about what you would need to see to trust it.\n\n"
                        + creative_context
                    ),
                },
            ],
            model=st.session_state.openai_model,
            temperature=float(st.session_state.openai_temperature),
        )
        st.session_state.debate_history.append({"name": p_b["core"].get("name"), "uid": p_b["uid"], "text": msg_b})
        st.markdown(f"**{p_b['core'].get('name')} (Skeptic)**: {msg_b}")
        time.sleep(0.3)

        # 3) Believer retort
        msg_a2 = query_openai(
            [
                {"role": "system", "content": base_instruction + "\n\n" + role_a},
                {
                    "role": "user",
                    "content": (
                        f"You just got critiqued. The Skeptic ({p_b['core'].get('name')}) said:\n{msg_b}\n\n"
                        "Respond as yourself. Acknowledge what feels fair, and restate what still excites you.\n\n"
                        "Keep it grounded in what you saw."
                    ),
                },
            ],
            model=st.session_state.openai_model,
            temperature=float(st.session_state.openai_temperature),
        )
        st.session_state.debate_history.append({"name": p_a["core"].get("name"), "uid": p_a["uid"], "text": msg_a2})
        st.markdown(f"**{p_a['core'].get('name')} (Believer)**: {msg_a2}")

        # 4) Moderator
        st.markdown("---")
        st.subheader("Strategic Analysis (Moderator)")

        transcript = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])

        # Moderator sees: transcript + brief (if present) + the same excerpt + (optional) the full copy
        full_copy_included = processed_text
        full_copy_note = "Full copy included."

        # Hard guardrail for extremely long copy to reduce failures.
        MOD_MAX_WORDS = 4500
        if word_count(processed_text) > MOD_MAX_WORDS:
            full_copy_included = truncate_to_words(processed_text, MOD_MAX_WORDS)
            full_copy_note = f"Full copy was very long. Moderator only received first {MOD_MAX_WORDS} words."

        brief_json_block = ""
        if isinstance(st.session_state.creative_brief_json, dict):
            brief_json_block = json.dumps(st.session_state.creative_brief_json, ensure_ascii=False)

        mod_prompt = f"""
You are a legendary Direct Response Copywriter (Motley Fool style) acting as a focus-group moderator.

GOAL:
Produce structured, actionable critique and a format-appropriate rewrite.

COPY TYPE: {st.session_state.copy_type}
SCOPE SHOWN TO PARTICIPANTS: {scope_note}
FULL COPY NOTE: {full_copy_note}

FOCUS GROUP TRANSCRIPT:
{transcript}

CREATIVE EXCERPT SHOWN TO PARTICIPANTS:
{excerpt_text}

EXTRACTED CREATIVE BRIEF (JSON, if available):
{brief_json_block}

FULL CREATIVE (as available):
{full_copy_included}

OUTPUT:
Return ONLY a single JSON object (no markdown, no commentary) with EXACTLY this shape (keys may be empty strings or empty lists, but must exist):

{moderator_schema_example(st.session_state.copy_type)}

CONSTRAINTS:
- Be specific and diagnostic.
- Avoid guarantees or performance promises.
- If the excerpt is missing crucial info, say exactly what is missing.
- Make the rewrite fit the declared copy type.
- Keep rewrite lengths reasonable:
  - Headline: 8-12 headline options.
  - Email: subject+preheader + body (150-250 words max) + CTA + optional PS.
  - Sales Page: rewrite key blocks (hero headline/subhead, bullets, proof block, offer stack, CTA).
""".strip()

        with st.spinner("Moderator is analysing..."):
            mod_raw = query_gemini(mod_prompt, model_name=st.session_state.gemini_model)

        st.session_state.moderator_raw = mod_raw
        mj = extract_json_object(mod_raw)
        st.session_state.moderator_json = mj

        if mj is None:
            st.info(mod_raw)
            st.warning("Moderator output was not valid JSON; displayed raw text.")
        else:
            st.success("Moderator analysis ready.")

            # Display structured fields if present
            st.markdown(f"**Executive summary:** {mj.get('executive_summary','')}")
            st.markdown(f"**Real why:** {mj.get('real_why','')}")
            st.markdown(f"**Trust gap:** {mj.get('trust_gap','')}")

            if mj.get("key_objections"):
                st.markdown("**Key objections:**")
                for x in _ensure_list(mj.get("key_objections")):
                    st.markdown(f"- {x}")

            if mj.get("proof_needed"):
                st.markdown("**Proof needed:**")
                for x in _ensure_list(mj.get("proof_needed")):
                    st.markdown(f"- {x}")

            if mj.get("confusing_phrases"):
                st.markdown("**Confusing phrases:**")
                for x in _ensure_list(mj.get("confusing_phrases")):
                    st.markdown(f"- {x}")

            if mj.get("risk_flags"):
                st.markdown("**Risk flags:**")
                for x in _ensure_list(mj.get("risk_flags")):
                    st.markdown(f"- {x}")

            if mj.get("actionable_fixes"):
                st.markdown("**Actionable fixes:**")
                for x in _ensure_list(mj.get("actionable_fixes")):
                    st.markdown(f"- {x}")

            if mj.get("section_feedback"):
                with st.expander("Section-by-section feedback", expanded=False):
                    for sf in _ensure_list(mj.get("section_feedback")):
                        if not isinstance(sf, dict):
                            continue
                        st.markdown(f"**{sf.get('section','Section')}**")
                        ww = sf.get("what_works", "")
                        wf = sf.get("what_fails", "")
                        fx = sf.get("fix", "")
                        if ww:
                            st.markdown(f"- What works: {ww}")
                        if wf:
                            st.markdown(f"- What fails: {wf}")
                        if fx:
                            st.markdown(f"- Fix: {fx}")
                        st.markdown("---")

            st.markdown("---")
            st.markdown("### Rewrite")

            rw = _ensure_dict(mj.get("rewrite"))
            ct = mj.get("copy_type") or st.session_state.copy_type

            if ct == "Headline":
                heads = _ensure_list(rw.get("headlines"))
                if heads:
                    for i, h in enumerate(heads[:12], start=1):
                        st.markdown(f"{i}. {h}")
                supp = _ensure_list(rw.get("supporting_lines"))
                if supp:
                    st.markdown("**Supporting lines:**")
                    for s in supp[:6]:
                        st.markdown(f"- {s}")
                if rw.get("angle_notes"):
                    st.markdown(f"**Angle notes:** {rw.get('angle_notes')}")

            elif ct == "Email":
                if rw.get("subject"):
                    st.markdown(f"**Subject:** {rw.get('subject')}")
                if rw.get("preheader"):
                    st.markdown(f"**Preheader:** {rw.get('preheader')}")
                if rw.get("body"):
                    st.markdown("**Body:**")
                    st.write(str(rw.get("body")).strip())
                if rw.get("cta"):
                    st.markdown(f"**CTA:** {rw.get('cta')}")
                if rw.get("ps"):
                    st.markdown(f"**P.S.:** {rw.get('ps')}")

            elif ct == "Sales Page":
                if rw.get("hero_headline"):
                    st.markdown(f"**Hero headline:** {rw.get('hero_headline')}")
                if rw.get("hero_subheadline"):
                    st.markdown(f"**Hero subheadline:** {rw.get('hero_subheadline')}")
                bullets = _ensure_list(rw.get("bullets"))
                if bullets:
                    st.markdown("**Bullets:**")
                    for b in bullets[:10]:
                        st.markdown(f"- {b}")
                if rw.get("proof_block"):
                    st.markdown("**Proof block:**")
                    st.write(str(rw.get("proof_block")).strip())
                offer_stack = _ensure_list(rw.get("offer_stack"))
                if offer_stack:
                    st.markdown("**Offer stack:**")
                    for o in offer_stack[:10]:
                        st.markdown(f"- {o}")
                if rw.get("cta_button") or rw.get("cta_line"):
                    st.markdown("**CTA:**")
                    if rw.get("cta_button"):
                        st.markdown(f"- Button: {rw.get('cta_button')}")
                    if rw.get("cta_line"):
                        st.markdown(f"- Line: {rw.get('cta_line')}")

            else:
                if rw.get("headline"):
                    st.markdown(f"**Headline:** {rw.get('headline')}")
                if rw.get("body"):
                    st.markdown("**Body:**")
                    st.write(str(rw.get("body")).strip())

            st.session_state.suggested_rewrite = json.dumps(mj, ensure_ascii=False, indent=2)

    # Feedback loop & campaign generator
    if st.session_state.debate_history and (st.session_state.moderator_raw or st.session_state.moderator_json):
        st.markdown("---")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("### Iterate")
            st.caption("Re-run debate with the rewrite applied.")
            st.button("Test Rewrite", on_click=apply_rewrite_from_moderator)

        with col_b:
            st.markdown("### Production")
            st.caption("Turn the insight into ad assets.")

            if st.button("Generate Campaign Assets", type="secondary"):
                with st.spinner("Briefing the specialist copywriters..."):
                    insight = st.session_state.moderator_json or {"raw": st.session_state.moderator_raw}

                    campaign_prompt = f"""
You are a Full-Stack Marketing Team. Use the STRATEGIC INSIGHT below to generate campaign assets.

STRATEGIC INSIGHT (JSON):
{json.dumps(insight, ensure_ascii=False)}

TASKS:
1) GOOGLE SEARCH AD:
   - 3 headlines (<= 30 chars)
   - 2 descriptions (<= 90 chars)
   - Goal: high CTR, curiosity, credible.

2) META AD:
   - Primary text (scroll-stopper): pattern interrupt or story hook.
   - Headline (<= 5 words)

3) SALES PAGE HERO:
   - H1 headline
   - Subheadline (promise)
   - CTA button copy

Output as Markdown with headers: ### Google Ads, ### Meta Ad, ### Sales Page Hero
Avoid guarantees or performance promises.
""".strip()

                    assets = query_gemini(campaign_prompt, model_name=st.session_state.gemini_model)
                    st.session_state.campaign_assets = assets

        if st.session_state.campaign_assets:
            st.divider()
            st.subheader("Campaign Asset Pack")
            st.markdown(st.session_state.campaign_assets)

        # Export
        st.divider()
        st.subheader("Export")

        transcript_txt = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])
        st.download_button(
            "Download transcript (txt)",
            data=transcript_txt,
            file_name="focus_group_transcript.txt",
            mime="text/plain",
        )

        if isinstance(st.session_state.creative_brief_json, dict):
            st.download_button(
                "Download creative brief (json)",
                data=json.dumps(st.session_state.creative_brief_json, ensure_ascii=False, indent=2),
                file_name="creative_brief.json",
                mime="application/json",
            )

        if st.session_state.moderator_json:
            st.download_button(
                "Download moderator analysis (json)",
                data=json.dumps(st.session_state.moderator_json, ensure_ascii=False, indent=2),
                file_name="moderator_analysis.json",
                mime="application/json",
            )

        if st.session_state.campaign_assets:
            st.download_button(
                "Download campaign assets (md)",
                data=st.session_state.campaign_assets,
                file_name="campaign_assets.md",
                mime="text/markdown",
            )
