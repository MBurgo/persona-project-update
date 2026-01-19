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
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
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
        r"(?i)^\s*\u00a9\b",
        r"(?i)\bdisclaimer\b",
        r"(?i)\bthis email\b",
    ]

    for i, line in enumerate(lines):
        if any(re.search(p, line) for p in patterns):
            trimmed = "\n".join(lines[:i]).strip()
            return trimmed if trimmed else text.strip()
    return text.strip()


def claim_risk_flags(text: str) -> List[str]:
    """Very lightweight claim-risk heuristic for marketing copy.

    This is NOT a legal assessment. It is only a quick signal that the copy
    contains language patterns that often increase compliance/trust risk.
    """
    if not text:
        return []
    t = text.lower()
    patterns = {
        "Guaranteed / certainty language": [
            "guaranteed",
            "can't lose",
            "sure thing",
            "no risk",
            "risk-free",
            "100%",
        ],
        "Urgency pressure": [
            "urgent",
            "act now",
            "act fast",
            "act quickly",
            "limited time",
            "limited time only",
            "today only",
            "last chance",
            "hurry",
            "ends soon",
            "expires",
            "flash sale",
            "before it's too late",
        ],
        "Implied future performance": [
            "will double",
            "will triple",
            "will soar",
            "take off explosively",
            "absolute fortunes",
            "next nvidia",
            "can't miss",
        ],
        "Overly absolute claims": [
            "always",
            "never",
            "everyone",
            "no one",
        ],
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
    # Keep empty by default so we do not show claim-risk flags on first load.
    st.session_state.marketing_topic = ""

if "copy_type" not in st.session_state:
    st.session_state.copy_type = "Email"

if "strip_footer" not in st.session_state:
    st.session_state.strip_footer = True

if "participant_scope" not in st.session_state:
    st.session_state.participant_scope = "First N words"

if "participant_n_words" not in st.session_state:
    st.session_state.participant_n_words = 650

if "participant_custom_excerpt" not in st.session_state:
    st.session_state.participant_custom_excerpt = ""

if "moderator_max_words" not in st.session_state:
    st.session_state.moderator_max_words = 3500

if "auto_extract_brief" not in st.session_state:
    st.session_state.auto_extract_brief = True

if "creative_excerpt" not in st.session_state:
    st.session_state.creative_excerpt = ""

if "creative_moderator_text" not in st.session_state:
    st.session_state.creative_moderator_text = ""

if "creative_brief" not in st.session_state:
    st.session_state.creative_brief = None

if "creative_brief_raw" not in st.session_state:
    st.session_state.creative_brief_raw = ""

if "creative_risk_flags" not in st.session_state:
    st.session_state.creative_risk_flags = []

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
        summary = default_summaries.get(normalize_dashes(label), "")

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
                if k not in {"scenarios", "peer_influence", "risk_tolerance_differences", "behavioural_enrichment"}
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
            pid = persona.get("id") or slugify(persona.get("core", {}).get("name", ""))
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
        "Be specific and concrete. Keep answers under ~160 words unless asked for depth."
    )


def _copy_type_id(label: str) -> str:
    x = (label or "").strip().lower()
    if x.startswith("headline"):
        return "headline"
    if x.startswith("email"):
        return "email"
    if x.startswith("sales"):
        return "sales_page"
    return "other"


def extract_structured_brief(copy_type_label: str, creative_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract a structured creative brief via OpenAI.

    Returns (brief_json, raw_text).
    """
    if not openai_ready:
        return None, "OpenAI not configured."

    copy_type = _copy_type_id(copy_type_label)
    prompt = f"""
You are a senior direct-response marketing strategist.

Task: Extract a structured brief from the creative below.

Return ONLY a single JSON object that matches this schema exactly:

{{
  "copy_type": "{copy_type}",
  "primary_promise": "...",
  "offer_summary": "...",
  "audience_assumed": "...",
  "cta": "...",
  "key_claims": ["..."],
  "proof_elements": ["..."],
  "missing_proof": ["..."],
  "urgency_elements": ["..."],
  "compliance_sensitive_phrases": ["..."],
  "sections_detected": [
    {{"section": "...", "notes": "..."}}
  ]
}}

Rules:
- Use short phrases.
- Do NOT invent facts not present.
- If unclear, write "Unknown" or leave empty arrays.

CREATIVE:
{creative_text}
""".strip()

    raw = query_openai(
        [{"role": "user", "content": prompt}],
        model=st.session_state.get("openai_model", "gpt-4o"),
        temperature=0.2,
    )

    brief = extract_json_object(raw)
    return brief, raw


def moderator_schema(copy_type_label: str) -> str:
    ct = _copy_type_id(copy_type_label)

    if ct == "headline":
        return (
            "{\n"
            "  \"copy_type\": \"headline\",\n"
            "  \"executive_summary\": \"...\",\n"
            "  \"real_why\": \"...\",\n"
            "  \"trust_gap\": \"...\",\n"
            "  \"key_objections\": [\"...\"],\n"
            "  \"proof_needed\": [\"...\"],\n"
            "  \"risk_flags\": [\"...\"],\n"
            "  \"actionable_fixes\": [\"...\"],\n"
            "  \"structured_feedback\": {\n"
            "    \"what_works\": [\"...\"],\n"
            "    \"what_breaks\": [\"...\"],\n"
            "    \"credibility_gap\": \"...\"\n"
            "  },\n"
            "  \"rewrite\": {\n"
            "    \"best_pick\": \"...\",\n"
            "    \"headlines\": [\"...\"],\n"
            "    \"supporting_lines\": [\"...\"]\n"
            "  }\n"
            "}"
        )

    if ct == "sales_page":
        return (
            "{\n"
            "  \\\"copy_type\\\": \\\"sales_page\\\",\n"
            "  \\\"executive_summary\\\": \\\"...\\\",\n"
            "  \\\"real_why\\\": \\\"...\\\",\n"
            "  \\\"trust_gap\\\": \\\"...\\\",\n"
            "  \\\"key_objections\\\": [\\\"...\\\"],\n"
            "  \\\"proof_needed\\\": [\\\"...\\\"],\n"
            "  \\\"risk_flags\\\": [\\\"...\\\"],\n"
            "  \\\"actionable_fixes\\\": [\\\"...\\\"],\n"
            "  \\\"structured_feedback\\\": {\n"
            "    \\\"sections\\\": [\n"
            "      {\\\"section\\\": \\\"Hero\\\", \\\"issue\\\": \\\"...\\\", \\\"fix\\\": \\\"...\\\"}\n"
            "    ]\n"
            "  },\n"
            "  \\\"rewrite\\\": {\n"
            "    \\\"hero_headline\\\": \\\"...\\\",\n"
            "    \\\"hero_subhead\\\": \\\"...\\\",\n"
            "    \\\"bullets\\\": [\\\"...\\\"],\n"
            "    \\\"proof_block\\\": \\\"...\\\",\n"
            "    \\\"offer_stack\\\": [\\\"...\\\"],\n"
            "    \\\"cta\\\": \\\"...\\\",\n"
            "    \\\"risk_reducer\\\": \\\"...\\\"\n"
            "  }\n"
            "}"
        )

    if ct == "email":
        return (
            "{\n"
            "  \\\"copy_type\\\": \\\"email\\\",\n"
            "  \\\"executive_summary\\\": \\\"...\\\",\n"
            "  \\\"real_why\\\": \\\"...\\\",\n"
            "  \\\"trust_gap\\\": \\\"...\\\",\n"
            "  \\\"key_objections\\\": [\\\"...\\\"],\n"
            "  \\\"proof_needed\\\": [\\\"...\\\"],\n"
            "  \\\"risk_flags\\\": [\\\"...\\\"],\n"
            "  \\\"actionable_fixes\\\": [\\\"...\\\"],\n"
            "  \\\"rewrite\\\": {\n"
            "    \\\"subject\\\": \\\"...\\\",\n"
            "    \\\"preheader\\\": \\\"...\\\",\n"
            "    \\\"body\\\": \\\"...\\\",\n"
            "    \\\"cta\\\": \\\"...\\\",\n"
            "    \\\"ps\\\": \\\"...\\\"\n"
            "  }\n"
            "}"
        )

    # other
    return (
        "{\n"
        "  \\\"copy_type\\\": \\\"other\\\",\n"
        "  \\\"executive_summary\\\": \\\"...\\\",\n"
        "  \\\"real_why\\\": \\\"...\\\",\n"
        "  \\\"trust_gap\\\": \\\"...\\\",\n"
        "  \\\"key_objections\\\": [\\\"...\\\"],\n"
        "  \\\"proof_needed\\\": [\\\"...\\\"],\n"
        "  \\\"risk_flags\\\": [\\\"...\\\"],\n"
        "  \\\"actionable_fixes\\\": [\\\"...\\\"],\n"
        "  \\\"rewrite\\\": {\n"
        "    \\\"headline\\\": \\\"...\\\",\n"
        "    \\\"body\\\": \\\"...\\\",\n"
        "    \\\"cta\\\": \\\"...\\\"\n"
        "  }\n"
        "}"
    )


def build_text_from_rewrite(copy_type_label: str, rewrite: Dict[str, Any]) -> str:
    ct = _copy_type_id(copy_type_label)

    if ct == "headline":
        best = (rewrite or {}).get("best_pick")
        headlines = (rewrite or {}).get("headlines") or []
        pick = best or (headlines[0] if headlines else "")
        return str(pick or "").strip()

    if ct == "email":
        subject = (rewrite or {}).get("subject")
        preheader = (rewrite or {}).get("preheader")
        body = (rewrite or {}).get("body")
        cta = (rewrite or {}).get("cta")
        ps = (rewrite or {}).get("ps")

        parts: List[str] = []
        if subject:
            parts.append(f"Subject: {subject}")
        if preheader:
            parts.append(f"Preheader: {preheader}")
        if body:
            parts.append(str(body).strip())
        if cta:
            parts.append(f"CTA: {str(cta).strip()}")
        if ps:
            parts.append(f"P.S.: {str(ps).strip()}")
        return "\n\n".join([p for p in parts if p.strip()]).strip()

    if ct == "sales_page":
        hero_h = (rewrite or {}).get("hero_headline")
        hero_s = (rewrite or {}).get("hero_subhead")
        bullets = (rewrite or {}).get("bullets") or []
        proof = (rewrite or {}).get("proof_block")
        offer = (rewrite or {}).get("offer_stack") or []
        cta = (rewrite or {}).get("cta")
        rr = (rewrite or {}).get("risk_reducer")

        out: List[str] = []
        if hero_h:
            out.append(f"H1: {hero_h}")
        if hero_s:
            out.append(f"Subhead: {hero_s}")
        if bullets:
            out.append("Bullets:\n" + "\n".join([f"- {b}" for b in bullets if str(b).strip()]))
        if proof:
            out.append("Proof block:\n" + str(proof).strip())
        if offer:
            out.append("Offer stack:\n" + "\n".join([f"- {x}" for x in offer if str(x).strip()]))
        if cta:
            out.append("CTA: " + str(cta).strip())
        if rr:
            out.append("Risk reducer: " + str(rr).strip())
        return "\n\n".join([x for x in out if x.strip()]).strip()

    # other
    headline = (rewrite or {}).get("headline")
    body = (rewrite or {}).get("body")
    cta = (rewrite or {}).get("cta")
    parts2: List[str] = []
    if headline:
        parts2.append(str(headline).strip())
    if body:
        parts2.append(str(body).strip())
    if cta:
        parts2.append("CTA: " + str(cta).strip())
    return "\n\n".join([p for p in parts2 if p.strip()]).strip()


def apply_rewrite_from_moderator() -> None:
    mj = st.session_state.get("moderator_json")
    if not isinstance(mj, dict):
        return

    rewrite = mj.get("rewrite") or {}
    ct = mj.get("copy_type") or st.session_state.get("copy_type", "Other")

    merged = build_text_from_rewrite(str(ct), _ensure_dict(rewrite))
    if merged:
        st.session_state.marketing_topic = merged
        st.session_state.debate_history = []
        st.session_state.campaign_assets = None
        st.session_state.moderator_raw = ""
        st.session_state.moderator_json = None
        # Keep brief/excerpt settings; they will rerun.


def _prepare_creative_inputs(raw_text: str) -> Tuple[str, str, str, List[str]]:
    """Return (cleaned_full, participant_excerpt, moderator_text, risk_flags)."""
    text = (raw_text or "").strip()

    if st.session_state.get("strip_footer"):
        text = strip_common_email_footer(text)

    # Moderator text (bigger cap)
    moderator_max = int(st.session_state.get("moderator_max_words", 3500) or 3500)
    moderator_max = max(300, min(4500, moderator_max))
    moderator_text = truncate_to_words(text, moderator_max)

    # Participant excerpt
    scope = st.session_state.get("participant_scope", "First N words")
    excerpt_source = text
    if scope == "Custom excerpt":
        custom = (st.session_state.get("participant_custom_excerpt") or "").strip()
        if custom:
            excerpt_source = custom
    elif scope == "First N words":
        pass

    n_words = int(st.session_state.get("participant_n_words", 650) or 650)
    n_words = max(150, min(1500, n_words))

    if scope == "Full text":
        participant_text = truncate_to_words(excerpt_source, 1500)
    elif scope == "Custom excerpt":
        participant_text = truncate_to_words(excerpt_source, 1500)
    else:
        participant_text = truncate_to_words(excerpt_source, n_words)

    flags = claim_risk_flags(text)

    return text, participant_text, moderator_text, flags


def _debate_base_instruction() -> str:
    return (
        "IMPORTANT: This is a simulation for marketing research. "
        "You are roleplaying a specific persona. Do NOT sound like a generic AI. "
        "Use specific vocabulary, worldview, and constraints from the persona. "
        "Do not give financial advice; focus on reactions to marketing, trust, and clarity. "
        "Keep each turn under 160 words." 
    )


def _role_prompt(entry: Dict[str, Any], stance: str) -> str:
    core = entry["core"]
    bt = _ensure_dict(core.get("behavioural_traits"))

    values = ", ".join(_ensure_list(core.get("values"))[:5])
    goals = "; ".join(_ensure_list(core.get("goals"))[:4])
    concerns = "; ".join(_ensure_list(core.get("concerns"))[:4])

    stance_text = (
        "You WANT the marketing message to be true. You focus on upside, possibility, and emotional appeal. "
        "You still notice missing details, but your default posture is optimistic."
        if stance == "The Believer"
        else "You are critical of marketing hype and naturally risk-aware. You look for missing details, credibility gaps, and implied claims. "
        "You call out anything that sounds too good to be true."
    )

    return (
        f"ROLE: You are {core.get('name')}, a {core.get('age')}-year-old {core.get('occupation')}.\n"
        f"BIO: {core.get('narrative','')}\n"
        f"VALUES: {values}\n"
        f"GOALS: {goals}\n"
        f"CONCERNS: {concerns}\n"
        f"RISK TOLERANCE: {bt.get('risk_tolerance','Unknown')}\n\n"
        f"CONTEXT: In this focus group, you represent '{stance}'.\n{stance_text}"
    )


def _safe_json_dumps(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, indent=2)
    except Exception:
        return "{}"


# -----------------------------------------------------------------------------
# SIDEBAR CONFIG
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Settings")

    st.session_state.openai_model = st.selectbox(
        "OpenAI model",
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

tab1, tab2 = st.tabs(["Individual Interview", "Focus Group Debate"])


# ==============================================================================
# TAB 1: INDIVIDUAL INTERVIEW
# ==============================================================================
with tab1:
    # Segment filter
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
        # Show overview for the selected segment
        selected = next((s for s in segment_options if s["segment_id"] == selected_segment_id), None)
        if selected and selected.get("summary"):
            with st.expander("Segment Overview", expanded=True):
                st.write(selected["summary"])

    filtered_list = (
        all_personas_flat
        if selected_segment_id == "All"
        else [p for p in all_personas_flat if p["segment_id"] == selected_segment_id]
    )

    # Persona grid
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

    # Detailed profile
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

        # Suggested questions
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

        st.markdown("---")
        st.markdown("### Ask a question")
        user_input = st.text_input("Your question", value=st.session_state.get("question_input", ""), key="question_input")
        ask_all = st.checkbox("Ask ALL visible personas", value=False)

        if st.button("Send", type="primary"):
            if not openai_ready:
                st.error("OpenAI is not configured. Set OPENAI_API_KEY.")
                st.stop()

            if not user_input.strip():
                st.warning("Please enter a question.")
                st.stop()

            if ask_all:
                targets = filtered_list[: int(st.session_state.max_batch)]
                for target in targets:
                    persona_uid = target["uid"]
                    sys_prompt = build_persona_system_prompt(target["core"])
                    answer = query_openai(
                        [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_input},
                        ],
                        model=st.session_state.openai_model,
                        temperature=float(st.session_state.openai_temperature),
                    )
                    st.session_state.chat_history.setdefault(persona_uid, []).append((user_input, answer))
                st.success("Responses received!")
                st.session_state.question_input = ""
                st.rerun()
            else:
                persona_uid = selected_entry["uid"]
                sys_prompt = build_persona_system_prompt(core)
                answer = query_openai(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    model=st.session_state.openai_model,
                    temperature=float(st.session_state.openai_temperature),
                )
                st.session_state.chat_history.setdefault(persona_uid, []).append((user_input, answer))
                st.success("Response received!")
                st.session_state.question_input = ""
                st.rerun()

        # Display history
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
    persona_labels = {uid: f"{p['core'].get('name','Unknown')} ({p['segment_label']})" for uid, p in persona_options.items()}

    c1, c2, c3 = st.columns(3)
    with c1:
        p1_uid = st.selectbox(
            "Participant 1 (The Believer)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=0 if persona_options else 0,
        )
    with c2:
        p2_uid = st.selectbox(
            "Participant 2 (The Skeptic)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=1 if len(persona_options) > 1 else 0,
        )
    with c3:
        st.info("Mode: Adversarial stress test")

    st.session_state.copy_type = st.selectbox(
        "Copy type",
        options=["Headline", "Email", "Sales Page", "Other"],
        index=["Headline", "Email", "Sales Page", "Other"].index(st.session_state.get("copy_type", "Email")),
    )

    marketing_topic = st.text_area("Paste creative", key="marketing_topic", height=260)

    wc = word_count(marketing_topic)
    tk = estimate_tokens(marketing_topic)
    if wc > 0:
        st.caption(f"Input size: {wc} words (approx {tk} tokens)")

    if wc >= 900:
        st.info(
            "Long copy detected. Recommended: enable brief extraction and show participants a shorter excerpt (First N words)."
        )

    # Heuristic claim-risk flags shown ONLY here (not on app landing)
    live_flags = claim_risk_flags(marketing_topic)
    if live_flags:
        st.warning("Claim-risk flags (heuristic): " + ", ".join(live_flags))
        st.caption(
            "These flags are triggered by wording patterns (e.g., urgency language). "
            "They are not a legal judgment, but they often correlate with trust/compliance risk."
        )

    with st.expander("Long copy settings", expanded=False):
        st.checkbox("Strip common email footer/boilerplate", value=bool(st.session_state.get("strip_footer", True)), key="strip_footer")

        st.radio(
            "What participants see",
            options=["Full text", "First N words", "Custom excerpt"],
            index=["Full text", "First N words", "Custom excerpt"].index(st.session_state.get("participant_scope", "First N words")),
            key="participant_scope",
            horizontal=True,
        )

        if st.session_state.participant_scope == "First N words":
            st.slider("N words", 150, 1500, int(st.session_state.get("participant_n_words", 650)), 50, key="participant_n_words")
        elif st.session_state.participant_scope == "Custom excerpt":
            st.text_area(
                "Custom excerpt (paste only the section you want tested)",
                value=st.session_state.get("participant_custom_excerpt", ""),
                key="participant_custom_excerpt",
                height=140,
            )

        st.slider(
            "Max words sent to moderator",
            300,
            4500,
            int(st.session_state.get("moderator_max_words", 3500)),
            100,
            key="moderator_max_words",
        )

        st.checkbox(
            "Auto-extract structured brief (recommended for long copy)",
            value=bool(st.session_state.get("auto_extract_brief", True)),
            key="auto_extract_brief",
        )

        st.caption("Hard caps: participants <= 1500 words; moderator <= 4500 words.")

    c_run, c_reset = st.columns([1, 1])
    with c_run:
        run = st.button("Start Focus Group", type="primary")
    with c_reset:
        if st.button("Reset focus group"):
            for k, v in {
                "marketing_topic": "",
                "debate_history": [],
                "moderator_raw": "",
                "moderator_json": None,
                "suggested_rewrite": "",
                "campaign_assets": None,
                "creative_excerpt": "",
                "creative_moderator_text": "",
                "creative_brief": None,
                "creative_brief_raw": "",
                "creative_risk_flags": [],
            }.items():
                st.session_state[k] = v
            st.rerun()

    if run:
        st.session_state.debate_history = []
        st.session_state.moderator_raw = ""
        st.session_state.moderator_json = None
        st.session_state.suggested_rewrite = ""
        st.session_state.campaign_assets = None
        st.session_state.creative_brief = None
        st.session_state.creative_brief_raw = ""

        if not openai_ready:
            st.error("OpenAI is not configured. Set OPENAI_API_KEY.")
            st.stop()

        p_a = persona_options.get(p1_uid)
        p_b = persona_options.get(p2_uid)
        if not p_a or not p_b:
            st.error("Please select two participants.")
            st.stop()

        if not (marketing_topic or "").strip():
            st.warning("Paste some creative first.")
            st.stop()

        cleaned_full, participant_excerpt, moderator_text, flags = _prepare_creative_inputs(marketing_topic)
        st.session_state.creative_excerpt = participant_excerpt
        st.session_state.creative_moderator_text = moderator_text
        st.session_state.creative_risk_flags = flags

        brief: Optional[Dict[str, Any]] = None
        brief_raw: str = ""

        if st.session_state.get("auto_extract_brief"):
            with st.spinner("Extracting structured brief..."):
                brief, brief_raw = extract_structured_brief(st.session_state.copy_type, moderator_text)
            st.session_state.creative_brief = brief
            st.session_state.creative_brief_raw = brief_raw

        brief_summary = format_brief_summary(brief) if isinstance(brief, dict) else ""

        st.markdown("---")
        st.subheader("What the personas saw")
        with st.expander("Preview excerpt used in the debate", expanded=True):
            st.code(participant_excerpt or "(empty)")

        if st.session_state.get("auto_extract_brief"):
            with st.expander("Preview extracted brief JSON", expanded=False):
                if isinstance(brief, dict):
                    st.code(_safe_json_dumps(brief))
                else:
                    st.code(brief_raw or "(No brief JSON extracted)")

        st.markdown("---")
        st.subheader("Debate")

        base_instruction = _debate_base_instruction()
        role_a = _role_prompt(p_a, "The Believer")
        role_b = _role_prompt(p_b, "The Skeptic")

        creative_context = (
            f"COPY TYPE: {st.session_state.copy_type}\n\n"
            "CREATIVE (what you saw):\n"
            f"{participant_excerpt}\n\n"
        )
        if brief_summary:
            creative_context += "CONTEXT SUMMARY (extracted):\n" + brief_summary + "\n\n"

        # Round 1: Believer opening
        msg_a = query_openai(
            [
                {"role": "system", "content": base_instruction + "\n\n" + role_a},
                {
                    "role": "user",
                    "content": (
                        creative_context
                        + "TASK: Give your opening statement as The Believer.\n"
                        "Output with these labels exactly:\n"
                        "- Open or ignore (and why):\n"
                        "- What excites me / what I want to believe:\n"
                        "- Where I hesitate / what I need clarified:\n"
                        "- One specific change that would increase trust (quote the line you would add/replace):\n"
                    ),
                },
            ],
            model=st.session_state.openai_model,
            temperature=float(st.session_state.openai_temperature),
        )
        st.session_state.debate_history.append({"name": p_a["core"].get("name"), "uid": p_a["uid"], "text": msg_a})
        st.markdown(f"**{p_a['core'].get('name')} (Believer):** {msg_a}")
        time.sleep(0.25)

        # Round 2: Skeptic opening
        msg_b = query_openai(
            [
                {"role": "system", "content": base_instruction + "\n\n" + role_b},
                {
                    "role": "user",
                    "content": (
                        creative_context
                        + f"Believer just said:\n{msg_a}\n\n"
                        "TASK: Respond as The Skeptic. Address the believer's points and the creative.\n"
                        "Output with these labels exactly:\n"
                        "- My immediate reaction:\n"
                        "- Specific red flags (quote phrases):\n"
                        "- Proof I'd need to see:\n"
                        "- One change that would make me consider it:\n"
                    ),
                },
            ],
            model=st.session_state.openai_model,
            temperature=float(st.session_state.openai_temperature),
        )
        st.session_state.debate_history.append({"name": p_b["core"].get("name"), "uid": p_b["uid"], "text": msg_b})
        st.markdown(f"**{p_b['core'].get('name')} (Skeptic):** {msg_b}")
        time.sleep(0.25)

        # Round 3: Believer rebuttal (more debate-like)
        msg_a2 = query_openai(
            [
                {"role": "system", "content": base_instruction + "\n\n" + role_a},
                {
                    "role": "user",
                    "content": (
                        creative_context
                        + f"Skeptic said:\n{msg_b}\n\n"
                        "TASK: Rebut the skeptic.\n"
                        "Rules:\n"
                        "- Do NOT repeat your opening statement.\n"
                        "- Do NOT use the opening labels (Open or ignore, etc.). Write as a direct reply.\n"
                        "- Directly respond to the skeptic's top 2 points.\n"
                        "- Quote at least 1 phrase from the creative.\n"
                        "- Give 1 compromise edit that would keep curiosity but reduce manipulation.\n"
                        "- End with 1 concrete proof element you would accept.\n"
                        "- Keep it punchy (max 6 sentences).\n"
                    ),
                },
            ],
            model=st.session_state.openai_model,
            temperature=float(st.session_state.openai_temperature),
        )
        st.session_state.debate_history.append({"name": p_a["core"].get("name"), "uid": p_a["uid"], "text": msg_a2})
        st.markdown(f"**{p_a['core'].get('name')} (Believer):** {msg_a2}")
        time.sleep(0.25)

        # Round 4: Skeptic counter
        msg_b2 = query_openai(
            [
                {"role": "system", "content": base_instruction + "\n\n" + role_b},
                {
                    "role": "user",
                    "content": (
                        creative_context
                        + f"Believer rebuttal:\n{msg_a2}\n\n"
                        "TASK: Counter the believer.\n"
                        "Rules:\n"
                        "- Do NOT restate your opening statement.\n"
                        "- Do NOT use the opening labels. Write as a direct reply.\n"
                        "- Name the minimum credible version of this message (what it must say).\n"
                        "- Provide 1 question you'd ask before paying/clicking.\n"
                        "- Provide 1 tone change (e.g., less hype, more specifics).\n"
                        "- Keep it punchy (max 6 sentences).\n"
                    ),
                },
            ],
            model=st.session_state.openai_model,
            temperature=float(st.session_state.openai_temperature),
        )
        st.session_state.debate_history.append({"name": p_b["core"].get("name"), "uid": p_b["uid"], "text": msg_b2})
        st.markdown(f"**{p_b['core'].get('name')} (Skeptic):** {msg_b2}")

        # Moderator
        st.markdown("---")
        st.subheader("Strategic Analysis (Moderator)")

        transcript = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])

        mod_prompt = f"""
You are a legendary direct response copywriter acting as a focus-group moderator.

COPY TYPE: {st.session_state.copy_type}

WHAT THE PARTICIPANTS SAW:
{participant_excerpt}

STRUCTURED BRIEF (may be null):
{_safe_json_dumps(brief) if isinstance(brief, dict) else 'null'}

TRANSCRIPT:
{transcript}

FULL CREATIVE (moderator view, may be truncated):
{moderator_text}

OUTPUT:
Return ONLY a single JSON object (no markdown, no commentary) that matches exactly this schema:

{moderator_schema(st.session_state.copy_type)}

Constraints:
- No guarantees or performance promises.
- Reduce hype; increase specificity and credibility.
- For Email rewrites: body ~150-250 words.
- For Headline rewrites: provide 8-12 options; pick a best_pick.
- For Sales Page rewrites: rewrite key blocks, not the entire page.
""".strip()

        with st.spinner("Moderator is analysing..."):
            mod_raw = query_gemini(mod_prompt, model_name=st.session_state.gemini_model)

        st.session_state.moderator_raw = mod_raw
        mj = extract_json_object(mod_raw)

        # If JSON parse failed, try one repair pass via OpenAI
        if mj is None and openai_ready:
            repair_prompt = f"""
Convert the following into valid JSON ONLY, matching this schema exactly:

{moderator_schema(st.session_state.copy_type)}

TEXT TO CONVERT:
{mod_raw}
""".strip()
            repaired = query_openai(
                [{"role": "user", "content": repair_prompt}],
                model=st.session_state.openai_model,
                temperature=0.0,
            )
            mj = extract_json_object(repaired)
            if mj is not None:
                st.session_state.moderator_raw = repaired

        st.session_state.moderator_json = mj

        if mj is None:
            st.info(mod_raw)
            st.warning("Moderator output wasn't valid JSON; displayed raw text.")
        else:
            st.success("Moderator analysis ready.")

            st.markdown(f"**Executive summary:** {mj.get('executive_summary','')}")
            st.markdown(f"**Real why:** {mj.get('real_why','')}")
            st.markdown(f"**Trust gap:** {mj.get('trust_gap','')}")

            if mj.get("key_objections"):
                st.markdown("**Key objections:**")
                for x in mj.get("key_objections"):
                    st.markdown(f"- {x}")

            if mj.get("proof_needed"):
                st.markdown("**Proof needed:**")
                for x in mj.get("proof_needed"):
                    st.markdown(f"- {x}")

            if mj.get("risk_flags"):
                st.markdown("**Risk flags:**")
                for x in mj.get("risk_flags"):
                    st.markdown(f"- {x}")

            if mj.get("actionable_fixes"):
                st.markdown("**Actionable fixes:**")
                for x in mj.get("actionable_fixes"):
                    st.markdown(f"- {x}")

            rewrite = _ensure_dict(mj.get("rewrite"))

            st.markdown("---")
            st.markdown("### Rewrite")

            ct_id = _copy_type_id(str(mj.get("copy_type") or st.session_state.copy_type))

            if ct_id == "headline":
                best = rewrite.get("best_pick")
                if best:
                    st.markdown(f"**Best pick:** {best}")
                if rewrite.get("headlines"):
                    st.markdown("**Headline options:**")
                    for h in rewrite.get("headlines"):
                        st.markdown(f"- {h}")
                if rewrite.get("supporting_lines"):
                    st.markdown("**Supporting lines:**")
                    for sline in rewrite.get("supporting_lines"):
                        st.markdown(f"- {sline}")

            elif ct_id == "sales_page":
                if rewrite.get("hero_headline"):
                    st.markdown(f"**Hero headline:** {rewrite.get('hero_headline')}")
                if rewrite.get("hero_subhead"):
                    st.markdown(f"**Hero subhead:** {rewrite.get('hero_subhead')}")
                if rewrite.get("bullets"):
                    st.markdown("**Bullets:**")
                    for b in rewrite.get("bullets"):
                        st.markdown(f"- {b}")
                if rewrite.get("proof_block"):
                    st.markdown("**Proof block:**")
                    st.write(rewrite.get("proof_block"))
                if rewrite.get("offer_stack"):
                    st.markdown("**Offer stack:**")
                    for o in rewrite.get("offer_stack"):
                        st.markdown(f"- {o}")
                if rewrite.get("cta"):
                    st.markdown(f"**CTA:** {rewrite.get('cta')}")
                if rewrite.get("risk_reducer"):
                    st.markdown(f"**Risk reducer:** {rewrite.get('risk_reducer')}")

            elif ct_id == "email":
                if rewrite.get("subject"):
                    st.markdown(f"**Subject:** {rewrite.get('subject')}")
                if rewrite.get("preheader"):
                    st.markdown(f"**Preheader:** {rewrite.get('preheader')}")
                if rewrite.get("body"):
                    st.markdown("**Body:**")
                    st.write(rewrite.get("body"))
                if rewrite.get("cta"):
                    st.markdown(f"**CTA:** {rewrite.get('cta')}")
                if rewrite.get("ps"):
                    st.markdown(f"**P.S.:** {rewrite.get('ps')}")

            else:
                if rewrite.get("headline"):
                    st.markdown(f"**Headline:** {rewrite.get('headline')}")
                if rewrite.get("body"):
                    st.markdown("**Body:**")
                    st.write(rewrite.get("body"))
                if rewrite.get("cta"):
                    st.markdown(f"**CTA:** {rewrite.get('cta')}")

            st.session_state.suggested_rewrite = _safe_json_dumps(mj)

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
                    brief2 = st.session_state.creative_brief

                    campaign_prompt = f"""
You are a full-stack marketing team. Use the strategic insight below to generate campaign assets.

COPY TYPE: {st.session_state.copy_type}

STRUCTURED BRIEF (may be null):
{_safe_json_dumps(brief2) if isinstance(brief2, dict) else 'null'}

STRATEGIC INSIGHT (JSON):
{_safe_json_dumps(insight)}

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

        if st.session_state.creative_brief is not None:
            st.download_button(
                "Download creative brief (json)",
                data=_safe_json_dumps(st.session_state.creative_brief),
                file_name="creative_brief.json",
                mime="application/json",
            )

        if st.session_state.moderator_json:
            st.download_button(
                "Download moderator analysis (json)",
                data=_safe_json_dumps(st.session_state.moderator_json),
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
