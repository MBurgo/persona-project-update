import os
import json
import re
import time
import hashlib
from datetime import datetime
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
    .pill {
        display:inline-block; padding:2px 10px; border-radius:999px;
        background:#f0f2f6; border:1px solid #d0d6e2; font-size:0.85rem;
        margin-right:6px; margin-bottom:6px;
    }
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
    s = (s or "").lower()
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
    """Very lightweight claim-risk heuristic for marketing copy."""
    if not text:
        return []
    t = text.lower()
    patterns = {
        "Guaranteed / certainty language": ["guaranteed", "can't lose", "sure thing", "no risk", "risk-free", "100%"],
        "Urgency pressure": ["urgent", "act now", "limited time", "today only", "last chance", "flash sale", "act quickly"],
        "Implied future performance": ["will double", "will triple", "can't miss", "next nvidia", "explode", "take off explosively"],
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


def compose_rewrite_for_textarea(copy_type: str, rewrite: Dict[str, Any]) -> str:
    """Convert moderator rewrite JSON into a single text blob suitable for the creative text box."""
    ct = (copy_type or "").lower()
    rw = _ensure_dict(rewrite)

    if ct == "headline":
        headlines = _ensure_list(rw.get("headlines"))
        if headlines:
            return str(headlines[0]).strip()
        if rw.get("headline"):
            return str(rw.get("headline")).strip()
        return ""

    if ct == "email":
        subject = rw.get("subject") or ""
        preheader = rw.get("preheader") or ""
        body = rw.get("body") or ""
        cta = rw.get("cta") or ""
        ps = rw.get("ps") or ""

        parts: List[str] = []
        if subject:
            parts.append(f"Subject: {subject}")
        if preheader:
            parts.append(f"Preheader: {preheader}")
        if body:
            parts.append("Body:\n" + str(body).strip())
        if cta:
            parts.append("CTA: " + str(cta).strip())
        if ps:
            parts.append("P.S.: " + str(ps).strip())
        return "\n\n".join(parts).strip()

    if ct in {"sales page", "sales_page", "salespage"}:
        hero_h1 = rw.get("hero_headline") or rw.get("h1") or ""
        subhead = rw.get("hero_subhead") or rw.get("subhead") or ""
        bullets = _ensure_list(rw.get("bullets"))
        proof = rw.get("proof_block") or ""
        offer = rw.get("offer_stack")
        cta = rw.get("cta") or ""

        out: List[str] = []
        if hero_h1:
            out.append(f"H1: {hero_h1}")
        if subhead:
            out.append(f"Subhead: {subhead}")
        if bullets:
            out.append("Key bullets:\n- " + "\n- ".join([str(b).strip() for b in bullets if str(b).strip()]))
        if proof:
            out.append("Proof block:\n" + str(proof).strip())
        if offer:
            offer_list = _ensure_list(offer)
            if offer_list:
                out.append("Offer stack:\n- " + "\n- ".join([str(x).strip() for x in offer_list if str(x).strip()]))
            elif isinstance(offer, str) and offer.strip():
                out.append("Offer stack:\n" + offer.strip())
        if cta:
            out.append("CTA: " + str(cta).strip())
        return "\n\n".join(out).strip()

    # Other / fallback
    headline = rw.get("headline") or ""
    body = rw.get("body") or ""
    if headline and body:
        return f"{headline}\n\n{body}".strip()
    return str(body or headline or "").strip()


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
st.session_state.setdefault("chat_history", {})
st.session_state.setdefault("debate_history", [])
st.session_state.setdefault("moderator_raw", "")
st.session_state.setdefault("moderator_json", None)
st.session_state.setdefault("campaign_assets", None)
st.session_state.setdefault("question_input", "")
st.session_state.setdefault("selected_persona_uid", None)

# Focus group settings
st.session_state.setdefault("marketing_topic", "")
st.session_state.setdefault("copy_type", "Email")
st.session_state.setdefault("fg_strip_footer", True)
st.session_state.setdefault("fg_extract_brief", True)
st.session_state.setdefault("fg_scope", "First N words")
st.session_state.setdefault("fg_first_n_words", 350)
st.session_state.setdefault("fg_custom_excerpt", "")
st.session_state.setdefault("fg_participant_max_words", 900)
st.session_state.setdefault("fg_moderator_max_words", 4500)

# Persisted "last run" context (used for re-rendering after reruns)
st.session_state.setdefault("fg_last_copy_used", "")
st.session_state.setdefault("fg_last_excerpt", "")
st.session_state.setdefault("fg_last_brief_json", None)
st.session_state.setdefault("fg_last_brief_raw", "")
st.session_state.setdefault("fg_last_copy_type", "")
st.session_state.setdefault("fg_last_p1_uid", "")
st.session_state.setdefault("fg_last_p2_uid", "")

# Simple run history
st.session_state.setdefault("fg_runs", [])  # list of dicts
st.session_state.setdefault("fg_current_run_id", None)


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent


def find_personas_file() -> Optional[Path]:
    """Find personas JSON in a robust way.

    Priority:
    1) personas.json in same folder as this app
    2) personas.json in current working directory
    3) any file matching personas*.json in same folder
    4) any file matching personas*.json in CWD
    """
    candidates: List[Path] = []
    direct = [APP_DIR / "personas.json", Path.cwd() / "personas.json"]
    candidates.extend([p for p in direct if p.exists()])

    if candidates:
        return candidates[0]

    for root in [APP_DIR, Path.cwd()]:
        for p in sorted(root.glob("personas*.json")):
            if p.exists():
                return p

    return None


def _patch_core(core: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(core or {})
    c.setdefault("future_confidence", 3)
    c.setdefault("family_support_received", False)
    c.setdefault("ideal_salary_for_comfort", 120_000)
    c.setdefault("budget_adjustments_6m", [])
    c.setdefault("super_engagement", "Unknown")
    c.setdefault("property_via_super_interest", "No")
    c.setdefault("income", 80_000)
    c.setdefault("goals", [])
    c.setdefault("values", [])
    c.setdefault("concerns", [])
    c.setdefault("decision_making", "")

    bt = _ensure_dict(c.get("behavioural_traits"))
    bt.setdefault("risk_tolerance", "Moderate")
    bt.setdefault("investment_experience", "Unknown")
    bt.setdefault("information_sources", [])
    bt.setdefault("preferred_channels", [])
    c["behavioural_traits"] = bt

    # normalize spelling for enrichment if present (legacy)
    if "behavioral_enrichment" in c and "behavioural_enrichment" not in c:
        c["behavioural_enrichment"] = c.pop("behavioral_enrichment")

    return c


def _convert_old_schema(old: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the original {personas: [{segment, male, female}, ...]} format
    into the new schema used by this app.
    """
    groups = _ensure_list(old.get("personas"))

    default_summaries = {
        "Next Generation Investors (18-24 years)": "Tech-native, socially-conscious starters focused on building asset bases early.",
        "Next Generation Investors (18–24 years)": "Tech-native, socially-conscious starters focused on building asset bases early.",
        "Emerging Wealth Builders (25-34 years)": "Balancing house deposits, careers and investing; optimistic but wage-squeezed.",
        "Emerging Wealth Builders (25–34 years)": "Balancing house deposits, careers and investing; optimistic but wage-squeezed.",
        "Established Accumulators (35-49 years)": "Juggling family, mortgages and wealth growth; value efficiency and advice.",
        "Established Accumulators (35–49 years)": "Juggling family, mortgages and wealth growth; value efficiency and advice.",
        "Pre-Retirees (50-64 years)": "Capital-preservers planning retirement income; keen super watchers.",
        "Pre-Retirees (50–64 years)": "Capital-preservers planning retirement income; keen super watchers.",
        "Retirees (65+ years)": "Stability-seekers prioritising income and low volatility.",
    }

    segments: List[Dict[str, Any]] = []

    for g in groups:
        label = g.get("segment", "Unknown")
        label_norm = normalize_dashes(label)
        seg_id = slugify(label_norm)
        summary = default_summaries.get(label_norm, "")

        people: List[Dict[str, Any]] = []
        for gender in ("male", "female"):
            if gender not in g:
                continue
            p = dict(g[gender] or {})

            if "behavioral_enrichment" in p and "behavioural_enrichment" not in p:
                p["behavioural_enrichment"] = p.pop("behavioral_enrichment")

            core = {k: v for k, v in p.items() if k not in {"scenarios", "peer_influence", "risk_tolerance_differences", "behavioural_enrichment"}}
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

        segments.append({"id": seg_id, "label": label_norm, "summary": summary, "personas": people})

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
        fallback = query_openai(
            [{"role": "user", "content": prompt}],
            model=st.session_state.get("openai_model", "gpt-4o"),
            temperature=float(st.session_state.get("openai_temperature", 0.7)),
        )
        return "Gemini Error (not configured).\n\nFallback Output:\n" + fallback

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip() or ""
    except Exception as e:
        fallback = query_openai(
            [{"role": "user", "content": prompt}],
            model=st.session_state.get("openai_model", "gpt-4o"),
            temperature=float(st.session_state.get("openai_temperature", 0.7)),
        )
        return f"Gemini Error ({str(e)}).\n\nFallback Output:\n" + fallback


# -----------------------------------------------------------------------------
# PROMPT BUILDERS
# -----------------------------------------------------------------------------

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
        "Be specific and concrete. Keep answers under ~140 words unless asked for depth."
    )


def build_focus_group_system_prompt(core: Dict[str, Any], stance: str, copy_type: str) -> str:
    base = build_persona_system_prompt(core)
    stance = stance.strip()
    ct = (copy_type or "").strip()

    if stance == "Believer":
        stance_rules = (
            "In this focus group you are THE BELIEVER. You want the marketing to be true. "
            "You look for upside, curiosity, possibility. You still notice credibility gaps, but you default to optimistic interpretation. "
            "You respond to the skeptic directly when prompted, without repeating yourself."
        )
    else:
        stance_rules = (
            "In this focus group you are THE SKEPTIC. You are allergic to hype. "
            "You look for missing specificity, credibility gaps, and implied claims. "
            "You respond to the believer directly when prompted, without repeating yourself."
        )

    return (
        base
        + "\n\n"
        + f"Copy type: {ct}\n"
        + stance_rules
        + "\n\n"
        + "Output rules: Be punchy and specific. Avoid generic AI phrasing. Do not give financial advice."
    )


def build_brief_extraction_prompt(copy_type: str, creative: str) -> List[Dict[str, str]]:
    ct = (copy_type or "").lower()

    schema = {
        "copy_type": "headline|email|sales_page|other",
        "audience_assumed": "...",
        "primary_promise": "...",
        "offer_summary": "...",
        "cta": "...",
        "key_claims": ["..."],
        "proof_elements": ["..."],
        "missing_proof": ["..."],
        "tone": "...",
        "sections_detected": ["..."],
        "risk_flags": ["..."],
        "questions_to_answer": ["..."],
    }

    sys = (
        "You are a senior direct-response marketing strategist. "
        "Extract a structured brief from the creative. "
        "Return ONLY valid JSON. No markdown. No commentary."
    )

    user = (
        f"COPY TYPE (user-selected): {copy_type}\n\n"
        "CREATIVE (raw):\n"
        f"{creative}\n\n"
        "Return a single JSON object matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "- Keep strings concise but specific.\n"
        "- If something is not present, use an empty string or empty array.\n"
        "- For risk_flags, include items like 'urgency pressure', 'mystery offer', 'implied performance', 'authority name-drop', 'insufficient proof'.\n"
    )

    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def build_moderator_prompt(
    copy_type: str,
    transcript: str,
    creative_for_moderator: str,
    excerpt_for_participants: str,
    brief_json: Optional[Dict[str, Any]],
) -> str:
    ct = (copy_type or "").strip()

    # Build rewrite schema based on copy type
    if ct.lower() == "headline":
        rewrite_schema = {
            "headlines": ["..."],
            "angle_notes": "...",
        }
    elif ct.lower() == "email":
        rewrite_schema = {
            "subject": "...",
            "preheader": "...",
            "body": "...",
            "cta": "...",
            "ps": "...",
        }
    elif ct.lower() in {"sales page", "sales_page", "salespage"}:
        rewrite_schema = {
            "hero_headline": "...",
            "hero_subhead": "...",
            "bullets": ["..."],
            "proof_block": "...",
            "offer_stack": ["..."],
            "cta": "...",
        }
    else:
        rewrite_schema = {
            "headline": "...",
            "body": "...",
            "cta": "...",
        }

    output_schema = {
        "executive_summary": "...",
        "real_why": "...",
        "trust_gap": "...",
        "psychology": {
            "believer_why_it_works": ["..."],
            "skeptic_why_it_fails": ["..."],
        },
        "key_objections": ["..."],
        "proof_needed": ["..."],
        "confusing_phrases": ["..."],
        "risk_flags": ["..."],
        "actionable_fixes": ["..."],
        "rewrite": rewrite_schema,
    }

    brief_blob = json.dumps(brief_json, ensure_ascii=False, indent=2) if isinstance(brief_json, dict) else "null"

    return f"""
You are a legendary Direct Response Copywriter (Motley Fool style) acting as a focus-group moderator.

COPY TYPE: {ct}

WHAT PARTICIPANTS SAW (excerpt):
{excerpt_for_participants}

EXTRACTED BRIEF (JSON):
{brief_blob}

FOCUS GROUP TRANSCRIPT:
{transcript}

FULL CREATIVE (may be longer than excerpt):
{creative_for_moderator}

OUTPUT:
Return ONLY a single JSON object (no markdown, no commentary) matching this schema:
{json.dumps(output_schema, ensure_ascii=False)}

Constraints:
- Be concrete and diagnostic (name the specific credibility gaps).
- Avoid guarantees and performance promises.
- Actionable_fixes should be specific edits (not generic advice).
- Rewrite must be appropriate to the selected copy type.
- Keep rewrite concise but usable.
""".strip()


# -----------------------------------------------------------------------------
# FOCUS GROUP CORE LOGIC
# -----------------------------------------------------------------------------

def build_creative_views(raw_text: str) -> Tuple[str, str, str]:
    """Return (clean_text, excerpt_for_participants, creative_for_moderator)."""
    clean = (raw_text or "").strip()

    if st.session_state.get("fg_strip_footer", True):
        clean = strip_common_email_footer(clean)

    scope = st.session_state.get("fg_scope", "First N words")
    excerpt = clean

    if scope == "Custom excerpt":
        custom = (st.session_state.get("fg_custom_excerpt") or "").strip()
        if custom:
            excerpt = custom
    elif scope == "First N words":
        n = int(st.session_state.get("fg_first_n_words", 350) or 350)
        excerpt = truncate_to_words(clean, n)

    # Hard caps
    participant_cap = int(st.session_state.get("fg_participant_max_words", 900) or 900)
    moderator_cap = int(st.session_state.get("fg_moderator_max_words", 4500) or 4500)

    excerpt = truncate_to_words(excerpt, participant_cap)
    creative_for_moderator = truncate_to_words(clean, moderator_cap)

    return clean, excerpt, creative_for_moderator


def maybe_extract_brief(copy_type: str, creative_clean: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (brief_json, brief_raw)."""
    if not st.session_state.get("fg_extract_brief", True):
        return None, ""

    # Reduce cost by limiting the text for extraction
    extraction_text = truncate_to_words(creative_clean, 2200)
    messages = build_brief_extraction_prompt(copy_type, extraction_text)

    raw = query_openai(
        messages,
        model=st.session_state.get("openai_model", "gpt-4o"),
        temperature=0.2,
    )
    brief = extract_json_object(raw)
    return brief, raw


def run_focus_group(
    copy_type: str,
    p_believer: Dict[str, Any],
    p_skeptic: Dict[str, Any],
    creative_excerpt: str,
    brief_summary: str,
) -> List[Dict[str, Any]]:
    """Run a 4-turn believer/skeptic debate. Returns debate_history list."""
    base_instruction = (
        "IMPORTANT: This is a simulation for marketing research. "
        "You are roleplaying a specific persona. Do NOT sound like a generic AI. "
        "Use specific vocabulary, worldview, and constraints from the persona. "
        "Do not give financial advice; focus on reactions to marketing and credibility."
    )

    sys_a = base_instruction + "\n\n" + build_focus_group_system_prompt(p_believer["core"], "Believer", copy_type)
    sys_b = base_instruction + "\n\n" + build_focus_group_system_prompt(p_skeptic["core"], "Skeptic", copy_type)

    excerpt_block = creative_excerpt
    brief_block = brief_summary.strip()

    context_bits = [f"CREATIVE EXCERPT:\n{excerpt_block}"]
    if brief_block:
        context_bits.append(f"BRIEF SUMMARY (for context):\n{brief_block}")
    context = "\n\n".join(context_bits)

    model = st.session_state.get("openai_model", "gpt-4o")
    temp = float(st.session_state.get("openai_temperature", 0.7))

    debate: List[Dict[str, Any]] = []

    # 1) Believer initial
    prompt_a1 = (
        "React as the BELIEVER. Output 5 bullets, each 1-2 sentences max:\n"
        "- Open/keep reading (why)\n"
        "- What excites you\n"
        "- What you assume the offer is\n"
        "- Biggest concern / question\n"
        "- One change to increase trust without killing curiosity\n\n"
        + context
    )

    msg_a1 = query_openai(
        [{"role": "system", "content": sys_a}, {"role": "user", "content": prompt_a1}],
        model=model,
        temperature=temp,
    )
    debate.append(
        {
            "role": "participant",
            "stance": "Believer",
            "name": p_believer["core"].get("name"),
            "uid": p_believer.get("uid"),
            "text": msg_a1,
        }
    )

    time.sleep(0.3)

    # 2) Skeptic response
    prompt_b1 = (
        "React as the SKEPTIC. Output 5 bullets, each 1-2 sentences max:\n"
        "- Open/ignore (why)\n"
        "- Biggest red flag\n"
        "- What proof you'd need\n"
        "- What feels manipulative / unclear\n"
        "- One change that would make you take it seriously\n\n"
        + context
        + "\n\nBELIEVER JUST SAID:\n"
        + msg_a1
    )

    msg_b1 = query_openai(
        [{"role": "system", "content": sys_b}, {"role": "user", "content": prompt_b1}],
        model=model,
        temperature=temp,
    )
    debate.append(
        {
            "role": "participant",
            "stance": "Skeptic",
            "name": p_skeptic["core"].get("name"),
            "uid": p_skeptic.get("uid"),
            "text": msg_b1,
        }
    )

    time.sleep(0.3)

    # 3) Believer rebuttal
    prompt_a2 = (
        "Respond to the skeptic directly.\n"
        "Rules:\n"
        "- Do NOT repeat your 5-bullet format.\n"
        "- Address the skeptic's top 2 points in plain language.\n"
        "- Admit what feels fair.\n"
        "- End with 1 specific suggestion for the copy.\n"
        "Max 6 sentences.\n\n"
        + "SKEPTIC SAID:\n"
        + msg_b1
    )

    msg_a2 = query_openai(
        [{"role": "system", "content": sys_a}, {"role": "user", "content": prompt_a2}],
        model=model,
        temperature=temp,
    )
    debate.append(
        {
            "role": "participant",
            "stance": "Believer",
            "name": p_believer["core"].get("name"),
            "uid": p_believer.get("uid"),
            "text": msg_a2,
        }
    )

    time.sleep(0.3)

    # 4) Skeptic counter
    prompt_b2 = (
        "Counter the believer.\n"
        "Rules:\n"
        "- Do NOT repeat your 5-bullet format.\n"
        "- Point out what still doesn't add up, and what single detail would change your mind.\n"
        "- Provide one concrete rewrite suggestion (one sentence, or a CTA line).\n"
        "Max 6 sentences.\n\n"
        + "BELIEVER REBUTTAL:\n"
        + msg_a2
    )

    msg_b2 = query_openai(
        [{"role": "system", "content": sys_b}, {"role": "user", "content": prompt_b2}],
        model=model,
        temperature=temp,
    )
    debate.append(
        {
            "role": "participant",
            "stance": "Skeptic",
            "name": p_skeptic["core"].get("name"),
            "uid": p_skeptic.get("uid"),
            "text": msg_b2,
        }
    )

    return debate


def save_focus_group_run(run: Dict[str, Any]) -> None:
    runs = st.session_state.get("fg_runs") or []
    runs = [r for r in runs if isinstance(r, dict)]
    runs.append(run)

    # Keep the last 20 runs to prevent unbounded growth
    runs = runs[-20:]
    st.session_state.fg_runs = runs
    st.session_state.fg_current_run_id = run.get("id")


def get_current_run() -> Optional[Dict[str, Any]]:
    rid = st.session_state.get("fg_current_run_id")
    runs = st.session_state.get("fg_runs") or []
    if not rid:
        return None
    for r in reversed(runs):
        if isinstance(r, dict) and r.get("id") == rid:
            return r
    return None


def render_focus_group_run(run: Dict[str, Any]) -> None:
    """Render a previously saved focus group run."""
    if not isinstance(run, dict):
        return

    st.markdown("---")
    st.subheader("Results")

    # Meta
    meta_bits: List[str] = []
    if run.get("created_at"):
        meta_bits.append(f"<span class='pill'>Run: {run['created_at']}</span>")
    if run.get("copy_type"):
        meta_bits.append(f"<span class='pill'>Copy type: {run['copy_type']}</span>")
    if run.get("participants"):
        p = run.get("participants") or {}
        a = p.get("believer") or ""
        b = p.get("skeptic") or ""
        if a and b:
            meta_bits.append(f"<span class='pill'>Believer: {a}</span>")
            meta_bits.append(f"<span class='pill'>Skeptic: {b}</span>")

    if meta_bits:
        st.markdown("""<div style="margin-bottom:8px">""" + " ".join(meta_bits) + "</div>", unsafe_allow_html=True)

    # What participants saw
    with st.expander("What the personas saw", expanded=True):
        excerpt = run.get("excerpt") or ""
        st.markdown("**Excerpt used in the debate**")
        st.code(excerpt or "(no excerpt)")

        brief_json = run.get("brief_json")
        if brief_json is not None:
            st.markdown("**Extracted brief (JSON)**")
            st.code(json.dumps(brief_json, ensure_ascii=False, indent=2))
        elif run.get("brief_raw"):
            st.markdown("**Extracted brief (raw)**")
            st.code(run.get("brief_raw"))
        else:
            st.caption("No brief extracted for this run.")

    # Debate transcript (this is the key fix: always re-render from saved state)
    with st.expander("Debate transcript", expanded=True):
        debate = _ensure_list(run.get("debate_history"))
        if not debate:
            st.caption("No debate transcript stored.")
        for turn in debate:
            name = turn.get("name") or "Participant"
            stance = turn.get("stance") or ""
            txt = turn.get("text") or ""
            st.markdown(f"**{name} ({stance})**")
            st.markdown(txt)
            st.divider()

    # Moderator analysis
    st.subheader("Strategic Analysis (Moderator)")
    mj = run.get("moderator_json")
    raw = run.get("moderator_raw")

    if isinstance(mj, dict):
        st.success("Moderator analysis ready.")

        for k in ["executive_summary", "real_why", "trust_gap"]:
            if mj.get(k):
                st.markdown(f"**{k.replace('_',' ').title()}:** {mj.get(k)}")

        psych = _ensure_dict(mj.get("psychology"))
        if psych:
            with st.expander("Psychology mapping", expanded=False):
                bw = _ensure_list(psych.get("believer_why_it_works"))
                sw = _ensure_list(psych.get("skeptic_why_it_fails"))
                if bw:
                    st.markdown("**Believer: why it works**")
                    for x in bw:
                        st.markdown(f"- {x}")
                if sw:
                    st.markdown("**Skeptic: why it fails**")
                    for x in sw:
                        st.markdown(f"- {x}")

        def _render_list(title: str, items: Any):
            items = _ensure_list(items)
            if items:
                st.markdown(f"**{title}:**")
                for x in items:
                    st.markdown(f"- {x}")

        _render_list("Key objections", mj.get("key_objections"))
        _render_list("Proof needed", mj.get("proof_needed"))
        _render_list("Confusing phrases", mj.get("confusing_phrases"))
        _render_list("Risk flags", mj.get("risk_flags"))
        _render_list("Actionable fixes", mj.get("actionable_fixes"))

        st.markdown("---")
        st.markdown("### Rewrite")
        rw = _ensure_dict(mj.get("rewrite"))
        ct = (run.get("copy_type") or "").lower()
        if ct == "headline":
            heads = _ensure_list(rw.get("headlines"))
            if heads:
                for i, h in enumerate(heads[:12], start=1):
                    st.markdown(f"{i}. {h}")
            else:
                st.code(json.dumps(rw, ensure_ascii=False, indent=2))
        elif ct == "email":
            if rw.get("subject"):
                st.markdown(f"**Subject:** {rw.get('subject')}")
            if rw.get("preheader"):
                st.markdown(f"**Preheader:** {rw.get('preheader')}")
            if rw.get("body"):
                st.markdown("**Body:**")
                st.markdown(str(rw.get("body")))
            if rw.get("cta"):
                st.markdown(f"**CTA:** {rw.get('cta')}")
            if rw.get("ps"):
                st.markdown(f"**P.S.:** {rw.get('ps')}")
        elif ct in {"sales page", "sales_page", "salespage"}:
            if rw.get("hero_headline"):
                st.markdown(f"**H1:** {rw.get('hero_headline')}")
            if rw.get("hero_subhead"):
                st.markdown(f"**Subhead:** {rw.get('hero_subhead')}")
            bullets = _ensure_list(rw.get("bullets"))
            if bullets:
                st.markdown("**Bullets:**")
                for b in bullets:
                    st.markdown(f"- {b}")
            if rw.get("proof_block"):
                st.markdown("**Proof block:**")
                st.markdown(str(rw.get("proof_block")))
            offer = _ensure_list(rw.get("offer_stack"))
            if offer:
                st.markdown("**Offer stack:**")
                for o in offer:
                    st.markdown(f"- {o}")
            if rw.get("cta"):
                st.markdown(f"**CTA:** {rw.get('cta')}")
        else:
            st.code(json.dumps(rw, ensure_ascii=False, indent=2))
    else:
        if raw:
            st.info(raw)
        else:
            st.caption("No moderator output stored for this run.")

    # Campaign assets
    if run.get("campaign_assets"):
        st.divider()
        st.subheader("Campaign Asset Pack")
        st.markdown(run.get("campaign_assets"))

    # Export
    st.divider()
    st.subheader("Export")

    transcript_txt = "\n".join([f"{x.get('name')}: {x.get('text')}" for x in _ensure_list(run.get("debate_history"))])
    st.download_button(
        "Download transcript (txt)",
        data=transcript_txt,
        file_name="focus_group_transcript.txt",
        mime="text/plain",
        key=f"dl_transcript_{run.get('id')}",
    )

    if isinstance(run.get("moderator_json"), dict):
        st.download_button(
            "Download moderator analysis (json)",
            data=json.dumps(run.get("moderator_json"), ensure_ascii=False, indent=2),
            file_name="moderator_analysis.json",
            mime="application/json",
            key=f"dl_mod_{run.get('id')}",
        )

    if run.get("brief_json") is not None:
        st.download_button(
            "Download creative brief (json)",
            data=json.dumps(run.get("brief_json"), ensure_ascii=False, indent=2),
            file_name="creative_brief.json",
            mime="application/json",
            key=f"dl_brief_{run.get('id')}",
        )

    if run.get("campaign_assets"):
        st.download_button(
            "Download campaign assets (md)",
            data=run.get("campaign_assets"),
            file_name="campaign_assets.md",
            mime="text/markdown",
            key=f"dl_assets_{run.get('id')}",
        )


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
    st.markdown("Paste creative (headline, full email, sales page), run a believer vs skeptic debate, then generate a structured rewrite and assets.")

    # Run history selector
    runs = st.session_state.get("fg_runs") or []
    if runs:
        with st.expander("Run history", expanded=False):
            def _run_label(r: Dict[str, Any]) -> str:
                ts = r.get("created_at") or ""
                ct = r.get("copy_type") or ""
                a = _ensure_dict(r.get("participants")).get("believer") or ""
                b = _ensure_dict(r.get("participants")).get("skeptic") or ""
                return f"{ts} | {ct} | {a} vs {b}".strip(" |")

            run_ids = [r.get("id") for r in runs if isinstance(r, dict) and r.get("id")]
            labels = {r.get("id"): _run_label(r) for r in runs if isinstance(r, dict) and r.get("id")}

            current = st.session_state.get("fg_current_run_id")
            if current not in run_ids:
                current = run_ids[-1]
                st.session_state.fg_current_run_id = current

            st.selectbox(
                "Select a previous run",
                options=run_ids,
                format_func=lambda rid: labels.get(rid, rid),
                key="fg_current_run_id",
            )

            if st.button("Clear run history"):
                st.session_state.fg_runs = []
                st.session_state.fg_current_run_id = None
                st.session_state.debate_history = []
                st.session_state.moderator_raw = ""
                st.session_state.moderator_json = None
                st.session_state.campaign_assets = None
                st.rerun()

    persona_options = {p["uid"]: p for p in all_personas_flat}
    persona_labels = {uid: f"{p['core'].get('name','Unknown')} ({p['segment_label']})" for uid, p in persona_options.items()}

    c1, c2, c3 = st.columns(3)
    with c1:
        p1_uid = st.selectbox(
            "Participant 1 (Believer)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=0 if persona_options else 0,
            key="fg_p1_uid",
        )
    with c2:
        p2_uid = st.selectbox(
            "Participant 2 (Skeptic)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=1 if len(persona_options) > 1 else 0,
            key="fg_p2_uid",
        )
    with c3:
        st.session_state.copy_type = st.selectbox(
            "Copy type",
            options=["Headline", "Email", "Sales Page", "Other"],
            index=["Headline", "Email", "Sales Page", "Other"].index(st.session_state.get("copy_type", "Email")),
            key="copy_type",
        )

    # Creative input
    marketing_topic = st.text_area(
        "Paste creative",
        key="marketing_topic",
        height=220,
        placeholder="Paste a headline, full email (subject + body), or sales page copy here...",
    )

    # Input stats and risk flags
    wc = word_count(marketing_topic)
    tok = estimate_tokens(marketing_topic)
    st.caption(f"Input size: {wc} words (approx {tok} tokens)")

    risk_flags = claim_risk_flags(marketing_topic)
    if risk_flags:
        st.warning("Claim-risk flags detected: " + ", ".join(risk_flags))

    if wc >= 800:
        st.info("Long copy detected. Recommended: enable brief extraction and show participants a shorter excerpt (First N words).")

    with st.expander("Long copy settings", expanded=False):
        st.session_state.fg_strip_footer = st.checkbox("Strip common email footer boilerplate", value=bool(st.session_state.get("fg_strip_footer", True)), key="fg_strip_footer")
        st.session_state.fg_extract_brief = st.checkbox("Auto-extract a structured brief (recommended)", value=bool(st.session_state.get("fg_extract_brief", True)), key="fg_extract_brief")

        st.session_state.fg_scope = st.radio(
            "What participants should see",
            options=["Full text", "First N words", "Custom excerpt"],
            index=["Full text", "First N words", "Custom excerpt"].index(st.session_state.get("fg_scope", "First N words")),
            key="fg_scope",
        )

        if st.session_state.fg_scope == "First N words":
            st.session_state.fg_first_n_words = st.number_input(
                "N words",
                min_value=50,
                max_value=2000,
                value=int(st.session_state.get("fg_first_n_words", 350) or 350),
                step=25,
                key="fg_first_n_words",
            )
        elif st.session_state.fg_scope == "Custom excerpt":
            st.session_state.fg_custom_excerpt = st.text_area(
                "Custom excerpt (participants will see exactly this)",
                value=st.session_state.get("fg_custom_excerpt", ""),
                height=140,
                key="fg_custom_excerpt",
            )

        st.session_state.fg_participant_max_words = st.slider(
            "Hard cap (participants) - words",
            min_value=150,
            max_value=1500,
            value=int(st.session_state.get("fg_participant_max_words", 900) or 900),
            step=50,
            key="fg_participant_max_words",
        )

        st.session_state.fg_moderator_max_words = st.slider(
            "Hard cap (moderator) - words",
            min_value=500,
            max_value=6500,
            value=int(st.session_state.get("fg_moderator_max_words", 4500) or 4500),
            step=250,
            key="fg_moderator_max_words",
        )

    # Actions
    col_run, col_reset = st.columns([2, 1])
    with col_run:
        start_clicked = st.button("Start focus group", type="primary")
    with col_reset:
        reset_clicked = st.button("Reset focus group")

    if reset_clicked:
        st.session_state.marketing_topic = ""
        st.session_state.debate_history = []
        st.session_state.moderator_raw = ""
        st.session_state.moderator_json = None
        st.session_state.campaign_assets = None
        st.session_state.fg_last_copy_used = ""
        st.session_state.fg_last_excerpt = ""
        st.session_state.fg_last_brief_json = None
        st.session_state.fg_last_brief_raw = ""
        st.session_state.fg_current_run_id = None
        st.rerun()

    if start_clicked:
        if not marketing_topic.strip():
            st.warning("Paste some creative first.")
        else:
            p_a = persona_options.get(p1_uid)
            p_b = persona_options.get(p2_uid)

            if not p_a or not p_b:
                st.error("Please select two participants.")
                st.stop()

            copy_type = st.session_state.get("copy_type", "Email")

            creative_clean, excerpt_for_participants, creative_for_moderator = build_creative_views(marketing_topic)

            # Brief extraction
            brief_json, brief_raw = maybe_extract_brief(copy_type, creative_clean)
            brief_summary = format_brief_summary(brief_json) if isinstance(brief_json, dict) else ""

            # Run debate
            with st.spinner("Running debate..."):
                debate = run_focus_group(
                    copy_type=copy_type,
                    p_believer=p_a,
                    p_skeptic=p_b,
                    creative_excerpt=excerpt_for_participants,
                    brief_summary=brief_summary,
                )

            transcript = "\n".join([f"{x['name']} ({x['stance']}): {x['text']}" for x in debate])

            # Moderator
            mod_prompt = build_moderator_prompt(
                copy_type=copy_type,
                transcript=transcript,
                creative_for_moderator=creative_for_moderator,
                excerpt_for_participants=excerpt_for_participants,
                brief_json=brief_json,
            )

            with st.spinner("Moderator is analysing..."):
                mod_raw = query_gemini(mod_prompt, model_name=st.session_state.get("gemini_model", "gemini-3-flash-preview"))

            mj = extract_json_object(mod_raw)

            # Save to session_state (for compatibility) + run history (for persistence)
            st.session_state.debate_history = debate
            st.session_state.moderator_raw = mod_raw
            st.session_state.moderator_json = mj
            st.session_state.campaign_assets = None

            st.session_state.fg_last_copy_used = marketing_topic
            st.session_state.fg_last_excerpt = excerpt_for_participants
            st.session_state.fg_last_brief_json = brief_json
            st.session_state.fg_last_brief_raw = brief_raw
            st.session_state.fg_last_copy_type = copy_type
            st.session_state.fg_last_p1_uid = p1_uid
            st.session_state.fg_last_p2_uid = p2_uid

            run = {
                "id": sha256_text(datetime.utcnow().isoformat() + sha256_text(marketing_topic)[:12])[:16],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "copy_type": copy_type,
                "participants": {
                    "believer": persona_labels.get(p1_uid, p1_uid),
                    "skeptic": persona_labels.get(p2_uid, p2_uid),
                },
                "creative": marketing_topic,
                "creative_clean": creative_clean,
                "excerpt": excerpt_for_participants,
                "brief_json": brief_json,
                "brief_raw": brief_raw,
                "debate_history": debate,
                "moderator_raw": mod_raw,
                "moderator_json": mj,
                "campaign_assets": None,
            }
            save_focus_group_run(run)
            st.rerun()

    # Production (assets)
    current_run = get_current_run()

    if current_run:
        st.markdown("---")

        # Generate assets button, but keep debate visible (rendered below from run state)
        col_iter, col_prod = st.columns([1, 2])

        with col_iter:
            st.markdown("### Iterate")
            if st.button("Apply rewrite to text box", key="apply_rw"):
                mj = current_run.get("moderator_json")
                if isinstance(mj, dict):
                    rewrite = _ensure_dict(mj.get("rewrite"))
                    rt = compose_rewrite_for_textarea(current_run.get("copy_type", ""), rewrite)
                    if rt:
                        st.session_state.marketing_topic = rt
                st.rerun()

        with col_prod:
            st.markdown("### Production")
            st.caption("Generate ad assets using the moderator insight and the extracted brief.")

            if st.button("Generate campaign assets", type="secondary", key="gen_assets"):
                with st.spinner("Generating assets..."):
                    insight = current_run.get("moderator_json") or {"raw": current_run.get("moderator_raw", "")}
                    brief = current_run.get("brief_json") or {}
                    ct = current_run.get("copy_type")

                    campaign_prompt = f"""
You are a Full-Stack Marketing Team. Use the STRATEGIC INSIGHT below to generate campaign assets.

COPY TYPE: {ct}

STRATEGIC INSIGHT (JSON):
{json.dumps(insight, ensure_ascii=False)}

EXTRACTED BRIEF (JSON):
{json.dumps(brief, ensure_ascii=False)}

TASKS:
1) GOOGLE SEARCH ADS:
   - 6 headlines (<= 30 chars)
   - 4 descriptions (<= 90 chars)

2) META ADS:
   - 3 primary texts (each <= 120 words)
   - 3 headlines (<= 5 words)

3) EMAIL SUBJECT IDEAS:
   - 10 subjects (<= 70 chars)
   - 5 preheaders (<= 90 chars)

4) SALES PAGE HERO:
   - 5 H1 options
   - 5 subhead options
   - 5 CTA button copy options

Output as Markdown with headers: ### Google Ads, ### Meta Ads, ### Email, ### Sales Page Hero
Avoid guarantees or performance promises.
""".strip()

                    assets = query_gemini(campaign_prompt, model_name=st.session_state.get("gemini_model", "gemini-3-flash-preview"))

                # Save to run
                current_run["campaign_assets"] = assets

                # Update run store
                runs = st.session_state.get("fg_runs") or []
                for i in range(len(runs) - 1, -1, -1):
                    if isinstance(runs[i], dict) and runs[i].get("id") == current_run.get("id"):
                        runs[i] = current_run
                        break
                st.session_state.fg_runs = runs

                st.rerun()

        # Render the selected run (this ensures transcript stays visible after reruns, including asset generation)
        render_focus_group_run(current_run)

    else:
        st.caption("No focus group run yet. Paste creative and click 'Start focus group'.")
