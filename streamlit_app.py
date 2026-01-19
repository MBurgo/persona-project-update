import os
import json
import re
import time
from dataclasses import dataclass
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


# ────────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foolish Persona Portal", layout="centered", page_icon="🃏")


# ────────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ────────────────────────────────────────────────────────────────────────────────
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
    code { white-space: pre-wrap; }
</style>
""",
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────
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


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def estimate_tokens(text: str) -> int:
    # Heuristic: in typical marketing copy, ~1 word ≈ 1.45 tokens
    wc = word_count(text)
    return int(wc * 1.45)


def truncate_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


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


def claim_risk_flags(text: str) -> List[str]:
    """Very lightweight claim-risk heuristic for marketing copy."""
    if not text or not text.strip():
        return []
    t = text.lower()
    patterns = {
        "Guaranteed / certainty language": ["guaranteed", "can't lose", "sure thing", "no risk", "risk-free", "100%"],
        "Urgency pressure": [
            "urgent",
            "act now",
            "limited time",
            "today only",
            "last chance",
            "ends tonight",
            "flash sale",
            "act quickly",
            "expires",
        ],
        "Implied future performance": ["will double", "will triple", "can't miss", "next nvidia", "take off explosively"],
        "Overly absolute claims": ["always", "never", "everyone", "no one"],
    }
    hits: List[str] = []
    for label, toks in patterns.items():
        if any(tok in t for tok in toks):
            hits.append(label)
    return hits


def safe_option(default: str, options: List[str]) -> str:
    return default if default in options else options[0]


def pretty_dt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


# ────────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ────────────────────────────────────────────────────────────────────────────────
# NOTE: do NOT assign session_state keys from widget return values. Widgets control
# their own session_state values via the `key` parameter.

st.session_state.setdefault("chat_history", {})  # persona_uid -> list[(q, a)]
st.session_state.setdefault("selected_persona_uid", None)

# Model settings
st.session_state.setdefault("openai_model", "gpt-4o")
st.session_state.setdefault("openai_temperature", 0.7)
st.session_state.setdefault("gemini_model", "gemini-3-flash-preview")
st.session_state.setdefault("brief_model", "gpt-4o-mini")
st.session_state.setdefault("max_batch", 10)

# Focus group inputs
st.session_state.setdefault("copy_type", "Email")
st.session_state.setdefault("marketing_topic", "")

# Long-copy settings
st.session_state.setdefault("fg_participant_scope", "First N words")
st.session_state.setdefault("fg_participant_n_words", 450)
st.session_state.setdefault("fg_custom_excerpt", "")
st.session_state.setdefault("fg_extract_brief", True)

# Focus group outputs (persisted)
st.session_state.setdefault("fg_last_run", None)  # dict


# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────────────────────
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
        for p in sorted(base.glob("personas*.json")):
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
    into the new schema {segments:[...]}.
    """
    groups = _ensure_list(old.get("personas"))

    default_summaries = {
        "Next Generation Investors (18–24 years)": "Tech-native starters building wealth early; influenced by peers and social proof.",
        "Emerging Wealth Builders (25–34 years)": "Balancing deposits, careers, and investing; want efficient, credible guidance.",
        "Established Accumulators (35–49 years)": "Time-poor family builders; value trust, clarity, and proven process.",
        "Pre-Retirees (50–64 years)": "Capital preservation + retirement income; skeptical of hype; want detail and proof.",
        "Retirees (65+ years)": "Stability seekers focused on income and peace of mind.",
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

            # Normalize enrichment spelling if present
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


# ────────────────────────────────────────────────────────────────────────────────
# AI CLIENTS
# ────────────────────────────────────────────────────────────────────────────────

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


def query_openai(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> str:
    if not openai_ready or client_openai is None:
        return "Error: OpenAI client not configured (missing OPENAI_API_KEY or openai package)."

    model = model or st.session_state.get("openai_model", "gpt-4o")
    temperature = float(temperature if temperature is not None else st.session_state.get("openai_temperature", 0.7))

    try:
        completion = client_openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error: {str(e)}"


def query_gemini(prompt: str, model_name: Optional[str] = None) -> str:
    model_name = model_name or st.session_state.get("gemini_model", "gemini-3-flash-preview")

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


# ────────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ────────────────────────────────────────────────────────────────────────────────

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
        f"You are {name}, a {age}-year-old {occupation} based in {location}. Income: ${income}.\n"
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
        "Be specific and concrete. Don't give financial advice; focus on reactions to marketing, credibility, and decision triggers. "
        "Keep answers under ~140 words unless asked for depth."
    )


def brief_extraction_prompt(copy_type: str, text: str) -> str:
    # Strict JSON extraction prompt for long copy.
    return f"""
You are a senior conversion strategist. Extract a structured brief from the marketing creative.

COPY TYPE: {copy_type}

CREATIVE (verbatim):
{text}

Return ONLY a single JSON object (no markdown, no commentary) with this structure:

{{
  "copy_type": "{copy_type}",
  "audience_assumed": "...",
  "primary_promise": "...",
  "mechanism_or_angle": "...",
  "offer_summary": "...",
  "cta": "...",
  "price_or_discount": "...",
  "key_claims": ["..."],
  "proof_elements_present": ["..."],
  "missing_proof": ["..."],
  "tone": ["..."],
  "sections_detected": ["..."],
  "confusing_or_unanswered": ["..."],
  "risk_flags": ["..."],
  "quick_fixes": ["..."]
}}

Rules:
- If something is unknown, use an empty string or empty list.
- Keep strings concise.
""".strip()


def summarize_brief_for_personas(brief: Optional[Dict[str, Any]]) -> str:
    if not isinstance(brief, dict):
        return ""

    def _clip(s: str, n: int = 140) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 1].rstrip() + "…"

    lines: List[str] = []
    if brief.get("primary_promise"):
        lines.append(f"Promise: {_clip(str(brief.get('primary_promise')))}")
    if brief.get("mechanism_or_angle"):
        lines.append(f"Angle: {_clip(str(brief.get('mechanism_or_angle')))}")
    if brief.get("offer_summary"):
        lines.append(f"Offer: {_clip(str(brief.get('offer_summary')))}")
    if brief.get("cta"):
        lines.append(f"CTA: {_clip(str(brief.get('cta')))}")
    claims = _ensure_list(brief.get("key_claims"))[:4]
    if claims:
        lines.append("Claims: " + "; ".join([_clip(str(x), 90) for x in claims]))
    missing = _ensure_list(brief.get("missing_proof"))[:3]
    if missing:
        lines.append("Missing proof: " + "; ".join([_clip(str(x), 90) for x in missing]))

    if not lines:
        return ""

    return "\n".join([f"- {ln}" for ln in lines])


def participant_task(copy_type: str) -> str:
    if copy_type == "Headline":
        return (
            "Answer in 4 short bullets:\n"
            "1) Click or ignore (and why)\n"
            "2) What you think this *really* means (implied promise)\n"
            "3) Trust reaction (what feels credible / not)\n"
            "4) One rewrite suggestion (<= 12 words)"
        )
    if copy_type == "Email":
        return (
            "Answer in 4 short bullets:\n"
            "1) Open or ignore (and why)\n"
            "2) Trust/credibility reaction\n"
            "3) Biggest question holding you back\n"
            "4) One change that improves it"
        )
    if copy_type == "Sales Page":
        return (
            "Answer in 5 short bullets:\n"
            "1) Would you keep reading or bounce (and where)\n"
            "2) Strongest section (and why)\n"
            "3) Weakest section (and why)\n"
            "4) Proof you need before believing\n"
            "5) One concrete fix"
        )
    return (
        "Answer in 4 short bullets:\n"
        "1) What grabs you (if anything)\n"
        "2) What feels off / unclear\n"
        "3) What proof you need\n"
        "4) One improvement"
    )


def moderator_prompt(copy_type: str, transcript: str, creative_for_moderator: str, brief_json: Optional[Dict[str, Any]]) -> str:
    # Structured output schemas by copy type.
    base_fields = (
        '  "executive_summary": "...",\n'
        '  "real_why": "...",\n'
        '  "trust_gap": "...",\n'
        '  "key_objections": ["..."],\n'
        '  "proof_needed": ["..."],\n'
        '  "risk_flags": ["..."],\n'
        '  "actionable_fixes": ["..."],\n'
    )

    if copy_type == "Headline":
        rewrite_schema = (
            '  "rewrite": {\n'
            '    "headlines": ["..."],\n'
            '    "angle_notes": "..."\n'
            '  },\n'
            '  "notes": "..."\n'
        )
        constraints = (
            "Constraints:\n"
            "- Provide 10 headlines. Each <= 12 words.\n"
            "- Make the angle specific early (avoid pure mystery).\n"
            "- Avoid guarantees or performance promises.\n"
        )
    elif copy_type == "Email":
        rewrite_schema = (
            '  "rewrite": {\n'
            '    "subject": "...",\n'
            '    "preheader": "...",\n'
            '    "body": "...",\n'
            '    "cta": "...",\n'
            '    "ps": "..."\n'
            '  },\n'
            '  "alt_subjects": ["..."],\n'
            '  "notes": "..."\n'
        )
        constraints = (
            "Constraints:\n"
            "- Subject <= 70 characters.\n"
            "- Preheader <= 110 characters.\n"
            "- Body 150-250 words, clear and credible (AU tone).\n"
            "- Avoid guarantees or performance promises.\n"
        )
    elif copy_type == "Sales Page":
        rewrite_schema = (
            '  "section_feedback": [\n'
            '    {"section": "Hero", "what_works": "...", "what_hurts": "...", "fix": "..."}\n'
            '  ],\n'
            '  "rewrite": {\n'
            '    "hero_headline": "...",\n'
            '    "hero_subhead": "...",\n'
            '    "bullets": ["..."],\n'
            '    "proof_block": "...",\n'
            '    "offer_stack": ["..."],\n'
            '    "cta_block": "...",\n'
            '    "cta_button": "..."\n'
            '  },\n'
            '  "notes": "..."\n'
        )
        constraints = (
            "Constraints:\n"
            "- Focus on rewriting key blocks (not the entire page).\n"
            "- Bullets: 5-7. Offer stack: 3-6 items.\n"
            "- Avoid guarantees or performance promises.\n"
        )
    else:
        rewrite_schema = (
            '  "rewrite": {\n'
            '    "headline": "...",\n'
            '    "body": "..."\n'
            '  },\n'
            '  "notes": "..."\n'
        )
        constraints = (
            "Constraints:\n"
            "- Keep rewrite concise and concrete.\n"
            "- Avoid guarantees or performance promises.\n"
        )

    brief_block = ""
    if isinstance(brief_json, dict) and brief_json:
        # Keep brief small to avoid token bloat
        brief_block = "\n\nEXTRACTED BRIEF (JSON):\n" + json.dumps(brief_json, ensure_ascii=False)

    return f"""
You are a legendary Direct Response Copywriter (Motley Fool style) acting as a focus-group moderator.
You are strict, practical, and credibility-first.

COPY TYPE: {copy_type}

TRANSCRIPT:
{transcript}

CREATIVE:
{creative_for_moderator}
{brief_block}

OUTPUT:
Return ONLY a single JSON object (no markdown, no commentary) with this structure:

{{
{base_fields}{rewrite_schema}}}

{constraints}
""".strip()


# ────────────────────────────────────────────────────────────────────────────────
# ACTION CALLBACKS
# ────────────────────────────────────────────────────────────────────────────────

def reset_focus_group() -> None:
    # This runs as a button callback, so it can safely mutate widget-bound keys.
    st.session_state["marketing_topic"] = ""
    st.session_state["fg_custom_excerpt"] = ""
    st.session_state["fg_last_run"] = None


def apply_rewrite_to_input() -> None:
    run = st.session_state.get("fg_last_run")
    if not isinstance(run, dict):
        return

    copy_type = run.get("copy_type") or st.session_state.get("copy_type", "Email")
    mj = run.get("moderator_json")
    if not isinstance(mj, dict):
        return

    rw = mj.get("rewrite") or {}

    new_text = ""
    if copy_type == "Headline":
        headlines = _ensure_list(rw.get("headlines"))
        if headlines:
            new_text = str(headlines[0]).strip()
    elif copy_type == "Email":
        subject = (rw.get("subject") or "").strip()
        preheader = (rw.get("preheader") or "").strip()
        body = (rw.get("body") or "").strip()
        cta = (rw.get("cta") or "").strip()
        ps = (rw.get("ps") or "").strip()

        parts = []
        if subject:
            parts.append(f"Subject: {subject}")
        if preheader:
            parts.append(f"Preheader: {preheader}")
        if body:
            parts.append(body)
        if cta:
            parts.append(f"CTA: {cta}")
        if ps:
            parts.append(f"P.S.: {ps}")
        new_text = "\n\n".join([p for p in parts if p])
    elif copy_type == "Sales Page":
        hero_h = (rw.get("hero_headline") or "").strip()
        hero_s = (rw.get("hero_subhead") or "").strip()
        bullets = _ensure_list(rw.get("bullets"))
        proof = (rw.get("proof_block") or "").strip()
        offer = _ensure_list(rw.get("offer_stack"))
        cta_block = (rw.get("cta_block") or "").strip()
        cta_btn = (rw.get("cta_button") or "").strip()

        lines: List[str] = []
        if hero_h:
            lines.append(f"H1: {hero_h}")
        if hero_s:
            lines.append(f"Subhead: {hero_s}")
        if bullets:
            lines.append("\nBullets:\n" + "\n".join([f"- {str(b).strip()}" for b in bullets if str(b).strip()]))
        if proof:
            lines.append(f"\nProof block:\n{proof}")
        if offer:
            lines.append("\nOffer stack:\n" + "\n".join([f"- {str(x).strip()}" for x in offer if str(x).strip()]))
        if cta_block:
            lines.append(f"\nCTA section:\n{cta_block}")
        if cta_btn:
            lines.append(f"CTA button: {cta_btn}")

        new_text = "\n\n".join([ln for ln in lines if ln.strip()])
    else:
        headline = (rw.get("headline") or "").strip()
        body = (rw.get("body") or "").strip()
        if headline and body:
            new_text = f"{headline}\n\n{body}"
        else:
            new_text = headline or body

    if new_text.strip():
        st.session_state["marketing_topic"] = new_text.strip()
        # Clear last run so the user can generate a fresh debate for the rewrite
        st.session_state["fg_last_run"] = None


# ────────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    st.selectbox(
        "OpenAI model",
        options=["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
        index=["gpt-4o", "gpt-4o-mini", "gpt-4.1"].index(safe_option(st.session_state.get("openai_model", "gpt-4o"), ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])),
        key="openai_model",
    )
    st.slider("OpenAI temperature", 0.0, 1.5, float(st.session_state.get("openai_temperature", 0.7)), 0.1, key="openai_temperature")

    st.selectbox(
        "Gemini model (moderator)",
        options=["gemini-3-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=["gemini-3-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash"].index(
            safe_option(st.session_state.get("gemini_model", "gemini-3-flash-preview"), ["gemini-3-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash"])
        ),
        key="gemini_model",
    )

    st.selectbox(
        "Brief extraction model (OpenAI)",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
        index=["gpt-4o-mini", "gpt-4o", "gpt-4.1"].index(
            safe_option(st.session_state.get("brief_model", "gpt-4o-mini"), ["gpt-4o-mini", "gpt-4o", "gpt-4.1"])
        ),
        key="brief_model",
        help="Used only when 'Auto-extract a structured brief' is enabled in Focus Group Debate.",
    )

    st.markdown("---")
    st.markdown("### 🧪 Batch Controls")
    st.slider("Max personas per batch", 1, 25, int(st.session_state.get("max_batch", 10)), 1, key="max_batch")

    st.markdown("---")
    st.markdown("### 📄 Data")
    if personas_path is None:
        st.error("No personas JSON found. Add a 'personas.json' next to this app.")
    else:
        st.caption(f"Loaded: {personas_path.name}")

    st.markdown("---")
    st.markdown("### 🔑 API Status")
    if openai_ready:
        st.success("OpenAI configured")
    else:
        st.warning("OpenAI missing OPENAI_API_KEY or package")

    if gemini_ready:
        st.success("Gemini configured")
    else:
        st.warning("Gemini missing GOOGLE_API_KEY or package")


# ────────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────────────────────────────────────
st.title("🧠 The Foolish Synthetic Audience")

st.markdown(
    """
<div style="background:#f0f2f6;padding:20px;border-left:6px solid #485cc7;border-radius:10px;margin-bottom:25px">
    <h4 style="margin-top:0">ℹ️ About This Tool</h4>
    <p>This tool uses <strong>OpenAI</strong> for persona simulation and <strong>Gemini</strong> (with OpenAI fallback) for strategic analysis and rewrite generation.</p>
    <p class="small-muted">Tip: For long sales pages, use the Long copy settings to show participants an excerpt while still briefing the moderator with more context.</p>
</div>
""",
    unsafe_allow_html=True,
)


tab1, tab2 = st.tabs(["🗣️ Individual Interview", "⚔️ Focus Group Debate"])


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
        key="interview_segment",
    )

    if selected_segment_id == "All":
        with st.expander("🔍 Segment Cheat Sheet"):
            for s in segment_options:
                if s.get("summary"):
                    st.markdown(f"**{s['segment_label']}**\n{s['summary']}\n")
    else:
        selected = next((s for s in segment_options if s["segment_id"] == selected_segment_id), None)
        if selected and selected.get("summary"):
            with st.expander("🔍 Segment Overview", expanded=True):
                st.write(selected["summary"])

    filtered_list = (
        all_personas_flat
        if selected_segment_id == "All"
        else [p for p in all_personas_flat if p["segment_id"] == selected_segment_id]
    )

    st.markdown("### 👥 Select a Persona")
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

        st.markdown("### 💡 Suggested Questions")
        suggestions = _ensure_list(core.get("suggestions"))
        if suggestions:
            cols_s = st.columns(min(len(suggestions), 3))
            for idx, s in enumerate(suggestions[:3]):
                if cols_s[idx % 3].button(f"Ask: {str(s)[:40]}...", key=f"sugg_{selected_entry['uid']}_{idx}"):
                    st.session_state["question_input"] = s
                    st.rerun()
        else:
            st.caption("No specific suggestions for this persona.")

        st.markdown("### 💬 Interaction")
        st.session_state.setdefault("question_input", "")
        user_input = st.text_area("Enter your question:", value=st.session_state.get("question_input", ""), key="q_input")
        ask_all = st.checkbox("Ask ALL visible personas (Batch Test)")

        if st.button("Ask Persona(s)", type="primary"):
            if not user_input.strip():
                st.warning("Please enter a question.")
            else:
                target_list = filtered_list if ask_all else [selected_entry]

                if ask_all and len(target_list) > int(st.session_state.get("max_batch", 10)):
                    st.warning(
                        f"Batch capped at {int(st.session_state.get('max_batch', 10))} personas (selected segment contains {len(target_list)})."
                    )
                    target_list = target_list[: int(st.session_state.get("max_batch", 10))]

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

                        ans = query_openai(messages)
                        st.session_state.chat_history.setdefault(persona_uid, []).append((user_input, ans))

                st.success("Responses received!")
                st.session_state["question_input"] = ""
                st.rerun()

        if ask_all:
            st.markdown("#### Batch Results")
            for target in filtered_list[: int(st.session_state.get("max_batch", 10))]:
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
    st.header("⚔️ Marketing Focus Group")
    st.markdown("Stress-test creative with a Believer vs Skeptic debate, then generate a format-appropriate rewrite.")

    persona_options = {p["uid"]: p for p in all_personas_flat}
    persona_labels = {uid: f"{p['core'].get('name','Unknown')} ({p['segment_label']})" for uid, p in persona_options.items()}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.selectbox(
            "Participant 1 (Believer)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=0 if persona_options else 0,
            key="fg_p1_uid",
        )
    with c2:
        st.selectbox(
            "Participant 2 (Skeptic)",
            options=list(persona_options.keys()),
            format_func=lambda uid: persona_labels.get(uid, uid),
            index=1 if len(persona_options) > 1 else 0,
            key="fg_p2_uid",
        )
    with c3:
        st.selectbox(
            "Copy type",
            options=["Headline", "Email", "Sales Page", "Other"],
            index=["Headline", "Email", "Sales Page", "Other"].index(
                safe_option(st.session_state.get("copy_type", "Email"), ["Headline", "Email", "Sales Page", "Other"])
            ),
            key="copy_type",
        )
    with c4:
        st.button("Reset", on_click=reset_focus_group)

    marketing_topic = st.text_area(
        "Paste creative",
        key="marketing_topic",
        height=240,
        placeholder="Paste a headline, full email (subject + body), or sales page copy here…",
    )

    wc = word_count(marketing_topic)
    tok = estimate_tokens(marketing_topic)
    st.caption(f"Input size: {wc} words (approx {tok} tokens)")

    risk = claim_risk_flags(marketing_topic)
    if risk:
        st.warning("Claim-risk flags detected: " + ", ".join(risk))

    if wc >= 800:
        st.info("Long copy detected. Recommended: enable brief extraction and show participants a shorter excerpt (First N words).")

    with st.expander("Long copy settings", expanded=False):
        st.selectbox(
            "Participants see",
            options=["Full text (capped)", "First N words", "Custom excerpt"],
            index=["Full text (capped)", "First N words", "Custom excerpt"].index(
                safe_option(st.session_state.get("fg_participant_scope", "First N words"), ["Full text (capped)", "First N words", "Custom excerpt"])
            ),
            key="fg_participant_scope",
            help="To keep the debate reliable, participants are capped even if you choose full text.",
        )

        st.slider(
            "First N words (if selected)",
            min_value=150,
            max_value=1200,
            value=int(st.session_state.get("fg_participant_n_words", 450)),
            step=50,
            key="fg_participant_n_words",
        )

        st.text_area(
            "Custom excerpt (if selected)",
            key="fg_custom_excerpt",
            height=140,
            placeholder="Paste the excerpt you want personas to react to (e.g., hero + offer).",
        )

        st.checkbox(
            "Auto-extract a structured brief (recommended for long copy)",
            value=bool(st.session_state.get("fg_extract_brief", True)),
            key="fg_extract_brief",
            disabled=not openai_ready,
            help="Uses OpenAI to extract a brief JSON (promise/offer/claims/proof).",
        )
        if st.session_state.get("fg_extract_brief") and not openai_ready:
            st.caption("OpenAI is not configured, so brief extraction is disabled.")

    # Compute excerpt and moderator text caps
    PARTICIPANT_CAP_WORDS = 1500
    MODERATOR_CAP_WORDS = 4500

    def get_excerpt_for_participants(full_text: str) -> str:
        scope = st.session_state.get("fg_participant_scope", "First N words")
        if scope == "Custom excerpt":
            custom = st.session_state.get("fg_custom_excerpt", "")
            if custom.strip():
                return truncate_words(custom.strip(), PARTICIPANT_CAP_WORDS)
            # fall back
            return truncate_words(full_text, int(st.session_state.get("fg_participant_n_words", 450)))
        if scope == "Full text (capped)":
            return truncate_words(full_text, PARTICIPANT_CAP_WORDS)
        # First N words
        return truncate_words(full_text, int(st.session_state.get("fg_participant_n_words", 450)))

    def get_text_for_moderator(full_text: str) -> str:
        return truncate_words(full_text, MODERATOR_CAP_WORDS)

    # --- Start focus group
    if st.button("🚀 Start Focus Group", type="primary"):
        if not marketing_topic.strip():
            st.warning("Please paste creative first.")
            st.stop()

        p1_uid = st.session_state.get("fg_p1_uid")
        p2_uid = st.session_state.get("fg_p2_uid")
        copy_type = st.session_state.get("copy_type", "Email")

        p_a = persona_options.get(p1_uid)
        p_b = persona_options.get(p2_uid)

        if not p_a or not p_b:
            st.error("Please select two participants.")
            st.stop()

        excerpt = get_excerpt_for_participants(marketing_topic)
        creative_for_moderator = get_text_for_moderator(marketing_topic)

        # 0) Brief extraction (optional)
        brief_raw = ""
        brief_json = None
        if bool(st.session_state.get("fg_extract_brief", True)) and openai_ready:
            with st.spinner("Extracting brief…"):
                brief_raw = query_openai(
                    [{"role": "user", "content": brief_extraction_prompt(copy_type, creative_for_moderator)}],
                    model=st.session_state.get("brief_model", "gpt-4o-mini"),
                    temperature=0.2,
                )
                brief_json = extract_json_object(brief_raw)

        brief_summary = summarize_brief_for_personas(brief_json)

        base_instruction = (
            "IMPORTANT: This is a simulation for marketing research. "
            "You are roleplaying a specific persona. Do NOT sound like a generic AI. "
            "Do not give financial advice; focus on reactions to marketing, credibility, and decision triggers. "
            "Be specific. Avoid repeating the same template in every turn."
        )

        def role_prompt(entry: Dict[str, Any], stance: str) -> str:
            core = entry["core"]
            bt = _ensure_dict(core.get("behavioural_traits"))
            values = ", ".join(_ensure_list(core.get("values"))[:5])
            goals = "; ".join(_ensure_list(core.get("goals"))[:4])
            concerns = "; ".join(_ensure_list(core.get("concerns"))[:4])

            stance_block = (
                "You WANT the message to be true. You focus on upside, possibility, and emotional appeal. "
                "You defend the message against skepticism, but you still sound like a real person."
                if stance == "Believer"
                else "You are allergic to hype. You look for missing specifics, credibility gaps, and implied claims. "
                "You call out anything that sounds too good to be true."
            )

            return (
                f"ROLE: You are {core.get('name')}, a {core.get('age')}-year-old {core.get('occupation')}.\n"
                f"BIO: {core.get('narrative','')}\n"
                f"VALUES: {values}\n"
                f"GOALS: {goals}\n"
                f"CONCERNS: {concerns}\n"
                f"RISK TOLERANCE: {bt.get('risk_tolerance','Unknown')}\n\n"
                f"STANCE: {stance}\n{stance_block}"
            )

        role_a = role_prompt(p_a, "Believer")
        role_b = role_prompt(p_b, "Skeptic")

        persona_brief = ""
        if brief_summary:
            persona_brief = "\n\nBRIEF SUMMARY (for context):\n" + brief_summary

        task = participant_task(copy_type)

        # 1) Believer
        with st.spinner("Believer reacting…"):
            msg_a = query_openai(
                [
                    {"role": "system", "content": base_instruction + "\n\n" + role_a},
                    {
                        "role": "user",
                        "content": (
                            f"You are reacting to {copy_type} creative.\n\n"
                            f"CREATIVE (excerpt):\n{excerpt}{persona_brief}\n\n"
                            f"TASK:\n{task}"
                        ),
                    },
                ]
            )

        time.sleep(0.2)

        # 2) Skeptic
        with st.spinner("Skeptic responding…"):
            msg_b = query_openai(
                [
                    {"role": "system", "content": base_instruction + "\n\n" + role_b},
                    {
                        "role": "user",
                        "content": (
                            f"You are reacting to the same {copy_type} creative.\n\n"
                            f"CREATIVE (excerpt):\n{excerpt}{persona_brief}\n\n"
                            f"The Believer said:\n{msg_a}\n\n"
                            "Respond directly to their points. Don't restate the creative. "
                            "Call out what feels manipulative or unclear.\n\n"
                            f"TASK:\n{task}"
                        ),
                    },
                ]
            )

        time.sleep(0.2)

        # 3) Believer rebuttal (tighter)
        with st.spinner("Believer rebutting…"):
            msg_a2 = query_openai(
                [
                    {"role": "system", "content": base_instruction + "\n\n" + role_a},
                    {
                        "role": "user",
                        "content": (
                            "Reply to the Skeptic in 5-6 sentences max.\n"
                            "- Acknowledge 1 fair critique\n"
                            "- Defend 1 element that still excites you\n"
                            "- Suggest 1 specific improvement that would keep the upside but build trust\n\n"
                            f"Skeptic said:\n{msg_b}"
                        ),
                    },
                ]
            )

        time.sleep(0.2)

        # 4) Skeptic counter (tighter)
        with st.spinner("Skeptic countering…"):
            msg_b2 = query_openai(
                [
                    {"role": "system", "content": base_instruction + "\n\n" + role_b},
                    {
                        "role": "user",
                        "content": (
                            "Counter the Believer in 5-6 sentences max.\n"
                            "- Say what specific proof/detail would convert you\n"
                            "- Name the single most damaging phrase or move in the creative\n"
                            "- Provide one rewrite principle (not a full rewrite)\n\n"
                            f"Believer said:\n{msg_a2}"
                        ),
                    },
                ]
            )

        debate_turns = [
            {"name": p_a["core"].get("name"), "uid": p_a["uid"], "role": "Believer", "text": msg_a},
            {"name": p_b["core"].get("name"), "uid": p_b["uid"], "role": "Skeptic", "text": msg_b},
            {"name": p_a["core"].get("name"), "uid": p_a["uid"], "role": "Believer", "text": msg_a2},
            {"name": p_b["core"].get("name"), "uid": p_b["uid"], "role": "Skeptic", "text": msg_b2},
        ]

        transcript = "\n".join([f"{x['name']} ({x['role']}): {x['text']}" for x in debate_turns])

        # Moderator
        st.spinner("Moderator analysing…")
        with st.spinner("Moderator analysing…"):
            mod_raw = query_gemini(moderator_prompt(copy_type, transcript, creative_for_moderator, brief_json))

        mod_json = extract_json_object(mod_raw)

        st.session_state["fg_last_run"] = {
            "created_at": time.time(),
            "p1_uid": p1_uid,
            "p2_uid": p2_uid,
            "copy_type": copy_type,
            "creative_full": marketing_topic,
            "excerpt": excerpt,
            "creative_for_moderator": creative_for_moderator,
            "brief_raw": brief_raw,
            "brief_json": brief_json,
            "debate_turns": debate_turns,
            "moderator_raw": mod_raw,
            "moderator_json": mod_json,
            "campaign_assets": None,
        }

        st.rerun()

    # --- Persisted output renderer (critical: stays visible across reruns)
    run = st.session_state.get("fg_last_run")
    if isinstance(run, dict):
        st.markdown("---")
        st.subheader("What the personas saw")

        with st.expander("Preview excerpt used in the debate", expanded=False):
            st.write(run.get("excerpt", ""))

        with st.expander("Preview extracted brief JSON", expanded=False):
            if run.get("brief_json"):
                st.code(json.dumps(run.get("brief_json"), ensure_ascii=False, indent=2), language="json")
            elif run.get("brief_raw"):
                st.code(run.get("brief_raw"), language="text")
            else:
                st.caption("No brief extracted.")

        st.subheader("Debate")
        for turn in _ensure_list(run.get("debate_turns")):
            name = turn.get("name", "Persona")
            role = turn.get("role", "")
            text = turn.get("text", "")
            st.markdown(f"**{name} ({role})**: {text}")
            st.divider()

        st.subheader("Strategic Analysis (Moderator)")
        mj = run.get("moderator_json")
        if isinstance(mj, dict):
            st.success("Moderator analysis ready.")
            if mj.get("executive_summary"):
                st.markdown(f"**Executive summary:** {mj.get('executive_summary')}")
            if mj.get("real_why"):
                st.markdown(f"**Real why:** {mj.get('real_why')}")
            if mj.get("trust_gap"):
                st.markdown(f"**Trust gap:** {mj.get('trust_gap')}")

            if mj.get("key_objections"):
                st.markdown("**Key objections:**")
                for x in _ensure_list(mj.get("key_objections")):
                    st.markdown(f"- {x}")

            if mj.get("proof_needed"):
                st.markdown("**Proof needed:**")
                for x in _ensure_list(mj.get("proof_needed")):
                    st.markdown(f"- {x}")

            if mj.get("risk_flags"):
                st.markdown("**Risk flags:**")
                for x in _ensure_list(mj.get("risk_flags")):
                    st.markdown(f"- {x}")

            if mj.get("actionable_fixes"):
                st.markdown("**Actionable fixes:**")
                for x in _ensure_list(mj.get("actionable_fixes")):
                    st.markdown(f"- {x}")

            st.markdown("---")
            st.markdown("### ✍️ Rewrite")
            copy_type = run.get("copy_type", "Email")
            rw = mj.get("rewrite") or {}

            if copy_type == "Headline":
                headlines = _ensure_list(rw.get("headlines"))
                if headlines:
                    for h in headlines:
                        st.markdown(f"- {h}")
                if rw.get("angle_notes"):
                    st.caption(str(rw.get("angle_notes")))
            elif copy_type == "Email":
                if rw.get("subject"):
                    st.markdown(f"**Subject:** {rw.get('subject')}")
                if rw.get("preheader"):
                    st.markdown(f"**Preheader:** {rw.get('preheader')}")
                if rw.get("body"):
                    st.markdown(f"**Body:**\n\n{rw.get('body')}")
                if rw.get("cta"):
                    st.markdown(f"**CTA:** {rw.get('cta')}")
                if rw.get("ps"):
                    st.markdown(f"**P.S.:** {rw.get('ps')}")
                if mj.get("alt_subjects"):
                    with st.expander("Alternate subjects", expanded=False):
                        for s in _ensure_list(mj.get("alt_subjects")):
                            st.markdown(f"- {s}")
            elif copy_type == "Sales Page":
                if rw.get("hero_headline"):
                    st.markdown(f"**Hero headline:** {rw.get('hero_headline')}")
                if rw.get("hero_subhead"):
                    st.markdown(f"**Hero subhead:** {rw.get('hero_subhead')}")
                if rw.get("bullets"):
                    st.markdown("**Bullets:**")
                    for b in _ensure_list(rw.get("bullets")):
                        st.markdown(f"- {b}")
                if rw.get("proof_block"):
                    st.markdown("**Proof block:**")
                    st.write(rw.get("proof_block"))
                if rw.get("offer_stack"):
                    st.markdown("**Offer stack:**")
                    for x in _ensure_list(rw.get("offer_stack")):
                        st.markdown(f"- {x}")
                if rw.get("cta_block"):
                    st.markdown("**CTA block:**")
                    st.write(rw.get("cta_block"))
                if rw.get("cta_button"):
                    st.markdown(f"**CTA button:** {rw.get('cta_button')}")

                if mj.get("section_feedback"):
                    with st.expander("Section-by-section feedback", expanded=False):
                        for s in _ensure_list(mj.get("section_feedback")):
                            if not isinstance(s, dict):
                                continue
                            st.markdown(f"**{s.get('section','Section')}**")
                            if s.get("what_works"):
                                st.markdown(f"- Works: {s.get('what_works')}")
                            if s.get("what_hurts"):
                                st.markdown(f"- Hurts: {s.get('what_hurts')}")
                            if s.get("fix"):
                                st.markdown(f"- Fix: {s.get('fix')}")
                            st.divider()
            else:
                if rw.get("headline"):
                    st.markdown(f"**Headline:** {rw.get('headline')}")
                if rw.get("body"):
                    st.markdown(f"**Body:**\n\n{rw.get('body')}")

        else:
            st.info(run.get("moderator_raw", ""))
            st.warning("Moderator output wasn't valid JSON; displayed raw text.")

        # Iterate & Production
        st.markdown("---")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("### 🔄 Iterate")
            st.caption("Apply the rewrite back into the input box, then re-run the debate.")
            st.button("Apply Rewrite to Input", on_click=apply_rewrite_to_input)

        with col_b:
            st.markdown("### 📢 Production")
            st.caption("Generate ad assets using the moderator insight (debate stays visible).")

            if st.button("✨ Generate Campaign Assets", type="secondary"):
                insight = run.get("moderator_json") or {"raw": run.get("moderator_raw")}
                brief = run.get("brief_json") or {}

                campaign_prompt = f"""
You are a Full-Stack Marketing Team.

COPY TYPE: {run.get('copy_type','Email')}

CREATIVE (for reference):
{run.get('creative_for_moderator','')}

EXTRACTED BRIEF (JSON):
{json.dumps(brief, ensure_ascii=False)}

STRATEGIC INSIGHT (JSON):
{json.dumps(insight, ensure_ascii=False)}

TASKS:
1) GOOGLE SEARCH ADS
   - 6 headlines (<= 30 chars)
   - 3 descriptions (<= 90 chars)

2) META AD
   - 3 primary text variants (80-140 words)
   - 3 short headlines (<= 6 words)

3) SALES PAGE HERO
   - 3 H1 options
   - 3 subheads
   - 3 CTA button options

Rules:
- Avoid guarantees or performance promises.
- Keep it credible and specific.

Output as Markdown with headers: ### Google Ads, ### Meta Ads, ### Sales Page Hero
""".strip()

                with st.spinner("Briefing the specialist copywriters…"):
                    assets = query_gemini(campaign_prompt)

                # Persist to the run so it never disappears on rerun
                updated = dict(run)
                updated["campaign_assets"] = assets
                st.session_state["fg_last_run"] = updated
                st.rerun()

        if run.get("campaign_assets"):
            st.divider()
            st.subheader("📦 Campaign Asset Pack")
            st.markdown(run.get("campaign_assets"))

        # Export
        st.divider()
        st.subheader("⬇️ Export")

        transcript_txt = "\n".join([f"{x.get('name','')}: {x.get('text','')}" for x in _ensure_list(run.get("debate_turns"))])
        st.download_button(
            "Download transcript (txt)",
            data=transcript_txt,
            file_name="focus_group_transcript.txt",
            mime="text/plain",
        )

        if isinstance(run.get("moderator_json"), dict):
            st.download_button(
                "Download moderator analysis (json)",
                data=json.dumps(run.get("moderator_json"), ensure_ascii=False, indent=2),
                file_name="moderator_analysis.json",
                mime="application/json",
            )

        if isinstance(run.get("brief_json"), dict):
            st.download_button(
                "Download creative brief (json)",
                data=json.dumps(run.get("brief_json"), ensure_ascii=False, indent=2),
                file_name="creative_brief.json",
                mime="application/json",
            )

        if run.get("campaign_assets"):
            st.download_button(
                "Download campaign assets (md)",
                data=run.get("campaign_assets"),
                file_name="campaign_assets.md",
                mime="text/markdown",
            )
