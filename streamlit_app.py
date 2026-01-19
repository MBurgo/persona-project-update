import os
import json
import re
import time
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
# PAGE CONFIGURATION
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


# ────────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────────────────────────
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
        globs = sorted(base.glob("personas*.json"))
        for p in globs:
            if p.exists() and p.is_file():
                return p

    return None


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


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
        "Next Generation Investors (18–24 years)": "Tech-native, socially-conscious starters focused on building asset bases early.",
        "Emerging Wealth Builders (25–34 years)": "Balancing house deposits, careers and investing; optimistic but wage-squeezed.",
        "Established Accumulators (35–49 years)": "Juggling family, mortgages and wealth growth; value efficiency and advice.",
        "Pre-Retirees (50–64 years)": "Capital-preservers planning retirement income; keen super watchers.",
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


def apply_rewrite_from_moderator() -> None:
    mj = st.session_state.get("moderator_json")
    if isinstance(mj, dict):
        subject = (mj.get("rewrite", {}) or {}).get("subject")
        body = (mj.get("rewrite", {}) or {}).get("body")
        if subject or body:
            merged = ""
            if subject:
                merged += f"Subject: {subject}\n\n"
            if body:
                merged += str(body).strip()
            st.session_state.marketing_topic = merged.strip()
            st.session_state.debate_history = []
            st.session_state.campaign_assets = None
            return

    # Fallback to raw heuristic if JSON isn't available
    raw = st.session_state.get("moderator_raw", "")
    if raw:
        st.session_state.marketing_topic = raw.strip()
        st.session_state.debate_history = []
        st.session_state.campaign_assets = None


# ────────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONFIG
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

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
    st.markdown("### 🧪 Batch Controls")
    st.session_state.max_batch = st.slider("Max personas per batch", 1, 25, 10, 1)

    st.markdown("---")
    st.markdown("### 📄 Data")
    if personas_path is None:
        st.error("No personas JSON found. Add a 'personas.json' next to this app.")
    else:
        st.caption(f"Loaded: {personas_path.name}")

    if not openai_ready:
        st.warning("OpenAI not configured (missing OPENAI_API_KEY).")
    if not gemini_ready:
        st.warning("Gemini not configured (missing GOOGLE_API_KEY).")


# ────────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────────────────────────────────────
st.title("🧠 The Foolish Synthetic Audience")

st.markdown(
    """
<div style="background:#f0f2f6;padding:20px;border-left:6px solid #485cc7;border-radius:10px;margin-bottom:25px">
    <h4 style="margin-top:0">ℹ️ About This Tool</h4>
    <p>This tool uses a hybrid setup: <strong>OpenAI</strong> for persona simulation and <strong>Gemini</strong> (with OpenAI fallback) for strategic analysis.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Quick claim-risk flags for the current marketing topic
risk_flags = claim_risk_flags(st.session_state.marketing_topic)
if risk_flags:
    st.warning("Claim-risk flags detected: " + ", ".join(risk_flags))


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
    )

    if selected_segment_id == "All":
        with st.expander("🔍 Segment Cheat Sheet"):
            for s in segment_options:
                if s["summary"]:
                    st.markdown(f"**{s['segment_label']}**\n{s['summary']}\n")
    else:
        # Show overview for the selected segment
        selected = next((s for s in segment_options if s["segment_id"] == selected_segment_id), None)
        if selected and selected.get("summary"):
            with st.expander("🔍 Segment Overview", expanded=True):
                st.write(selected["summary"])

    filtered_list = (
        all_personas_flat
        if selected_segment_id == "All"
        else [p for p in all_personas_flat if p["segment_id"] == selected_segment_id]
    )

    # Persona grid
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
        st.markdown("### 💡 Suggested Questions")
        suggestions = _ensure_list(core.get("suggestions"))
        if suggestions:
            cols_s = st.columns(min(len(suggestions), 3))
            for idx, s in enumerate(suggestions[:3]):
                if cols_s[idx % 3].button(f"Ask: {str(s)[:40]}...", key=f"sugg_{selected_entry['uid']}_{idx}"):
                    st.session_state.question_input = s
                    st.rerun()
        else:
            st.caption("No specific suggestions for this persona.")

        # Q&A
        st.markdown("### 💬 Interaction")
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
    st.header("⚔️ Marketing Focus Group")
    st.markdown("Pit two investors against each other to stress-test your copy.")

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
        st.info("🔥 Mode: Adversarial Stress Test")

    marketing_topic = st.text_area("Marketing Headline / Copy", key="marketing_topic", height=150)

    if st.button("🚀 Start Focus Group", type="primary"):
        st.session_state.debate_history = []
        st.session_state.moderator_raw = ""
        st.session_state.moderator_json = None
        st.session_state.suggested_rewrite = ""
        st.session_state.campaign_assets = None

        p_a = persona_options.get(p1_uid)
        p_b = persona_options.get(p2_uid)

        if not p_a or not p_b:
            st.error("Please select two participants.")
            st.stop()

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
                    "You defend the message against skepticism, but you still sound like a real person."
                    if stance == "The Believer"
                    else "You are critical of marketing hype and naturally risk-aware. You look for missing details, credibility gaps, and implied claims. "
                    "You call out anything that sounds too good to be true."
                )
            )

        role_a = role_prompt(p_a, "The Believer")
        role_b = role_prompt(p_b, "The Skeptic")

        chat_container = st.container()

        with chat_container:
            st.markdown(f"**Topic:** *{marketing_topic}*")
            st.divider()

            # 1) Believer
            msg_a = query_openai(
                [
                    {"role": "system", "content": base_instruction + "\n\n" + role_a},
                    {"role": "user", "content": f"React to this marketing text. What pulls you in? What do you *want* to believe?\n\nTEXT:\n{marketing_topic}"},
                ],
                model=st.session_state.openai_model,
                temperature=float(st.session_state.openai_temperature),
            )
            st.session_state.debate_history.append({"name": p_a["core"].get("name"), "uid": p_a["uid"], "text": msg_a})
            st.markdown(f"**{p_a['core'].get('name')} (The Believer)**: {msg_a}")
            time.sleep(0.5)

            # 2) Skeptic
            msg_b = query_openai(
                [
                    {"role": "system", "content": base_instruction + "\n\n" + role_b},
                    {
                        "role": "user",
                        "content": (
                            f"The marketing text is below. {p_a['core'].get('name')} just said: '{msg_a}'. "
                            "Give them a reality check. Be specific about what you'd need to see to trust it.\n\n"
                            f"TEXT:\n{marketing_topic}"
                        ),
                    },
                ],
                model=st.session_state.openai_model,
                temperature=float(st.session_state.openai_temperature),
            )
            st.session_state.debate_history.append({"name": p_b["core"].get("name"), "uid": p_b["uid"], "text": msg_b})
            st.markdown(f"**{p_b['core'].get('name')} (The Skeptic)**: {msg_b}")
            time.sleep(0.5)

            # 3) Believer retort
            msg_a2 = query_openai(
                [
                    {"role": "system", "content": base_instruction + "\n\n" + role_a},
                    {
                        "role": "user",
                        "content": (
                            f"You just got critiqued. {p_b['core'].get('name')} said: '{msg_b}'. "
                            "Respond as yourself. Acknowledge what feels fair, and restate what still excites you."
                        ),
                    },
                ],
                model=st.session_state.openai_model,
                temperature=float(st.session_state.openai_temperature),
            )
            st.session_state.debate_history.append({"name": p_a["core"].get("name"), "uid": p_a["uid"], "text": msg_a2})
            st.markdown(f"**{p_a['core'].get('name')} (The Believer)**: {msg_a2}")

            # 4) Moderator
            st.divider()
            st.subheader("📊 Strategic Analysis (Moderator)")

            transcript = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])

            mod_prompt = f"""
You are a legendary Direct Response Copywriter (Motley Fool style) acting as a focus-group moderator.

TRANSCRIPT:
{transcript}

MARKETING COPY:
{marketing_topic}

OUTPUT:
Return ONLY a single JSON object (no markdown, no commentary) with this structure:

{{
  \"real_why\": \"...\",
  \"trust_gap\": \"...\",
  \"key_objections\": [\"...\"],
  \"proof_needed\": [\"...\"],
  \"risk_flags\": [\"...\"],
  \"rewrite\": {{
    \"subject\": \"...\",
    \"body\": \"...\"
  }},
  \"notes\": \"...\"
}}

Constraints:
- Subject <= 70 characters.
- Body: 2-3 sentences, story-driven, personal, contrarian.
- Neutralise the skeptic's objection without making guarantees.
"""

            with st.spinner("Moderator is analysing..."):
                mod_raw = query_gemini(mod_prompt, model_name=st.session_state.gemini_model)

            st.session_state.moderator_raw = mod_raw
            mj = extract_json_object(mod_raw)
            st.session_state.moderator_json = mj

            if mj is None:
                st.info(mod_raw)
                st.warning("Moderator output wasn't valid JSON; displayed raw text.")
            else:
                st.success("Moderator analysis ready.")

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

                rewrite = mj.get("rewrite") or {}
                subject = rewrite.get("subject")
                body = rewrite.get("body")

                st.markdown("---")
                st.markdown("### ✍️ Rewrite")
                if subject:
                    st.markdown(f"**Subject:** {subject}")
                if body:
                    st.markdown(f"**Body:** {body}")

                st.session_state.suggested_rewrite = json.dumps(mj, ensure_ascii=False, indent=2)

    # Feedback loop & campaign generator
    if st.session_state.debate_history and (st.session_state.moderator_raw or st.session_state.moderator_json):
        st.markdown("---")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("### 🔄 Iterate")
            st.caption("Re-run debate with the rewrite applied.")
            st.button("Test Rewrite", on_click=apply_rewrite_from_moderator)

        with col_b:
            st.markdown("### 📢 Production")
            st.caption("Turn the insight into ad assets.")

            if st.button("✨ Generate Campaign Assets", type="secondary"):
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
"""

                    assets = query_gemini(campaign_prompt, model_name=st.session_state.gemini_model)
                    st.session_state.campaign_assets = assets

        if st.session_state.campaign_assets:
            st.divider()
            st.subheader("📦 Campaign Asset Pack")
            st.markdown(st.session_state.campaign_assets)

        # Export
        st.divider()
        st.subheader("⬇️ Export")

        transcript_txt = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])
        st.download_button(
            "Download transcript (txt)",
            data=transcript_txt,
            file_name="focus_group_transcript.txt",
            mime="text/plain",
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
