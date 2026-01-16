import streamlit as st
import json
from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────────
# Session‑state initialisation
# ────────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    # {persona_name|"ALL": [(question, answer), …]}
    st.session_state.chat_history: dict[str, list[tuple[str, str]]] = {}
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# ────────────────────────────────────────────────────────────────────────────────
# Load personas & patch missing Stake‑fields
# ────────────────────────────────────────────────────────────────────────────────
with open("personas.json", "r", encoding="utf-8") as f:
    persona_data = json.load(f)["personas"]

def _patch(p: dict) -> dict:
    """Ensure every persona has the Stake‑driven keys."""
    p.setdefault("future_confidence", 3)               # 1‑5 scale
    p.setdefault("family_support_received", False)     # bool
    p.setdefault("ideal_salary_for_comfort", 120_000)  # int
    p.setdefault("budget_adjustments_6m", [])          # list[str]
    p.setdefault("super_engagement", "Unknown")        # str
    p.setdefault("property_via_super_interest", "No")  # Yes | No | Maybe
    return p

for group in persona_data:
    for gender in ("male", "female"):
        if gender in group:
            group[gender] = _patch(group[gender])

# ────────────────────────────────────────────────────────────────────────────────
# OpenAI client
# ────────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

segment_summaries = {
    "Next Generation Investors (18–24 years)": "Tech‑native, socially‑conscious starters focused on building asset bases early.",
    "Emerging Wealth Builders (25–34 years)": "Balancing house deposits, careers and investing; optimistic but wage‑squeezed.",
    "Established Accumulators (35–49 years)": "Juggling family, mortgages and wealth growth; value efficiency and advice.",
    "Pre‑Retirees (50–64 years)": "Capital‑preservers planning retirement income; keen super watchers.",
    "Retirees (65+ years)": "Stability‑seekers prioritising income and low volatility.",
}

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Burgo's Persona Portal", layout="centered", page_icon="💬")

st.markdown(
    """<style>.stButton>button{border:1px solid #485cc7;border-radius:8px}</style>""",
    unsafe_allow_html=True,
)

st.title("🧠 Matt's Test Persona Portal")

st.markdown(
    """
    <div style="background:#f0f2f6;padding:20px;border-left:6px solid #485cc7;border-radius:10px;margin-bottom:25px">
        <h4 style="margin-top:0">ℹ️ About This Tool</h4>
        <p>This tool uses AI‑generated investor personas — built from real Australian investor research such as the ASX's "Investor Study", Investment Trends' "Australian Online Investor" study, and Stake’s 2024 "Ambition Report" — to simulate how different segments might respond to your ideas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("Select a persona to view their profile, then ask questions to gauge reactions.")

# ────────────────────────────────────────────────────────────────────────────────
# Segment filter + overview accordion
# ────────────────────────────────────────────────────────────────────────────────
segments = sorted({p["segment"] for p in persona_data})
selected_segment = st.selectbox("Filter by Segment", ["All"] + segments)

if selected_segment == "All":
    with st.expander("🔍 Segment cheat‑sheet"):
        for seg, blurb in segment_summaries.items():
            st.markdown(f"**{seg}**  \n{blurb}\n")
else:
    with st.expander("🔍 Segment overview", expanded=True):
        st.write(segment_summaries[selected_segment])

# ────────────────────────────────────────────────────────────────────────────────
# Persona grid
# ────────────────────────────────────────────────────────────────────────────────
filtered_personas = []
for group in persona_data:
    if selected_segment == "All" or group["segment"] == selected_segment:
        for gender in ("male", "female"):
            if (persona := group.get(gender)):
                filtered_personas.append({"persona": persona, "segment": group["segment"]})

st.markdown("## 👥 Select a Persona")
cols = st.columns([1, 1, 1], gap="small")
for i, entry in enumerate(filtered_personas):
    p = entry["persona"]
    with cols[i % 3]:
        if (img := p.get("image")):
            st.image(img, caption=p["name"], use_container_width=True)
        st.markdown(f"### {p['name']}")
        st.markdown(f"*{entry['segment']}*")
        st.markdown(f"📍 {p.get('location','')}")
        st.markdown(f"🎂 {p.get('age','')} years old")
        if st.button("Select", key=f"sel_{i}"):
            st.session_state.selected_persona = p
            st.session_state.selected_segment = entry["segment"]

# ────────────────────────────────────────────────────────────────────────────────
# Persona profile
# ────────────────────────────────────────────────────────────────────────────────
if "selected_persona" in st.session_state:
    persona = st.session_state.selected_persona
    seg = st.session_state.selected_segment

    st.markdown("---")
    st.markdown(
        f"""
        <div style="background:#e3f6d8;padding:20px;border-left:6px solid #43B02A;border-radius:10px">
            <h4 style="margin-top:0">{persona['name']} <span style="font-weight:normal">({seg})</span></h4>
            <p><strong>Age:</strong> {persona.get('age')}</p>
            <p><strong>Location:</strong> {persona.get('location')}</p>
            <p><strong>Occupation:</strong> {persona.get('occupation')}</p>
            <p><strong>Income:</strong> ${persona.get('income'):,}</p>
            <p><strong>Risk Tolerance:</strong> {persona.get('behavioural_traits', {}).get('risk_tolerance')}</p>
            <p><strong>Confidence score:</strong> {persona.get('future_confidence')}/5</p>
            <p><strong>Super engagement:</strong> {persona.get('super_engagement')}</p>
            <p><strong>Would tap super for property?</strong> {persona.get('property_via_super_interest')}</p>
            <p><strong>Ideal ‘comfortable’ salary:</strong> ${persona.get('ideal_salary_for_comfort'):,}</p>
            <p><strong>Recent budget cut‑backs:</strong><br>{''.join('• '+c+'<br>' for c in persona.get('budget_adjustments_6m'))}</p>
            <p><strong>Goals:</strong><br>{''.join('• '+g+'<br>' for g in persona.get('goals', []))}</p>
            <p><strong>Values:</strong> {', '.join(persona.get('values', []))}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### 💡 Suggested Questions")
    if persona.get("suggestions"):
        for s in persona["suggestions"]:
            if st.button(s):
                st.session_state.question_input = s
    else:
        st.markdown("*No suggestions for this persona.*")

# ────────────────────────────────────────────────────────────────────────────────
# Helper: build ChatCompletion messages with memory
# ────────────────────────────────────────────────────────────────────────────────
def _build_messages(persona_intro: str, history: list[tuple[str, str]], new_q: str):
    msgs = [
        {"role": "system", "content": "You are simulating an investor responding in a realistic, conversational tone."},
        {"role": "user", "content": persona_intro},
    ]
    for q, a in history[-4:]:
        msgs.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ])
    msgs.append({"role": "user", "content": new_q})
    return msgs

@st.cache_data(show_spinner=False)
def _ask_llm(messages: list):
    reply = client.chat.completions.create(model="o3", messages=messages)
    return reply.choices[0].message.content.strip()

def ask_persona(persona: dict, question: str):
    name = persona["name"]
    intro = (
        f"You are {name}, a {persona['age']}-year‑old {persona['occupation']} from {persona['location']}. "
        f"Your values: {', '.join(persona['values'])}. "
        f"Your confidence about the future is {persona['future_confidence']}/5 and you check super {persona['super_engagement'].lower()}. "
        "Respond as this individual."
    )
    hist = st.session_state.chat_history.get(name, [])
    answer = _ask_llm(_build_messages(intro, hist, question))
    st.session_state.chat_history.setdefault(name, []).append((question, answer))
    return answer

# ────────────────────────────────────────────────────────────────────────────────
# Q&A Section
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 💬 Ask a Question")
question = st.text_area("Enter your question:", value=st.session_state.get("question_input", ""))
ask_all = st.checkbox("Ask All Personas")

if st.button("Ask GPT"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Contacting personas…"):
            try:
                if ask_all:
                    for ent in filtered_personas:
                        ask_persona(ent["persona"], question)
                else:
                    if "selected_persona" not in st.session_state:
                        st.warning("Please select a persona.")
                        st.stop()
                    ask_persona(st.session_state.selected_persona, question)
            except Exception as e:
                st.error(f"Error: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Conversation history display
# ────────────────────────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🗂️ Conversation History")
    for persona_name, exchanges in st.session_state.chat_history.items():
        st.markdown(f"#### {persona_name}")
        for q, a in exchanges:
            st.markdown(f"<div style='background:#fafafa;padding:8px;border-left:4px solid #485cc7;border-radius:4px'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#e3f6d8;padding:8px;border-left:4px solid #43B02A;border-radius:4px'><strong>{persona_name}:</strong> {a}</div>", unsafe_allow_html=True)
            st.markdown("&nbsp;")
