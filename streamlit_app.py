import streamlit as st
import json
import time
from openai import OpenAI
import google.generativeai as genai

# ────────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foolish Persona Portal", layout="centered", page_icon="🃏")

# ────────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton>button{border:1px solid #485cc7;border-radius:8px;width:100%}
    .chat-bubble {
        padding: 15px; border-radius: 10px; margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-bubble { background-color: #f0f2f6; border-left: 5px solid #485cc7; }
    .bot-bubble { background-color: #e3f6d8; border-left: 5px solid #43B02A; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "marketing_topic" not in st.session_state:
    st.session_state.marketing_topic = "Subject: 3 AI Stocks better than Nvidia. Urgent Buy Alert!"
if "suggested_rewrite" not in st.session_state:
    st.session_state.suggested_rewrite = ""

# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_patch_data():
    with open("personas.json", "r", encoding="utf-8") as f:
        data = json.load(f)["personas"]

    def _patch(p: dict) -> dict:
        p.setdefault("future_confidence", 3)
        p.setdefault("family_support_received", False)
        p.setdefault("ideal_salary_for_comfort", 120_000)
        p.setdefault("budget_adjustments_6m", [])
        p.setdefault("super_engagement", "Unknown")
        p.setdefault("property_via_super_interest", "No")
        return p

    flat_personas = []
    for group in data:
        for gender in ("male", "female"):
            if gender in group:
                p = group[gender]
                _patch(p)
                flat_personas.append({"persona": p, "segment": group["segment"], "id": f"{p['name']} ({group['segment']})"})
    
    return data, flat_personas

persona_data_raw, all_personas_flat = load_and_patch_data()

# ────────────────────────────────────────────────────────────────────────────────
# AI CLIENTS (HYBRID ARCHITECTURE)
# ────────────────────────────────────────────────────────────────────────────────

# 1. OPENAI (For Personas/Drama)
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def query_openai(messages, model="gpt-4o"): 
    try:
        completion = client_openai.chat.completions.create(model=model, messages=messages)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# 2. GEMINI (For Moderator/Copywriting)
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("🚨 Missing GOOGLE_API_KEY in secrets.toml")

def query_gemini(prompt):
    """
    Uses Gemini 3 Flash Preview for the Moderator.
    Falls back to OpenAI if Gemini fails.
    """
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error ({str(e)}). \n\nFallback Analysis:\n" + \
               query_openai([{"role": "user", "content": prompt}])

# CALLBACK FUNCTION
def apply_rewrite():
    raw = st.session_state.suggested_rewrite
    if raw:
        # Improved parsing: Look for the FINAL bold line, which is usually the headline
        if "**" in raw:
            parts = raw.split("**")
            # Usually the last bold item is the headline in the new format
            clean = parts[-2] if len(parts) >= 2 else raw
        else:
            clean = raw
        
        # Cleanup
        clean = clean.strip(" :.\n\"")
        st.session_state.marketing_topic = clean
        st.session_state.debate_history = [] 

# ────────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────────────────────────────────────
st.title("🧠 The Foolish Synthetic Audience")

st.markdown(
    """
    <div style="background:#f0f2f6;padding:20px;border-left:6px solid #485cc7;border-radius:10px;margin-bottom:25px">
        <h4 style="margin-top:0">ℹ️ About This Tool</h4>
        <p>This tool uses a <strong>Hybrid AI Architecture</strong>: OpenAI (GPT-4o) for persona simulation (Drama) and Google Gemini (3.0 Flash) for strategic analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🗣️ Individual Interview", "⚔️ Focus Group Debate"])

# ================================================================================
# TAB 1: INDIVIDUAL INTERVIEW
# ================================================================================
with tab1:
    segments = sorted(list({p["segment"] for p in all_personas_flat}))
    selected_segment = st.selectbox("Filter by Segment", ["All"] + segments)
    
    filtered_list = all_personas_flat if selected_segment == "All" else [p for p in all_personas_flat if p["segment"] == selected_segment]

    st.markdown("### 👥 Select a Persona")
    cols = st.columns(3)
    
    for i, entry in enumerate(filtered_list):
        p = entry["persona"]
        with cols[i % 3]:
            with st.container():
                if p.get("image"):
                    st.image(p["image"], use_container_width=True)
                st.markdown(f"**{p['name']}**")
                st.caption(f"{entry['segment']}")
                if st.button("Select", key=f"sel_{p['name']}"):
                    st.session_state.selected_persona = p
                    st.session_state.selected_segment = entry["segment"]
                    st.rerun()

    if "selected_persona" in st.session_state:
        p = st.session_state.selected_persona
        seg = st.session_state.selected_segment

        st.markdown("---")
        st.markdown(
            f"""
            <div style="background:#e3f6d8;padding:20px;border-left:6px solid #43B02A;border-radius:10px">
                <h4 style="margin-top:0">{p['name']} <span style="font-weight:normal">({seg})</span></h4>
                <p><strong>Age:</strong> {p.get('age')} | <strong>Loc:</strong> {p.get('location')}</p>
                <p><strong>Narrative:</strong> {p['narrative']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### 💬 Ask a Question")
        user_input = st.text_area("Enter your question:", key="q_input")
        
        if st.button("Ask Persona", type="primary"):
            if not user_input:
                st.warning("Please enter a question.")
            else:
                sys_msg = (
                    f"You are {p['name']}, a {p['age']}-year-old {p['occupation']}. "
                    f"Bio: {p['narrative']}. Values: {', '.join(p['values'])}. "
                    "Respond in character. Be conversational."
                )
                
                hist = st.session_state.chat_history.get(p["name"], [])
                messages = [{"role": "system", "content": sys_msg}]
                for q, a in hist[-3:]:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user", "content": user_input})

                with st.spinner("Typing..."):
                    ans = query_openai(messages)
                
                st.session_state.chat_history.setdefault(p["name"], []).append((user_input, ans))
                st.rerun()

        if p["name"] in st.session_state.chat_history:
            st.markdown("#### Conversation History")
            for q, a in reversed(st.session_state.chat_history[p["name"]]):
                st.markdown(f"<div class='chat-bubble user-bubble'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble bot-bubble'><strong>{p['name']}:</strong> {a}</div>", unsafe_allow_html=True)


# ================================================================================
# TAB 2: FOCUS GROUP DEBATE (HYBRID)
# ================================================================================
with tab2:
    st.header("⚔️ Marketing Focus Group")
    st.markdown("Pit two investors against each other to stress-test your copy.")

    c1, c2, c3 = st.columns(3)
    persona_options = {p["id"]: p["persona"] for p in all_personas_flat}
    
    with c1:
        p1_key = st.selectbox("Participant 1 (The Believer)", options=list(persona_options.keys()), index=0)
    with c2:
        p2_key = st.selectbox("Participant 2 (The Skeptic)", options=list(persona_options.keys()), index=1)
    with c3:
        st.info("🔥 Mode: Adversarial Stress Test")

    marketing_topic = st.text_area(
        "Marketing Headline / Copy", 
        key="marketing_topic"
    )
    
    if st.button("🚀 Start Focus Group", type="primary"):
        st.session_state.debate_history = []
        st.session_state.suggested_rewrite = ""
        
        p_a = persona_options[p1_key]
        p_b = persona_options[p2_key]

        # ────────────────────────────────────────────────────────────────────────
        # EMOTIONAL PROMPTS (The "Hot-Take" Logic)
        # ────────────────────────────────────────────────────────────────────────
        base_instruction = (
            "IMPORTANT: This is a simulation for marketing research. "
            "You are roleplaying a real investor. Be conversational, not a caricature. "
            "Do NOT be polite just to be nice. Speak as if commenting on Reddit."
        )

        role_a = (
            f"ROLE: You are {p_a['name']}. "
            f"CONTEXT: You missed the Nvidia rally and you feel a deep, anxious FOMO. "
            f"You desperately WANT this headline to be true because you need a 'second chance'. "
            "You feel defensive when people question it. You are trying to convince yourself as much as the other person."
        )

        role_b = (
            f"ROLE: You are {p_b['name']}. "
            f"CONTEXT: You are weary of hype. You've seen friends lose money on 'Hot Tips' before. "
            f"You aren't angry, just cynical. You view this headline as a probable trap. "
            "You are trying to be the voice of reason. You are calm but firm."
        )

        # ────────────────────────────────────────────────────────────────────────
        # THE DEBATE LOOP (POWERED BY OPENAI)
        # ────────────────────────────────────────────────────────────────────────
        chat_container = st.container()
        
        with chat_container:
            st.markdown(f"**Topic:** *{marketing_topic}*")
            st.divider()
            
            # 1. BULL SPEAKS
            msg_a = query_openai([
                {"role": "system", "content": base_instruction + "\n" + role_a},
                {"role": "user", "content": f"React to this headline: '{marketing_topic}'."}
            ])
            st.session_state.debate_history.append({"name": p_a["name"], "text": msg_a})
            st.markdown(f"**{p_a['name']} (The Believer)**: {msg_a}")
            time.sleep(1)

            # 2. BEAR RESPONDS
            msg_b = query_openai([
                {"role": "system", "content": base_instruction + "\n" + role_b},
                {"role": "user", "content": f"The headline is '{marketing_topic}'. {p_a['name']} just said: '{msg_a}'. Give them a reality check."}
            ])
            st.session_state.debate_history.append({"name": p_b["name"], "text": msg_b})
            st.markdown(f"**{p_b['name']} (The Skeptic)**: {msg_b}")
            time.sleep(1)

            # 3. BULL RETORTS
            msg_a_2 = query_openai([
                {"role": "system", "content": base_instruction + "\n" + role_a},
                {"role": "user", "content": f"You just got critiqued. {p_b['name']} said: '{msg_b}'. Explain why you think THIS time is different."}
            ])
            st.session_state.debate_history.append({"name": p_a["name"], "text": msg_a_2})
            st.markdown(f"**{p_a['name']} (The Believer)**: {msg_a_2}")

            # ────────────────────────────────────────────────────────────────────
            # MODERATOR (POWERED BY GEMINI 3 FLASH PREVIEW)
            # ────────────────────────────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Strategic Analysis (Powered by Gemini 3 Flash)")
            with st.spinner("Gemini 3 is analyzing the psychology..."):
                transcript = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])
                
                # UPDATED PROMPT: STRICT COPYWRITING CONSTRAINTS
                mod_prompt = f"""
                You are a legendary Direct Response Copywriter (Motley Fool Style).
                
                TRANSCRIPT OF DEBATE:
                {transcript}
                
                MARKETING HOOK: "{marketing_topic}"
                
                TASK:
                1. THE "REAL" WHY: Analyze the deep emotional driver (e.g., Redemption, Status).
                2. THE TRUST GAP: Analyze the specific logical objection.
                3. THE "FOOLISH" REWRITE:
                   - Write a SUBJECT LINE (Max 15 words).
                   - STOP EXPLAINING. START SELLING.
                   - Hit the emotional trigger (Redemption) AND hint at the logical solution (The Mechanism) without giving it away.
                   - Do not use technical jargon. Use "Tease" logic.
                   - Output ONLY the final subject line in bold.
                """
                
                summary = query_gemini(mod_prompt)
                st.info(summary)
                st.session_state.suggested_rewrite = summary 

    # ────────────────────────────────────────────────────────────────────────
    # FEEDBACK LOOP BUTTON
    # ────────────────────────────────────────────────────────────────────────
    if st.session_state.debate_history and st.session_state.suggested_rewrite:
        st.markdown("---")
        st.markdown("### 🔄 Iterate")
        st.write("Test Gemini's suggestion?")
        
        col_a, col_b = st.columns([1, 4])
        with col_a:
            st.button("Test Rewrite", on_click=apply_rewrite)
