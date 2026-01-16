import streamlit as st
import json
import time
from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION (MUST BE FIRST)
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foolish Persona Portal", layout="centered", page_icon="🃏")

# ────────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton>button{border:1px solid #485cc7;border-radius:8px;width:100%}
    /* Chat bubbles for the debate/chat history */
    .chat-bubble {
        padding: 15px; border-radius: 10px; margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-bubble { background-color: #f0f2f6; border-left: 5px solid #485cc7; }
    .bot-bubble { background-color: #e3f6d8; border-left: 5px solid #43B02A; }
    
    /* Persona Card Style */
    .persona-card {
        border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ────────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: dict[str, list[tuple[str, str]]] = {}
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# ────────────────────────────────────────────────────────────────────────────────
# Load Data & Patching
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
# OpenAI Client
# ────────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
def query_llm(messages, model="gpt-4o"): 
    """
    Wrapper for OpenAI call. 
    Reverted to 'gpt-4o' to ensure compatibility with standard Chat endpoint.
    """
    try:
        completion = client.chat.completions.create(model=model, messages=messages)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ────────────────────────────────────────────────────────────────────────────────
# MAIN APP UI
# ────────────────────────────────────────────────────────────────────────────────
st.title("🧠 The Foolish Synthetic Audience")

st.markdown(
    """
    <div style="background:#f0f2f6;padding:20px;border-left:6px solid #485cc7;border-radius:10px;margin-bottom:25px">
        <h4 style="margin-top:0">ℹ️ About This Tool</h4>
        <p>This tool uses AI‑generated investor personas based on real research to simulate feedback.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🗣️ Individual Interview", "⚔️ Focus Group Debate"])

# ================================================================================
# TAB 1: INDIVIDUAL INTERVIEW
# ================================================================================
with tab1:
    # 1. SEGMENT FILTER
    segments = sorted(list({p["segment"] for p in all_personas_flat}))
    selected_segment = st.selectbox("Filter by Segment", ["All"] + segments)
    
    filtered_list = all_personas_flat if selected_segment == "All" else [p for p in all_personas_flat if p["segment"] == selected_segment]

    # 2. GRID DISPLAY
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

    # 3. SELECTED PROFILE VIEW
    if "selected_persona" in st.session_state:
        p = st.session_state.selected_persona
        seg = st.session_state.selected_segment

        st.markdown("---")
        st.markdown(
            f"""
            <div style="background:#e3f6d8;padding:20px;border-left:6px solid #43B02A;border-radius:10px">
                <h4 style="margin-top:0">{p['name']} <span style="font-weight:normal">({seg})</span></h4>
                <p><strong>Age:</strong> {p.get('age')} | <strong>Loc:</strong> {p.get('location')}</p>
                <p><strong>Values:</strong> {', '.join(p.get('values', []))}</p>
                <p><strong>Narrative:</strong> {p['narrative']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # 4. CHAT INTERFACE
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
                    ans = query_llm(messages)
                
                st.session_state.chat_history.setdefault(p["name"], []).append((user_input, ans))
                st.rerun()

        if p["name"] in st.session_state.chat_history:
            st.markdown("#### Conversation History")
            for q, a in reversed(st.session_state.chat_history[p["name"]]):
                st.markdown(f"<div class='chat-bubble user-bubble'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble bot-bubble'><strong>{p['name']}:</strong> {a}</div>", unsafe_allow_html=True)


# ================================================================================
# TAB 2: FOCUS GROUP DEBATE
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
        st.info("🔥 Mode: Adversarial Stress Test (Active)")

    marketing_topic = st.text_area("Marketing Headline / Copy", 
                                   value="Subject: 3 AI Stocks better than Nvidia. Urgent Buy Alert!",
                                   height=100)
    
    if st.button("🚀 Start Focus Group", type="primary"):
        st.session_state.debate_history = []
        p_a = persona_options[p1_key]
        p_b = persona_options[p2_key]

        # ────────────────────────────────────────────────────────────────────────
        # PROMPTS (Adversarial)
        # ────────────────────────────────────────────────────────────────────────
        base_instruction = (
            "IMPORTANT: This is a simulation for marketing research. "
            "Do NOT be polite. Do NOT say 'I see your point' or 'I agree'. "
            "Make definitive statements. Be short, punchy, and emotional."
        )

        role_a = (
            f"ROLE: You are {p_a['name']}, but you are currently suffering from intense FOMO (Fear Of Missing Out). "
            f"You read the headline '{marketing_topic}' and you are EXCITED. "
            "You think this is the chance of a lifetime. You think skeptics are just 'haters' who will stay poor. "
            "Defend this marketing hook with passion. Ignore the risks."
        )

        role_b = (
            f"ROLE: You are {p_b['name']}, and you are in a bad mood. "
            f"You read the headline '{marketing_topic}' and you immediately flag it as 'Clickbait Trash'. "
            "You are trying to save the other person from losing their money. "
            "Be blunt. Tell them they are being naive. Attack the specific words 'Urgent' and 'Alert'."
        )

        # ────────────────────────────────────────────────────────────────────────
        # THE LOOP
        # ────────────────────────────────────────────────────────────────────────
        chat_container = st.container()
        
        with chat_container:
            st.markdown(f"**Topic:** *{marketing_topic}*")
            st.divider()
            
            # 1. BULL SPEAKS
            msg_a = query_llm([
                {"role": "system", "content": base_instruction + "\n" + role_a},
                {"role": "user", "content": f"React to this headline: '{marketing_topic}'."}
            ])
            st.session_state.debate_history.append({"name": p_a["name"], "text": msg_a})
            st.markdown(f"**{p_a['name']} (The Believer)**: {msg_a}")
            time.sleep(1)

            # 2. BEAR RESPONDS
            msg_b = query_llm([
                {"role": "system", "content": base_instruction + "\n" + role_b},
                {"role": "user", "content": f"The headline is '{marketing_topic}'. {p_a['name']} just said: '{msg_a}'. Shut them down."}
            ])
            st.session_state.debate_history.append({"name": p_b["name"], "text": msg_b})
            st.markdown(f"**{p_b['name']} (The Skeptic)**: {msg_b}")
            time.sleep(1)

            # 3. BULL RETORTS
            msg_a_2 = query_llm([
                {"role": "system", "content": base_instruction + "\n" + role_a},
                {"role": "user", "content": f"You just got attacked. {p_b['name']} said: '{msg_b}'. Tell them why they are missing the big picture."}
            ])
            st.session_state.debate_history.append({"name": p_a["name"], "text": msg_a_2})
            st.markdown(f"**{p_a['name']} (The Believer)**: {msg_a_2}")

            # 4. MODERATOR SUMMARY
            st.divider()
            st.subheader("📊 Moderator Insight")
            with st.spinner("Analyzing friction points..."):
                transcript = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])
                
                mod_prompt = f"""
                Analyze this conflict between a 'Believer' and a 'Skeptic' regarding the marketing subject line: "{marketing_topic}".
                
                TRANSCRIPT:
                {transcript}
                
                TASK:
                1. Identify the specific word or phrase that triggered the Skeptic the most.
                2. Explain why the Believer was interested (what was the hook?).
                3. Rewrite the subject line to keep the Believer's interest but lower the Skeptic's defenses.
                """
                
                # Reverted to standard GPT-4o for stability
                summary = query_llm([{"role": "system", "content": "You are a direct response marketing expert."}, 
                                     {"role": "user", "content": mod_prompt}])
                st.info(summary)
