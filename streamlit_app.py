import streamlit as st
import json
import time
from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ────────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: dict[str, list[tuple[str, str]]] = {}
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []  # List of dicts for debate messages
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
        """Ensure every persona has the Stake-driven keys."""
        p.setdefault("future_confidence", 3)
        p.setdefault("family_support_received", False)
        p.setdefault("ideal_salary_for_comfort", 120_000)
        p.setdefault("budget_adjustments_6m", [])
        p.setdefault("super_engagement", "Unknown")
        p.setdefault("property_via_super_interest", "No")
        return p

    # Flatten personas for easier access in dropdowns
    flat_personas = []
    for group in data:
        for gender in ("male", "female"):
            if gender in group:
                p = group[gender]
                _patch(p)
                flat_personas.append({"persona": p, "segment": group["segment"], "id": f"{p['name']} ({group['segment']})"})
    
    return data, flat_personas

persona_data_raw, all_personas_flat = load_and_patch_data()

# Segment Summaries for the UI
segment_summaries = {
    "Next Generation Investors (18–24 years)": "Tech-native, socially-conscious starters focused on building asset bases early.",
    "Emerging Wealth Builders (25–34 years)": "Balancing house deposits, careers and investing; optimistic but wage-squeezed.",
    "Established Accumulators (35–49 years)": "Juggling family, mortgages and wealth growth; value efficiency and advice.",
    "Pre-Retirees (50–64 years)": "Capital-preservers planning retirement income; keen super watchers.",
    "Retirees (65+ years)": "Stability-seekers prioritising income and low volatility.",
}

# ────────────────────────────────────────────────────────────────────────────────
# OpenAI Client
# ────────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
def construct_system_prompt(persona, context_instruction=""):
    """Builds the persona definition for the LLM."""
    return (
        f"You are {persona['name']}, a {persona['age']}-year-old {persona['occupation']} from {persona['location']}. "
        f"Your values: {', '.join(persona['values'])}. "
        f"Your confidence about the future is {persona['future_confidence']}/5. "
        f"Bio/Narrative: {persona['narrative']}. "
        f"Risk Tolerance: {persona['behavioural_traits'].get('risk_tolerance')}. "
        f"{context_instruction}"
        "Speak in the first person. Keep responses conversational, under 100 words, and authentic to your Aussie context."
    )

def query_llm(messages, model="gpt-4o"):
    """Simple wrapper for OpenAI call."""
    try:
        completion = client.chat.completions.create(model=model, messages=messages)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ────────────────────────────────────────────────────────────────────────────────
# UI Layout
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foolish Persona Portal", layout="wide", page_icon="🃏")

# Custom CSS for chat bubbles
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

st.title("🧠 The Foolish Synthetic Audience")

# Top Level Intro
with st.expander("ℹ️ How to use this tool", expanded=False):
    st.write("This tool uses AI personas based on real Australian investor research to simulate feedback.")
    st.write("- **Individual Interview:** Deep dive with one persona.")
    st.write("- **Focus Group Debate:** Pit two personas against each other to test marketing hooks.")

# ────────────────────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🗣️ Individual Interview", "⚔️ Focus Group Debate"])

# ================================================================================
# TAB 1: INDIVIDUAL INTERVIEW (Existing Logic)
# ================================================================================
with tab1:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Select Persona")
        # Segment Filter
        segments = sorted(list({p["segment"] for p in all_personas_flat}))
        selected_segment = st.selectbox("Filter by Segment", ["All"] + segments)
        
        # Filter Logic
        filtered_list = all_personas_flat if selected_segment == "All" else [p for p in all_personas_flat if p["segment"] == selected_segment]
        
        # Display Mini Cards
        for entry in filtered_list:
            p = entry["persona"]
            with st.container():
                st.markdown(f"**{p['name']}** ({p['age']})")
                st.caption(f"{entry['segment']}")
                if st.button(f"Select {p['name']}", key=f"btn_{p['name']}"):
                    st.session_state.selected_persona = p
                    st.session_state.selected_segment = entry["segment"]
                st.divider()

    with col_right:
        if "selected_persona" in st.session_state:
            p = st.session_state.selected_persona
            
            # Profile Card
            st.markdown(f"""
            <div style="background:#fff;padding:20px;border:1px solid #ddd;border-radius:10px;">
                <h3 style="margin-top:0;color:#43B02A">{p['name']} <span style="font-size:0.6em;color:#666">({st.session_state.selected_segment})</span></h3>
                <p><strong>{p['occupation']}</strong> | <strong>${p['income']:,}</strong> | <strong>{p['location']}</strong></p>
                <p><em>"{p['narrative']}"</em></p>
                <p><strong>Goals:</strong> {', '.join(p['goals'][:2])}...</p>
            </div>
            """, unsafe_allow_html=True)

            # Chat Interface
            st.subheader("Ask a Question")
            user_input = st.text_area("Your Question:", height=100, key="single_chat_input")
            
            if st.button("Ask Persona", type="primary"):
                if not user_input:
                    st.warning("Please type a question.")
                else:
                    # Construct Prompt
                    hist = st.session_state.chat_history.get(p["name"], [])
                    sys_prompt = construct_system_prompt(p)
                    
                    messages = [{"role": "system", "content": sys_prompt}]
                    # Add simple history (last 2 turns)
                    for q, a in hist[-2:]:
                        messages.append({"role": "user", "content": q})
                        messages.append({"role": "assistant", "content": a})
                    messages.append({"role": "user", "content": user_input})

                    with st.spinner(f"{p['name']} is thinking..."):
                        answer = query_llm(messages)
                    
                    # Save History
                    if p["name"] not in st.session_state.chat_history:
                        st.session_state.chat_history[p["name"]] = []
                    st.session_state.chat_history[p["name"]].append((user_input, answer))
                    st.rerun()

            # Display History
            if p["name"] in st.session_state.chat_history:
                st.markdown("#### Conversation History")
                for q, a in reversed(st.session_state.chat_history[p["name"]]):
                    st.markdown(f"<div class='chat-bubble user-bubble'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-bubble bot-bubble'><strong>{p['name']}:</strong> {a}</div>", unsafe_allow_html=True)
        else:
            st.info("👈 Select a persona from the left menu to start.")


# ================================================================================
# TAB 2: FOCUS GROUP DEBATE (New Feature)
# ================================================================================
with tab2:
    st.header("⚔️ Marketing Focus Group")
    st.markdown("Simulate a discussion between two different investors to stress-test your marketing copy.")

    # 1. Setup Controls
    c1, c2, c3 = st.columns(3)
    
    # Dropdown Options
    persona_options = {p["id"]: p["persona"] for p in all_personas_flat}
    
    with c1:
        p1_key = st.selectbox("Select Participant 1", options=list(persona_options.keys()), index=0)
    with c2:
        p2_key = st.selectbox("Select Participant 2", options=list(persona_options.keys()), index=1)
    with c3:
        debate_mode = st.radio("Conversation Mode", ["Natural Discussion", "Adversarial (Bull vs Bear)"])

    marketing_topic = st.text_area("Marketing Headline / Email Subject / Topic", 
                                   value="Subject: 3 AI Stocks better than Nvidia. Urgent Buy Alert!",
                                   help="Paste the copy you want them to react to.")
    
    start_debate = st.button("🚀 Start Focus Group", type="primary")

    # 2. Debate Logic
    if start_debate and marketing_topic:
        st.session_state.debate_history = [] # Reset
        persona_a = persona_options[p1_key]
        persona_b = persona_options[p2_key]

        # Define Roles based on Mode
        if debate_mode == "Adversarial (Bull vs Bear)":
            role_a = "You are generally optimistic and curious. Look for the potential upside."
            role_b = "You are highly skeptical. Look for the risks, catch, or clickbait nature of this."
        else:
            role_a = "React naturally based on your defined personality and risk profile."
            role_b = "React naturally based on your defined personality and risk profile."

        # Initial Prompt
        context_msg = f"We are discussing this marketing message: '{marketing_topic}'."
        
        # Container for live chat
        chat_container = st.container()

        # --- TURN 1: Persona A ---
        with chat_container:
            st.markdown(f"**Topic:** *{marketing_topic}*")
            st.divider()
            
            # Persona A speaks
            sys_a = construct_system_prompt(persona_a, f"{role_a} You are speaking to {persona_b['name']}.")
            msg_a = query_llm([
                {"role": "system", "content": sys_a},
                {"role": "user", "content": f"{context_msg} What is your immediate reaction?"}
            ])
            st.session_state.debate_history.append({"name": persona_a["name"], "text": msg_a, "color": "#e3f6d8"})
            st.markdown(f"**{persona_a['name']}**: {msg_a}")
            time.sleep(1) # UX Pause

            # --- TURN 2: Persona B responds ---
            sys_b = construct_system_prompt(persona_b, f"{role_b} You are speaking to {persona_a['name']}.")
            msg_b = query_llm([
                {"role": "system", "content": sys_b},
                {"role": "user", "content": f"{context_msg} {persona_a['name']} just said: '{msg_a}'. Respond to them."}
            ])
            st.session_state.debate_history.append({"name": persona_b["name"], "text": msg_b, "color": "#f0f2f6"})
            st.markdown(f"**{persona_b['name']}**: {msg_b}")
            time.sleep(1)

            # --- TURN 3: Persona A responds back ---
            msg_a_2 = query_llm([
                {"role": "system", "content": sys_a},
                {"role": "user", "content": f"The discussion is about '{marketing_topic}'. Previous history: You said '{msg_a}'. {persona_b['name']} replied: '{msg_b}'. Respond to their point."}
            ])
            st.session_state.debate_history.append({"name": persona_a["name"], "text": msg_a_2, "color": "#e3f6d8"})
            st.markdown(f"**{persona_a['name']}**: {msg_a_2}")

            # --- MODERATOR SUMMARY ---
            st.divider()
            st.subheader("📊 Moderator Insight")
            with st.spinner("Analyzing discussion..."):
                transcript = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])
                
                mod_prompt = f"""
                Analyze the following focus group discussion about the marketing hook: "{marketing_topic}".
                
                Transcript:
                {transcript}
                
                Provide 3 bullet points on:
                1. The main objection raised.
                2. What resonated (if anything).
                3. A suggested tweak to the copy to address the skepticism.
                """
                
                summary = query_llm([{"role": "system", "content": "You are a senior marketing strategist."}, 
                                     {"role": "user", "content": mod_prompt}])
                st.info(summary)
