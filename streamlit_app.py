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
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADING & SEGMENT INFO
# ────────────────────────────────────────────────────────────────────────────────
segment_summaries = {
    "Next Generation Investors (18–24 years)": "Tech‑native, socially‑conscious starters focused on building asset bases early.",
    "Emerging Wealth Builders (25–34 years)": "Balancing house deposits, careers and investing; optimistic but wage‑squeezed.",
    "Established Accumulators (35–49 years)": "Juggling family, mortgages and wealth growth; value efficiency and advice.",
    "Pre‑Retirees (50–64 years)": "Capital‑preservers planning retirement income; keen super watchers.",
    "Retirees (65+ years)": "Stability‑seekers prioritising income and low volatility.",
}

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
        p.setdefault("income", 80000)
        p.setdefault("goals", [])
        p.setdefault("behavioural_traits", {"risk_tolerance": "Moderate"})
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
        clean = raw
        if "3. THE" in raw:
            parts = raw.split("3. THE")
            if len(parts) > 1:
                clean = parts[1].split("\n", 1)[-1].strip()
        elif "REWRITE:" in raw:
            clean = raw.split("REWRITE:")[-1].strip()
        
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
        <p>This tool uses a <strong>Hybrid AI Architecture</strong>: OpenAI (GPT-4o) for persona simulation and Google Gemini (3.0 Flash) for strategic analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🗣️ Individual Interview", "⚔️ Focus Group Debate"])

# ================================================================================
# TAB 1: INDIVIDUAL INTERVIEW (RESTORED FUNCTIONALITY)
# ================================================================================
with tab1:
    # 1. Segment Filter & Cheat Sheet
    segments = sorted(list({p["segment"] for p in all_personas_flat}))
    selected_segment = st.selectbox("Filter by Segment", ["All"] + segments)
    
    if selected_segment == "All":
        with st.expander("🔍 Segment Cheat Sheet"):
            for seg, blurb in segment_summaries.items():
                st.markdown(f"**{seg}**\n{blurb}\n")
    elif selected_segment in segment_summaries:
        with st.expander("🔍 Segment Overview", expanded=True):
            st.write(segment_summaries[selected_segment])

    filtered_list = all_personas_flat if selected_segment == "All" else [p for p in all_personas_flat if p["segment"] == selected_segment]

    # 2. Persona Grid
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

    # 3. Detailed Profile View
    if "selected_persona" in st.session_state:
        p = st.session_state.selected_persona
        seg = st.session_state.selected_segment

        st.markdown("---")
        # RESTORED: Rich Profile HTML
        st.markdown(
            f"""
            <div style="background:#e3f6d8;padding:20px;border-left:6px solid #43B02A;border-radius:10px">
                <h4 style="margin-top:0">{p['name']} <span style="font-weight:normal">({seg})</span></h4>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <p><strong>Age:</strong> {p.get('age')}</p>
                        <p><strong>Location:</strong> {p.get('location')}</p>
                        <p><strong>Occupation:</strong> {p.get('occupation')}</p>
                    </div>
                    <div>
                        <p><strong>Income:</strong> ${p.get('income', 0):,}</p>
                        <p><strong>Risk:</strong> {p.get('behavioural_traits', {}).get('risk_tolerance', 'Unknown')}</p>
                        <p><strong>Confidence:</strong> {p.get('future_confidence')}/5</p>
                    </div>
                </div>
                <hr style="margin:10px 0; border-top: 1px solid #ccc;">
                <p><strong>Values:</strong> {', '.join(p.get('values', []))}</p>
                <p><strong>Goals:</strong> {'; '.join(p.get('goals', []))}</p>
                <p><strong>Narrative:</strong> {p['narrative']}</p>
                <p><strong>Recent Budget Cuts:</strong> {', '.join(p.get('budget_adjustments_6m', ['None']))}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # RESTORED: Suggested Questions
        st.markdown("### 💡 Suggested Questions")
        if p.get("suggestions"):
            cols_s = st.columns(len(p["suggestions"]))
            for idx, s in enumerate(p["suggestions"]):
                if cols_s[idx].button(f"Ask: {s[:30]}...", key=f"sugg_{idx}"):
                    st.session_state.question_input = s
                    st.rerun()
        else:
            st.caption("No specific suggestions for this persona.")

        # 4. Q&A Interface
        st.markdown("### 💬 Interaction")
        user_input = st.text_area("Enter your question:", value=st.session_state.question_input, key="q_input")
        ask_all = st.checkbox("Ask ALL visible personas (Batch Test)")

        if st.button("Ask Persona(s)", type="primary"):
            if not user_input:
                st.warning("Please enter a question.")
            else:
                target_list = filtered_list if ask_all else [{"persona": p}]
                
                with st.spinner(f"Interviewing {len(target_list)} persona(s)..."):
                    for target in target_list:
                        tp = target["persona"]
                        
                        sys_msg = (
                            f"You are {tp['name']}, a {tp['age']}-year-old {tp['occupation']}. "
                            f"Bio: {tp['narrative']}. Values: {', '.join(tp['values'])}. "
                            f"Income: ${tp.get('income')}. Risk Tolerance: {tp.get('behavioural_traits', {}).get('risk_tolerance')}. "
                            "Respond in character. Be conversational."
                        )
                        
                        hist = st.session_state.chat_history.get(tp["name"], [])
                        messages = [{"role": "system", "content": sys_msg}]
                        for q, a in hist[-3:]:
                            messages.append({"role": "user", "content": q})
                            messages.append({"role": "assistant", "content": a})
                        messages.append({"role": "user", "content": user_input})

                        ans = query_openai(messages) # Still using GPT-4o for individual chats
                        st.session_state.chat_history.setdefault(tp["name"], []).append((user_input, ans))
                
                st.success("Responses received!")
                st.rerun()

        # Display History
        if ask_all:
             st.markdown("#### Batch Results")
             for target in filtered_list:
                 tp = target["persona"]
                 if tp["name"] in st.session_state.chat_history:
                     last_q, last_a = st.session_state.chat_history[tp["name"]][-1]
                     if last_q == user_input:
                         st.markdown(f"**{tp['name']}:** {last_a}")
                         st.divider()
        else:
            if p["name"] in st.session_state.chat_history:
                st.markdown("#### Conversation History")
                for q, a in reversed(st.session_state.chat_history[p["name"]]):
                    st.markdown(f"<div class='chat-bubble user-bubble'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-bubble bot-bubble'><strong>{p['name']}:</strong> {a}</div>", unsafe_allow_html=True)


# ================================================================================
# TAB 2: FOCUS GROUP DEBATE (TOPIC AGNOSTIC)
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
        key="marketing_topic",
        height=150 
    )
    
    if st.button("🚀 Start Focus Group", type="primary"):
        st.session_state.debate_history = []
        st.session_state.suggested_rewrite = ""
        st.session_state.campaign_assets = None
        
        p_a = persona_options[p1_key]
        p_b = persona_options[p2_key]

        # ────────────────────────────────────────────────────────────────────────
        # DYNAMIC PERSONA PROMPTS (TOPIC AGNOSTIC)
        # ────────────────────────────────────────────────────────────────────────
        base_instruction = (
            "IMPORTANT: This is a simulation for marketing research. "
            "You are roleplaying a specific persona. Do NOT sound like a generic AI. "
            "Use your specific vocabulary, age-appropriate slang, and worldview."
        )

        # ROLE A: THE BELIEVER (Optimistic, Opportunity-Seeking)
        role_a = (
            f"ROLE: You are {p_a['name']}, a {p_a['age']}-year-old {p_a['occupation']}. "
            f"BIO: {p_a['narrative']} "
            f"VALUES: {', '.join(p_a['values'])}. \n"
            f"CONTEXT: In this focus group, you represent 'The Believer'. "
            f"You are naturally optimistic and looking for opportunities to grow your wealth. "
            f"You WANT this marketing message to be true. You focus on the potential upside and the promise. "
            f"React to the text based on your specific life stage and goals. Defend the idea against skepticism."
        )

        # ROLE B: THE SKEPTIC (Cautious, Risk-Averse)
        role_b = (
            f"ROLE: You are {p_b['name']}, a {p_b['age']}-year-old {p_b['occupation']}. "
            f"BIO: {p_b['narrative']} "
            f"VALUES: {', '.join(p_b['values'])}. \n"
            f"CONTEXT: In this focus group, you represent 'The Skeptic'. "
            f"You are critical of marketing hype and naturally risk-averse. "
            f"You focus on the downsides, the missing details, and the risks. "
            f"You doubt the claims made in the text. Call out anything that sounds too good to be true."
        )

        chat_container = st.container()
        
        with chat_container:
            st.markdown(f"**Topic:** *{marketing_topic}*")
            st.divider()
            
            # 1. BULL SPEAKS
            msg_a = query_openai([
                {"role": "system", "content": base_instruction + "\n" + role_a},
                {"role": "user", "content": f"React to this marketing text: '{marketing_topic}'."}
            ])
            st.session_state.debate_history.append({"name": p_a["name"], "text": msg_a})
            st.markdown(f"**{p_a['name']} (The Believer)**: {msg_a}")
            time.sleep(1)

            # 2. BEAR RESPONDS
            msg_b = query_openai([
                {"role": "system", "content": base_instruction + "\n" + role_b},
                {"role": "user", "content": f"The marketing text is '{marketing_topic}'. {p_a['name']} just said: '{msg_a}'. Give them a reality check."}
            ])
            st.session_state.debate_history.append({"name": p_b["name"], "text": msg_b})
            st.markdown(f"**{p_b['name']} (The Skeptic)**: {msg_b}")
            time.sleep(1)

            # 3. BULL RETORTS
            msg_a_2 = query_openai([
                {"role": "system", "content": base_instruction + "\n" + role_a},
                {"role": "user", "content": f"You just got critiqued. {p_b['name']} said: '{msg_b}'. Explain why you think THIS opportunity is worth it."}
            ])
            st.session_state.debate_history.append({"name": p_a["name"], "text": msg_a_2})
            st.markdown(f"**{p_a['name']} (The Believer)**: {msg_a_2}")

            # 4. MODERATOR (GEMINI 3 FLASH)
            st.divider()
            st.subheader("📊 Strategic Analysis (Powered by Gemini 3 Flash)")
            with st.spinner("Gemini 3 is analyzing the psychology..."):
                transcript = "\n".join([f"{x['name']}: {x['text']}" for x in st.session_state.debate_history])
                
                mod_prompt = f"""
                You are a legendary Direct Response Copywriter (Motley Fool Style).
                
                TRANSCRIPT OF DEBATE:
                {transcript}
                
                MARKETING HOOK: "{marketing_topic}"
                
                TASK:
                1. THE "REAL" WHY: Analyze the deep emotional driver (e.g., Greed, Security, Freedom).
                2. THE TRUST GAP: Analyze the specific logical objection raised by the Skeptic.
                3. THE "FOOLISH" REWRITE:
                   - Write a new **Subject Line** AND a **Killer Email Opening** (2-3 sentences).
                   - Address the emotional driver while neutralizing the objection.
                   - Style: Story-driven, Personal, Contrarian.
                   - Output format: 
                     **Subject:** [Your Subject]
                     
                     **Body:** [Your Body]
                """
                
                summary = query_gemini(mod_prompt)
                st.info(summary)
                st.session_state.suggested_rewrite = summary 

    # ────────────────────────────────────────────────────────────────────────
    # FEEDBACK LOOP & CAMPAIGN GENERATOR
    # ────────────────────────────────────────────────────────────────────────
    if st.session_state.debate_history and st.session_state.suggested_rewrite:
        st.markdown("---")
        
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("### 🔄 Iterate")
            st.caption("Re-run debate with this new hook.")
            st.button("Test Rewrite", on_click=apply_rewrite)
            
        with col_b:
            st.markdown("### 📢 Production")
            st.caption("Turn this insight into ad assets.")
            if st.button("✨ Generate Campaign Assets", type="secondary"):
                with st.spinner("Briefing the specialist copywriters..."):
                    brief = st.session_state.suggested_rewrite
                    
                    campaign_prompt = f"""
                    You are a Full-Stack Marketing Team. Use the following STRATEGIC INSIGHT to generate campaign assets.
                    
                    STRATEGIC INSIGHT:
                    {brief}
                    
                    TASKS:
                    1. GOOGLE SEARCH AD: 
                       - 3x Headlines (30 chars max).
                       - 2x Descriptions (90 chars max).
                       - Focus: High CTR, Curiosity.
                       
                    2. FACEBOOK/META AD:
                       - Primary Text (The "Scroll Stopper"): Use a Pattern Interrupt or Story hook based on the insight.
                       - Headline: Short, punchy (5 words max).
                       
                    3. SALES PAGE HERO:
                       - H1 Headline.
                       - Subheadline (The Promise).
                       - CTA Button Copy.
                       
                    OUTPUT FORMAT:
                    Use Markdown Headers for each section (e.g. ### Google Ads).
                    """
                    
                    assets = query_gemini(campaign_prompt)
                    st.session_state.campaign_assets = assets
        
        if "campaign_assets" in st.session_state and st.session_state.campaign_assets:
            st.divider()
            st.subheader("📦 Campaign Asset Pack")
            st.markdown(st.session_state.campaign_assets)
