# ui_streamlit.py
import streamlit as st
import os
import time

# Import the agent directly
from src.udmm2.agent import UDMM_Agent

AGENT_ID = "standalone_streamlit_agent"

st.set_page_config(layout="wide")
st.title("🧠 الوكيل الديناميكي الكامل (UDMM + AAR) - نسخة مستقلة")
st.caption("واجهة مستخدم تفاعلية تعمل كتطبيق مستقل")

# Initialize the agent in session state if it doesn't exist
if "agent" not in st.session_state:
    st.session_state.agent = UDMM_Agent(AGENT_ID)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_intervention" not in st.session_state:
    st.session_state.last_intervention = False

# Retrieve the agent from session state
agent = st.session_state.agent

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 المحادثة")
    chat_container = st.container(height=500)
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("أرسل رسالة إلى الوكيل...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generate response directly from the agent object
        agent_reply, intervention_occurred = agent.generate_response(prompt)
        st.session_state.last_intervention = intervention_occurred

        st.session_state.messages.append({"role": "assistant", "content": agent_reply})

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(agent_reply)

        # We need to rerun to update the state display on the right
        st.rerun()

with col2:
    st.header("📊 الحالة الداخلية للوكيل")

    # The state is now always available directly from the agent object
    state = agent.get_detailed_state()

    if state:
        st.subheader("الجسم الرقمي")
        body_state = state.get("body", [0, 0, 0])
        st.slider("⚡ الطاقة (Energy)", 0.0, 1.0, float(body_state[0]))
        st.slider("🔥 الإثارة (Arousal)", 0.0, 1.0, float(body_state[1]))

        st.subheader("النظام الوجداني")
        affect_state = state.get("affect", {})
        st.slider("❤️ الوجدان الخام (A_r)", 0.0, 1.0, float(affect_state.get("A_r", 0)))
        st.metric("التكافؤ (Valence)", f"{affect_state.get('valence', 0):.2f}")
        st.metric("تقلب الوجدان", f"{affect_state.get('variance', 0):.3f}")

        st.subheader("وكيل التحكم (Meta-Agent)")
        meta_state = state.get("meta", {})
        st.slider("🧠 التوتر المعلوماتي (IT)", 0.0, 1.0, float(meta_state.get("IT", 0)))
        st.slider("🔀 تباعد KL", 0.0, 1.0, float(meta_state.get("KL_divergence", 0)))

        st.subheader("الجاذب الافتراضي")
        st.info(f"**الحوض الحالي:** `{state.get('attractor', 'غير معروف')}`")

        # Display last intervention if any
        if st.session_state.last_intervention:
            st.success("✅ تم تفعيل تدخل AAR في الجولة الأخيرة!")

        with st.expander("عرض الحالة الكاملة (JSON)"):
            st.json(state)
    else:
        st.info("لم يتم استلام أي حالة من الوكيل بعد.")

    if st.button("🚨 إعادة تعيين الوكيل"):
        st.session_state.agent = UDMM_Agent(AGENT_ID)
        st.session_state.messages = []
        st.session_state.last_intervention = False
        st.success("تم إعادة تعيين الوكيل بنجاح!")
        st.rerun()
