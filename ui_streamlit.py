# ui_streamlit.py
import streamlit as st
import requests
import os
import json
import time

# Use environment variable for API endpoint, with a default
API_URL = os.getenv("UDMM_API_URL", "http://127.0.0.1:8000")
AGENT_ID = "streamlit_agent"

st.set_page_config(layout="wide")
st.title("🧠 الوكيل الديناميكي الكامل (UDMM + AAR)")
st.caption("واجهة مستخدم تفاعلية لاختبار ومراقبة الوكيل الديناميكي")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {}
if "last_response" not in st.session_state:
    st.session_state.last_response = {}

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

        try:
            payload = {"message": prompt, "agent_id": AGENT_ID}
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            st.session_state.last_response = data

            agent_reply = data.get("response", "لم يتمكن الوكيل من الرد.")
            st.session_state.messages.append({"role": "assistant", "content": agent_reply})

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(agent_reply)

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ خطأ في الاتصال بواجهة برمجة التطبيقات: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "لا يمكنني الاتصال بنفسي الآن."})

with col2:
    st.header("📊 الحالة الداخلية للوكيل")

    if st.button("🔄 تحديث الحالة"):
        try:
            state_res = requests.get(f"{API_URL}/agent/{AGENT_ID}/state", timeout=10)
            state_res.raise_for_status()
            st.session_state.agent_state = state_res.json()
        except requests.exceptions.RequestException:
            st.warning("لم يتمكن من جلب الحالة. هل الوكيل يعمل؟")

    if st.session_state.last_response:
        st.session_state.agent_state = st.session_state.last_response.get("state", {})

    if st.session_state.agent_state:
        state = st.session_state.agent_state

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
        if st.session_state.last_response.get("intervention_occurred", False):
            st.success("✅ تم تفعيل تدخل AAR في الجولة الأخيرة!")

        with st.expander("عرض الحالة الكاملة (JSON)"):
            st.json(state)
    else:
        st.info("لم يتم استلام أي حالة من الوكيل بعد.")

    if st.button("🚨 إعادة تعيين الوكيل"):
        try:
            requests.post(f"{API_URL}/agent/{AGENT_ID}/reset", timeout=10)
            st.session_state.messages = []
            st.session_state.agent_state = {}
            st.session_state.last_response = {}
            st.success("تم إعادة تعيين الوكيل بنجاح!")
            st.rerun()
        except requests.exceptions.RequestException:
            st.error("فشل في إعادة تعيين الوكيل.")
