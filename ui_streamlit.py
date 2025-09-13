# ui_streamlit.py
import streamlit as st, requests, os, json, time
API = os.getenv("UDMM_API", "http://127.0.0.1:8000")

st.title("UDMM LLM Agent (Streamlit UI)")

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([3,1])
with col1:
    for m in st.session_state.messages:
        st.write(m)

    prompt = st.text_input("اسأل الوكيل أو علّمه:")
    if st.button("إرسال"):
        if prompt:
            res = requests.post(f"{API}/ask", json={"question": prompt}).json()
            st.session_state.messages.append({"user": prompt})
            st.session_state.messages.append({"agent": res.get("response")})
            # if teach request detected
            if "[TEACH_ASSIST:" in res.get("response", ""):
                # parse teach question naive
                start = res["response"].find("[TEACH_ASSIST:")
                teach_text = res["response"][start:]
                st.session_state.teach_request = teach_text

with col2:
    st.subheader("حالة")
    try:
        st.write(requests.get(f"{API}/status").json())
    except Exception as e:
        st.error("لا يمكن الاتصال بالـ API")
