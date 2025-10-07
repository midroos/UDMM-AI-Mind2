# ui_streamlit.py
import streamlit as st
import requests
import os
from src.udmm2.simulation.hybrid_model import HybridUDMMSimulation
import matplotlib.pyplot as plt

# Get API endpoint from environment variable or use default
API = os.getenv("UDMM_API", "http://127.0.0.1:8000")

st.set_page_config(layout="wide")

st.title("UDMM LLM Agent (Streamlit UI)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define columns for layout
col1, col2 = st.columns([3, 1])

# --- Main Chat Column ---
with col1:
    # Display chat messages from history
    for message in st.session_state.messages:
        # Use a dictionary to map role to a user-friendly name if needed
        # For simplicity, we assume roles are 'user' and 'agent'
        role = message.get("user", "agent")
        content = message.get(role)
        with st.chat_message(role):
            st.markdown(content)

    # User input
    if prompt := st.chat_input("اسأل الوكيل أو علّمه:"):
        # Add user message to chat history
        st.session_state.messages.append({"user": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("agent"):
            message_placeholder = st.empty()
            try:
                res = requests.post(f"{API}/ask", json={"question": prompt}).json()
                response = res.get("response", "No response from API.")
                message_placeholder.markdown(response)
                st.session_state.messages.append({"agent": response})

                # Handle learning requests
                if "[TEACH_ASSIST:" in response:
                    start = response.find("[TEACH_ASSIST:")
                    st.session_state.teach_request = response[start:]

            except requests.exceptions.ConnectionError:
                message_placeholder.error("لا يمكن الاتصال بالـ API")

# --- Sidebar Column ---
with col2:
    st.subheader("حالة")
    try:
        status_info = requests.get(f"{API}/status").json()
        st.json(status_info)
    except requests.exceptions.ConnectionError:
        st.error("لا يمكن الاتصال بالـ API")

    # --- Hybrid Simulation Section ---
    with st.expander("المحاكاة الهجينة", expanded=False):
        if st.button("تشغيل المحاكاة"):
            with st.spinner("...جاري تشغيل المحاكاة"):
                # Define parameters and initial conditions for the simulation
                base_params = {
                    'eta': 0.3, 'alpha_IT': 0.5, 'beta_M': 0.4, 'gamma_M': 0.3,
                    'delta_C': 0.6, 'epsilon_C': 0.4, 'theta_C': 0.25, 'zeta_A': 0.5,
                    'k_alpha': 0.05, 'k_beta': 0.05, 'k_gamma': 0.05,
                    'structural_need': 0.4, 'phenomenal_need': 0.4, 'symbolic_need': 0.2
                }
                initial_conditions_y = [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.2]

                # Instantiate and run the simulation
                simulation = HybridUDMMSimulation(base_params)
                history = simulation.run_simulation(initial_conditions_y)

                # Generate and display the plot
                fig = simulation.plot_results(history)
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free up memory