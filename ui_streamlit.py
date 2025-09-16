import streamlit as st
import pandas as pd
import sys
import os

# Ensure the src directory is in the path to allow importing the core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.udmm2.udmm_v4_core import UDMMCore

# --- Page Configuration ---
st.set_page_config(
    page_title="UDMM v4 Interactive Chat",
    page_icon="🧠",
    layout="wide"
)

# --- Initialization ---
# Initialize the UDMM core and chat history in the session state
if "udmm_core" not in st.session_state:
    st.session_state.udmm_core = UDMMCore(dimensionality=5)
    print("Initialized new UDMMCore instance.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Layout ---
st.title("🧠 UDMM v4 Interactive Agent")
st.caption("An interactive chat with an agent based on the Dynamic Attractor Architecture.")

# Create two columns: one for the chat, one for the internal state
col1, col2 = st.columns([2, 1])

# --- Column 1: Chat Interface ---
with col1:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask the agent a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                # Process the input through the UDMM core
                agent_response = st.session_state.udmm_core.process_input(prompt)
                # The response from the placeholder is verbose, let's show it all
                st.markdown(agent_response)

        # Add agent response to chat history
        st.session_state.messages.append({"role": "assistant", "content": agent_response})

# --- Column 2: Internal State Visualization ---
with col2:
    st.header("Agent's Internal State")
    st.write("The agent's internal 'attractor state' evolves with each interaction.")

    # Get the current attractor state from the core
    attractor_state = st.session_state.udmm_core.attractor_dynamics.get_attractor_state()
    attractor_labels = st.session_state.udmm_core.attractor_dynamics.attractor_labels

    # Create a DataFrame for charting
    attractor_df = pd.DataFrame({
        'Strength': attractor_state
    }, index=attractor_labels)

    # Display the bar chart
    st.bar_chart(attractor_df)

    with st.expander("See Raw State Data"):
        st.write("Current Attractor State:")
        st.json(dict(zip(attractor_labels, attractor_state)))
        st.write("Current Intent Hierarchy:")
        st.json({k: dict(zip(attractor_labels, v)) for k, v in st.session_state.udmm_core.intent_manager.get_intent_hierarchy().items()})
