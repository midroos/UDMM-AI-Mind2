import streamlit as st
import pandas as pd
import numpy as np

# Correctly import from the src directory
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
    # Use environment variables for LLM provider, defaulting to echo
    llm_provider = os.environ.get("LLM_PROVIDER", "echo")
    st.session_state.udmm_core = UDMMCore(dimensionality=5, llm_provider=llm_provider)
    print("Initialized new UDMMCore instance.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Layout ---
st.title("🧠 UDMM v4 Interactive Agent")
st.caption(f"An interactive chat with the UDMM v4 agent. (Using '{st.session_state.udmm_core.llm.provider}' LLM)")

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
                response_dict = st.session_state.udmm_core.process_input(prompt)
                agent_response_text = response_dict.get("response", "Error: No response text found.")
                st.markdown(agent_response_text)

        # Add agent response to chat history
        st.session_state.messages.append({"role": "assistant", "content": agent_response_text})

# --- Column 2: Internal State Visualization ---
with col2:
    st.header("Agent's Internal State")
    st.write("The agent's internal 'attractor state' evolves with each interaction.")

    # Get the current attractor state from the core
    attractor_state = st.session_state.udmm_core.attractor.get_state()
    attractor_labels = st.session_state.udmm_core.attractor.labels

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
        st.json({k: v.tolist() for k, v in st.session_state.udmm_core.intent.get_hierarchy().items()})
        st.write("Schemas in Memory:")
        st.write(f"{len(st.session_state.udmm_core.memory.schemas)} schemas")
        st.json([s[0].tolist() for s in st.session_state.udmm_core.memory.schemas])
