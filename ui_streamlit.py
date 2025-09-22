import streamlit as st
import pandas as pd
import numpy as np
import os

# Correctly import from the src directory
from src.udmm2.udmm_v4_core import UDMMCore

# --- Page Configuration ---
st.set_page_config(
    page_title="UDMM v4 Gemini Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 UDMM v4 Gemini-Powered Agent")

# --- API Key Management in Sidebar ---
with st.sidebar:
    st.header("Configuration")
    st.write("This agent uses the Gemini LLM. Please provide your API key to begin.")

    # Get API key from user input
    gemini_api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")

    # Store the key in session state when the user provides it
    if gemini_api_key:
        st.session_state.gemini_api_key = gemini_api_key
        st.success("API Key loaded successfully!")

    # Display status
    if "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
        st.info("Agent is ready.")
    else:
        st.warning("Agent is waiting for API Key.")

# --- Initialization ---
# Initialize the UDMM core only if the API key is available
if "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
    if "udmm_core" not in st.session_state:
        st.session_state.udmm_core = UDMMCore(dimensionality=5, gemini_api_key=st.session_state.gemini_api_key)
        print("Initialized new UDMMCore instance with Gemini key.")
    # If key changes, re-initialize
    elif st.session_state.udmm_core.llm.api_key != st.session_state.gemini_api_key:
        st.session_state.udmm_core = UDMMCore(dimensionality=5, gemini_api_key=st.session_state.gemini_api_key)
        print("Re-initialized UDMMCore instance with new Gemini key.")

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Main Chat and Visualization Area ---
if "udmm_core" in st.session_state:
    col1, col2 = st.columns([2, 1])

    # Column 1: Chat Interface
    with col1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask the agent..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    response_dict = st.session_state.udmm_core.process_input(prompt)
                    agent_response_text = response_dict.get("response", "Error: No response text found.")
                    st.markdown(agent_response_text)

            st.session_state.messages.append({"role": "assistant", "content": agent_response_text})

    # Column 2: Internal State Visualization
    with col2:
        st.header("Agent's Internal State")
        st.write("The agent's internal 'attractor state' evolves with each interaction.")

        attractor_state = st.session_state.udmm_core.attractor.get_state()
        attractor_labels = st.session_state.udmm_core.attractor.labels
        attractor_df = pd.DataFrame({'Strength': attractor_state}, index=attractor_labels)

        st.bar_chart(attractor_df)

        with st.expander("See Raw State Data"):
            st.write("Current Attractor State:")
            st.json(dict(zip(attractor_labels, attractor_state)))
            st.write("Current Intent Hierarchy:")
            st.json({k: v.tolist() for k, v in st.session_state.udmm_core.intent.get_hierarchy().items()})
            st.write("Schemas in Memory:")
            st.write(f"{len(st.session_state.udmm_core.memory.schemas)} schemas")

else:
    st.info("Please enter your Gemini API Key in the sidebar to start the chat.")
