import streamlit as st
import requests
import uuid

# Server URL (adjust if deployed elsewhere)
API_URL = "http://localhost:8000/chat"

# Generate a unique ID per user session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.markdown("<h2 style='text-align: center;'>🩺 Personal Medical Assistant</h2>", unsafe_allow_html=True)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Type your medical symptoms or question here...")

# Display previous chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send to FastAPI backend
    try:
        response = requests.post(API_URL, json={
            "user_id": st.session_state["user_id"],
            "question": user_input
        })
        response.raise_for_status()
        result = response.json()["response"]
    except Exception as e:
        result = f"❌ Server Error: {e}"

    # Add assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": result})
    with st.chat_message("assistant"):
        st.markdown(result)
