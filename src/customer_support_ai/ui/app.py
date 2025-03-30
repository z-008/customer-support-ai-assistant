"""
Streamlit UI for the Customer Support AI Assistant
"""

import streamlit as st
import requests
from datetime import datetime
import json

# Constants
API_URL = (
    "http://localhost:8000/api/v1"  # Update this if your API runs on a different port
)


def init_session_state():
    """Initialize session state variables"""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def create_new_chat():
    """Create a new chat session"""
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.rerun()


def get_chat_history(conversation_id):
    """Fetch conversation history from API"""
    try:
        response = requests.get(f"{API_URL}/conversation/{conversation_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching history: {str(e)}")
        return None


def send_message(message):
    """Send message to the API and get response"""
    try:
        # Prepare request payload
        payload = {"query": message}

        # Only add conversation_id and max_history_length if there's an existing conversation
        if st.session_state.conversation_id:
            payload.update(
                {
                    "conversation_id": st.session_state.conversation_id,
                    "max_history_length": 5,
                }
            )

        response = requests.post(f"{API_URL}/generate_response", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None


def main():
    st.title("Customer Support AI Assistant")

    # Initialize session state
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "past_conversations" not in st.session_state:
        st.session_state.past_conversations = []

    # Sidebar with past conversations (limited to 5)
    with st.sidebar:
        if st.button("New Chat"):
            # Add current conversation to past_conversations if it exists
            if st.session_state.conversation_id:
                if not any(
                    conv["id"] == st.session_state.conversation_id
                    for conv in st.session_state.past_conversations
                ):
                    st.session_state.past_conversations.append(
                        {
                            "id": st.session_state.conversation_id,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "preview": (
                                st.session_state.messages[0]["content"][:30] + "..."
                                if st.session_state.messages
                                else "Empty chat"
                            ),
                        }
                    )
                    # Keep only the last 5 conversations
                    st.session_state.past_conversations = (
                        st.session_state.past_conversations[-5:]
                    )

            # Reset current conversation
            st.session_state.conversation_id = None
            st.session_state.messages = []

        st.markdown("### Past Conversations")
        # Show last 5 conversations that aren't the current one
        for conv in reversed(
            st.session_state.past_conversations[-5:]
        ):  # Show most recent first
            if conv["id"] != st.session_state.conversation_id:
                if st.button(f"{conv['date']} - {conv['preview']}", key=conv["id"]):
                    st.session_state.conversation_id = conv["id"]

    # Main chat area - display current conversation
    if st.session_state.conversation_id:
        history = get_chat_history(st.session_state.conversation_id)
        if history:
            for interaction in history["interactions"]:
                with st.chat_message("user"):
                    st.write(interaction["query"])
                with st.chat_message("assistant"):
                    st.write(interaction["response"])
                    with st.expander("Response Details"):
                        st.markdown("**Expanded Query:**")
                        st.write(interaction["query"])

                        st.markdown("**Retrieved Documents:**")
                        st.json(interaction["retrieved_documents"])

                        st.markdown("**Evaluation Metrics:**")
                        st.json(interaction["evaluation_metrics"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_message(prompt)
                if response:
                    # Store conversation_id from response
                    st.session_state.conversation_id = response["conversation_id"]

                    # Add to past conversations if it's a new conversation
                    if not any(
                        conv["id"] == response["conversation_id"]
                        for conv in st.session_state.past_conversations
                    ):
                        st.session_state.past_conversations.append(
                            {
                                "id": response["conversation_id"],
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "preview": prompt[:30] + "...",
                            }
                        )
                        # Keep only the last 5 conversations
                        st.session_state.past_conversations = (
                            st.session_state.past_conversations[-5:]
                        )

                    # Display response
                    st.write(response["response"])

                    # Display expanded details in expander
                    with st.expander("Response Details"):
                        st.markdown("**Expanded Query:**")
                        st.write(response["expanded_query"])

                        st.markdown("**Retrieved Documents:**")
                        st.json(response["retrieved_documents"])

                        st.markdown("**Evaluation Metrics:**")
                        st.json(response["evaluation_metrics"])


if __name__ == "__main__":
    main()
