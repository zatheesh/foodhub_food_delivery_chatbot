import streamlit as st
from foodhub_chatbot import chatagent

st.set_page_config(
    page_title="FoodHub Food Delivery Customer Support Chatbot",
    page_icon="🍔",
    layout="centered"
)

USER_CREDENTIALS = {
    "satheesh": "foodhub123",
    "admin": "admin123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.logged_in:
    st.title("🍔 FoodHub Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

st.sidebar.success(f"Logged in as: {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

st.title("🍔 FoodHub Food Delivery Customer Support Chatbot")
st.caption("Ask about orders, delivery status, refunds, payments, or cancellations.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I’m FoodHub Assistant. How can I help you today?"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Example: Where is my order O12488?")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking FoodHub records..."):
            try:
                response = chatagent(user_input)
            except Exception:
                response = (
                    "I apologize, but I'm experiencing a temporary issue. "
                    "Please try again."
                )

        st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
