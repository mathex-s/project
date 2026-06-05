import streamlit as st
import google.generativeai as genai

# Configure page
st.set_page_config(
    page_title="Healthcare Query Assistant",
    page_icon="🏥",
    layout="wide"
)

# Read API key from Streamlit Secrets
API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure Gemini
genai.configure(api_key=API_KEY)

# Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# Title
st.title("🏥 Healthcare Query Assistant")
st.markdown("Ask any healthcare-related question.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
prompt = st.chat_input("Type your healthcare question...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    try:

        healthcare_prompt = f"""
        You are a Healthcare Query Assistant.

        Instructions:
        - Answer only healthcare and wellness related questions.
        - Use simple language.
        - Do not provide a medical diagnosis.
        - Suggest consulting a doctor when necessary.
        - Keep answers concise and informative.

        Question:
        {prompt}
        """

        response = model.generate_content(healthcare_prompt)

        answer = response.text

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        st.error(f"Error: {e}")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "Healthcare Query Assistant powered by Google Gemini AI."
    )

    st.header("Example Questions")
    st.write("• What is diabetes?")
    st.write("• Symptoms of dengue")
    st.write("• How can I improve my sleep?")
    st.write("• What causes headaches?")
    st.write("• Benefits of drinking water")
