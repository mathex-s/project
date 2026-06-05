import streamlit as st
import google.generativeai as genai

# Configure page
st.set_page_config(
    page_title="Healthcare Query Assistant",
    page_icon="🏥",
    layout="wide"
)

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# Title
st.title("🏥 Healthcare Query Assistant")
st.write("Ask any healthcare-related question.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your healthcare question...")

if prompt:

    # Show user message
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.spinner("Generating answer..."):

        healthcare_prompt = f"""
        You are a Healthcare Query Assistant.

        Rules:
        - Answer healthcare-related questions clearly.
        - Use simple language.
        - Do not provide medical diagnoses.
        - Suggest consulting a healthcare professional when necessary.
        - If the question is unrelated to healthcare, politely say that you only answer healthcare questions.

        User Question:
        {prompt}
        """

        try:
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
        "AI-powered Healthcare Query Assistant using Google Gemini."
    )

    st.header("Sample Questions")
    st.write("• What is diabetes?")
    st.write("• Symptoms of dengue")
    st.write("• How can I improve my sleep?")
    st.write("• What causes headaches?")
    st.write("• How much water should I drink daily?")
