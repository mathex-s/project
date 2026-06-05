import streamlit as st
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Healthcare Query Assistant",
    page_icon="🏥",
    layout="centered"
)

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Title
st.title("🏥 Healthcare Query Assistant")
st.write("Ask any healthcare-related question and get AI-powered guidance.")

# User input
question = st.text_area(
    "Enter your healthcare question:",
    placeholder="Example: What are the symptoms of diabetes?"
)

# Button
if st.button("Get Answer"):

    if question.strip():

        with st.spinner("Generating answer..."):

            prompt = f"""
            You are a healthcare assistant.

            Rules:
            - Answer only healthcare-related questions.
            - Use simple and clear language.
            - Keep answers informative and concise.
            - Do not provide medical diagnoses.
            - Recommend consulting a healthcare professional when necessary.

            User Question:
            {question}
            """

            try:
                response = model.generate_content(prompt)

                st.subheader("Answer")
                st.write(response.text)

                st.info(
                    "Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice."
                )

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please enter a question.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This Healthcare Query Assistant uses Google's Gemini AI "
        "to answer healthcare-related questions."
    )

    st.header("Example Questions")
    st.write("• What is diabetes?")
    st.write("• What causes headaches?")
    st.write("• How can I improve my sleep?")
    st.write("• What are the symptoms of dengue?")
    st.write("• How much water should I drink daily?")
