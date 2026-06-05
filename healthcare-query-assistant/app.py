```python
import streamlit as st
import google.generativeai as genai

# Configure page
st.set_page_config(
    page_title="Healthcare Query Assistant",
    page_icon="🏥",
    layout="centered"
)

# Gemini API Key from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")

# Title
st.title("🏥 Healthcare Query Assistant")
st.markdown("Ask any healthcare-related question and get AI-powered guidance.")

# User Input
question = st.text_area(
    "Enter your healthcare question:",
    placeholder="Example: What are the symptoms of diabetes?"
)

# Ask Button
if st.button("Get Answer"):

    if question.strip():

        with st.spinner("Generating answer..."):

            prompt = f"""
            You are a Healthcare Query Assistant.

            Guidelines:
            - Answer healthcare-related questions clearly.
            - Use simple language.
            - Keep answers concise and informative.
            - Do not diagnose diseases.
            - Advise users to consult a healthcare professional for medical concerns.
            - If the question is unrelated to healthcare, politely explain that you only answer healthcare questions.

            User Question:
            {question}
            """

            try:
                response = model.generate_content(prompt)

                st.subheader("Answer")
                st.write(response.text)

                st.info(
                    "Disclaimer: This chatbot provides general health information only and is not a substitute for professional medical advice."
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

    st.header("Examples")
    st.write("• What is diabetes?")
    st.write("• How can I improve my sleep?")
    st.write("• What causes headaches?")
    st.write("• Symptoms of dengue")
    st.write("• Benefits of drinking water")
```
