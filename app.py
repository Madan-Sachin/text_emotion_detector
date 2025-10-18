import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel, Field
import os

# -----------------------------
# Streamlit App Title
# -----------------------------
st.set_page_config(page_title="Text Emotion Detector", layout="centered")
st.title("🧠 Text Emotion Detector (Happy, Anger, Love, Sad)")

# -----------------------------
# Load HuggingFace API Key from Streamlit secrets
# -----------------------------
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error(
        "HuggingFace API token not found in Streamlit secrets!\n"
        "Add HUGGINGFACEHUB_API_TOKEN in secrets.toml."
    )
    st.stop()

# Optional: set environment variable for HuggingFace libraries
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Enter your text here:")

# -----------------------------
# Pydantic Model for Output
# -----------------------------
class EmotionOutput(BaseModel):
    emotion: str = Field(..., description="Detected emotion: happy, anger, love, or sad")
    confidence: float = Field(..., description="Confidence score of prediction (0-1)")

# -----------------------------
# Detect Emotion Button
# -----------------------------
if st.button("Detect Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        # -----------------------------
        # Prompt Template – restrict to 4 emotions
        # -----------------------------
        prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are an emotion classifier. Classify the following text into exactly one of these emotions: "
                "happy, anger, love, sad.\n\n"
                "Text: {text}\n"
                "Answer in JSON format like: {{\"emotion\": \"happy\", \"confidence\": 0.95}}"
            )
        )

        # -----------------------------
        # HuggingFace LLM Endpoint (Mistral)
        # -----------------------------
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            huggingfacehub_api_token=HF_TOKEN,
            model_kwargs={"temperature": 0.0, "max_new_tokens": 100}  # temperature 0 for deterministic
        )

        # -----------------------------
        # LangChain LLMChain
        # -----------------------------
        chain = LLMChain(
            llm=llm_endpoint,
            prompt=prompt,
            output_parser=PydanticOutputParser(pydantic_object=EmotionOutput)
        )

        # -----------------------------
        # Run Chain and Display
        # -----------------------------
        try:
            result = chain.run(user_input)
            st.success(f"**Emotion:** {result.emotion}\n**Confidence:** {result.confidence}")
        except Exception as e:
            st.error(f"Error: {e}")
