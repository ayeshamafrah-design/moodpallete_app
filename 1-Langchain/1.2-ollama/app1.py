import streamlit as st 
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import re  # ✅ NEW: to extract the mood keyword safely

load_dotenv()

# Langsmith tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Predefined color palette for moods
mood_colors = {
    "happy": ["#FFF700", "#FFB300", "#FFD700"],
    "sad": ["#4A90E2", "#357ABD", "#203864"],
    "angry": ["#D7263D", "#9E1B32", "#730000"],
    "calm": ["#7FDBB6", "#A8E6CF", "#D0EFFF"],
    "anxious": ["#FF6F61", "#E94E77", "#5D4E60"],
    "excited": ["#FF5E00", "#FF9500", "#FFC300"],
    "bored": ["#9E9E9E", "#BDBDBD", "#E0E0E0"]
}

# Prompt to detect mood
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a mood expert and color therapist. Respond ONLY with one of these moods: happy, sad, angry, calm, anxious, excited, bored."),
    ("user", "User feeling: {feeling}")
])

# LangChain setup
llm = Ollama(model="gemma3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit interface
st.title("🎨 MoodPalette – AI Color Therapist")
input_text = st.text_input("🧠 How are you feeling today?")

if input_text:
    with st.spinner("Analyzing your mood..."):
        raw_response = chain.invoke({"feeling": input_text})
        mood_key = raw_response.strip().lower()

        # ✅ Extract just the known mood
        mood_key = re.findall(r'\b(happy|sad|angry|calm|anxious|excited|bored)\b', mood_key)
        mood_key = mood_key[0] if mood_key else None

        if mood_key:
            st.markdown(f"### 🧭 Detected Mood: **{mood_key.capitalize()}**")
            colors = mood_colors[mood_key]

            # Show color blocks
            st.markdown("### 🎨 Color Palette")
            fig, ax = plt.subplots(figsize=(5, 1))
            for i, color in enumerate(colors):
                ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax.set_xlim(0, len(colors))
            ax.set_ylim(0, 1)
            ax.axis('off')
            st.pyplot(fig)

            # Motivational quote
            quote_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a motivational coach. Write a short quote for someone feeling {mood}."),
                ("user", f"mood: {mood_key}")
            ])
            quote_chain = quote_prompt | llm | output_parser
            quote = quote_chain.invoke({"mood": mood_key})
            st.markdown(f"> 💬 *{quote.strip()}*")
        else:
            st.warning("⚠️ Couldn't determine a known mood. Try using simple emotional words like 'I feel happy' or 'I'm bored'.")
