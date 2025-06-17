# 🎨 MoodPalette – AI Color Therapist

**MoodPalette** is a personalized color therapy web app powered by AI!  
Enter how you're feeling, and the app detects your mood, shows a matching color palette, and gives you a motivational quote 💬💖

![App Banner](![App Banner](https://github.com/ayeshamafrah-design/moodpallete_app/blob/main/Image.png)
)


---

## 💡 Features

- Detects user's mood from natural language input using an LLM (Gemma via Ollama)
- Displays a matching 3-color palette for common emotions
- Provides personalized motivational quotes
- User-friendly interface built with Streamlit
- Runs locally or deploys to Streamlit Cloud

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/) (for running Gemma model locally)
- [Matplotlib](https://matplotlib.org/) (for rendering color palettes)

---

## 🚀 How to Run Locally

### 1. Clone the repo

git clone https://github.com/ayeshamafrah-design/moodpalette_app.git

cd moodpalette_app

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the app

streamlit run app.py
#### Make sure Ollama is installed and the gemma3 model is pulled using:
ollama run gemma3

## 👩‍💻 Created By
Ayesha Mafrah – Researcher, AI & NLP Enthusiast
