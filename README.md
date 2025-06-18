# 🧠 PDF RAG Chatbot with Free LLM + ChromaDB

A Streamlit-based chatbot that lets you query PDF files using Retrieval-Augmented Generation (RAG) with ChromaDB and free HuggingFace LLMs.

## 🚀 Features

- Upload PDF
- Embeds text using `sentence-transformers`
- Stores vectors in local `ChromaDB`
- Uses HuggingFace inference API to answer questions
- Fully local + free LLM endpoint

## 🧩 Tech Stack

- `Streamlit` UI
- `ChromaDB` for vector storage
- `LangChain` for chaining + QA
- `Sentence-Transformers` for embeddings
- `FLAN-T5` or similar HuggingFace LLM endpoint

## 🛠 Setup

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# 2. Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your HuggingFace token to `.env`
HUGGINGFACEHUB_API_TOKEN=your_token_here

# 5. Run the app
streamlit run app.py
