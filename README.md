# 🧠 PDF RAG Chatbot with Free LLM + ChromaDB

A Streamlit-based chatbot that lets you query PDF files using Retrieval-Augmented Generation (RAG) with ChromaDB and free HuggingFace LLMs.

## 🚀 Features

- Upload any resume PDF 📄
- Parses and chunks documents using LangChain
- Uses `ParentDocumentRetriever` for hierarchical chunking
- Embeds using `sentence-transformers`
- Stores vectors locally with ChromaDB
- Answers powered by Hugging Face's `Mixtral-8x7B-Instruct` endpoint
- Returns answers with source snippets ✨

## 🧩 Tech Stack

- 🖥 **Streamlit** – UI for chat interface
- 🧠 **LangChain** – for RAG logic and document parsing
- 🔍 **ChromaDB** – local vector store
- 🧩 **Sentence-Transformers** – text embeddings
- 🤖 **Mixtral-8x7B-Instruct** – HuggingFace-hosted LLM (free tier)

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
