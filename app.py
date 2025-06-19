import streamlit as st
from pdf_loader import extract_chunks_from_pdf
from vector_store import store_pdf_chunks
from qa_chain import build_qa_chain

st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="centered")
st.title("📄 PDF RAG Chatbot with Free LLM + ChromaDB")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        chunks = extract_chunks_from_pdf(uploaded_file)
        # store_pdf_chunks(chunks)
        retriever = store_pdf_chunks(chunks)
        st.success("✅ PDF processed and embedded!")

    query = st.text_input("Ask a question about the PDF:")

    if query:
        # qa_chain = build_qa_chain()
        qa_chain = build_qa_chain(retriever)
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke(query)
            st.markdown(f"**Answer:** {response['result']}")

            source_docs = response.get("source_documents", [])
            if source_docs:
                seen = set()
                for doc in source_docs:
                    content = doc.page_content.strip()
                    if content not in seen:
                        seen.add(content)
                        st.markdown("**Source:**")
                        st.code(content[:500])  # limit for readability
                        break  # ✅ Only show the first unique source
             