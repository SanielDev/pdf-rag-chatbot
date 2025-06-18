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
        store_pdf_chunks(chunks)
        st.success("✅ PDF processed and embedded!")

    query = st.text_input("Ask a question about the PDF:")

    if query:
        qa_chain = build_qa_chain()
        with st.spinner("Generating answer..."):
            # response = qa_chain.run(query)
            response = qa_chain.invoke(query)
            # st.markdown(f"**Answer:** {response}")
            st.markdown(f"**Answer:** {response['result']}")
