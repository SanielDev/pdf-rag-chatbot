from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from vector_store import load_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

def build_qa_chain():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    retriever = load_vector_store()

    # llm = HuggingFaceEndpoint(
    #     repo_id="google/flan-t5-small",   # or "tiiuae/falcon-7b-instruct"
    #     huggingfacehub_api_token=token,
    #     temperature=0.5,
    #     # max_length=512
    # )
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=token,
        temperature=0.5,
        max_new_tokens=512,
        top_k=50
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
