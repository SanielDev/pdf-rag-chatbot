from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from vector_store import load_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

def build_qa_chain(retriever):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  
        huggingfacehub_api_token=token,
        temperature=0.5,
        max_new_tokens=512,
        top_k=50

    )

#     prompt_template =PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a highly factual resume parser. Only answer using information strictly from the given context below. 
# Do not assume, infer, or make up any details that are not explicitly present in the context.
# If the context does not contain the answer, reply with "Not mentioned in the resume."

# <context>
# {context}
# </context>

# Question: {question}
# Answer:"""
# )
    # prompt_template = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template="""
    #     You are an AI assistant reading a candidate’s resume. Use the context below to answer the question clearly and professionally.
    #     If the answer is not present, respond with "Not mentioned in the resume."

    #     Context:
    #     {context}

    #     Question:
    #     {question}

    #     Answer:
    #     """
    # )

    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant reviewing a candidate's resume. Carefully extract information from the context below.

Pay attention to structured sections like EDUCATION, PROJECTS, CERTIFICATIONS, SKILLS, and EXPERIENCE.
Do not assume or generate extra information — answer only based on the given context.

If the answer is not present, respond exactly with: "Not mentioned in the resume."

Context:
{context}

Question:
{question}

Answer:
"""
)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

