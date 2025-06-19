from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document

CHROMA_DIR = "db"

def store_pdf_chunks(chunks):
    # documents = [Document(page_content=chunk) for chunk in chunks]
    documents = chunks

    parent_splitter = CharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=50
        # separators=["\n\n", "\n", ".", " ", ""]
    )
    child_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(collection_name="split_parents", embedding_function=embedding)
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(documents)
    return retriever

def load_vector_store():
    raise NotImplementedError("Not used anymore — replaced by store_pdf_chunks returning retriever.")
