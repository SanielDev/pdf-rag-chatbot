from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def extract_chunks_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            doc = Document(page_content=text, metadata={"page": i})
            documents.append(doc)

    return documents
