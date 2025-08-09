from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# extract data from pdf file
def load_pdf_file(data):
    loader = DirectoryLoader(data,globe="*.pdf",loader = PyPDFLoader)
    
    documents = loader.load()
    
    return documents

# filter to minimal
def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    """
    given a list of document objects, return a new list of document objects
    containing  only 'source' in metadata and the  og page_content          
    """
    minimal_docs:List[Document]=[]
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source":src}
            )
        )
    return minimal_docs

# split the data into text chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    chunks = text_splitter.split_documents(minimal_docs)
    return chunks

# downloading the embeddings from huggingface
def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


