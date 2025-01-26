import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    print("\n---Document Chunks Information---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample document chunk: {docs[0].page_content}\n")
    
    print("\n---Creating Embeddings---")
    
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("\n---Finished Creating Embeddings---")
    
    print("\n---Creating Vector Store---")
    
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    
    print("\n---Finished Creating Vector Store---")
    
else:
    print("Persistent directory exists. No initialization required...")