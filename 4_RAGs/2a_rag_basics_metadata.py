import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
book_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_metadata")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    if not os.path.exists(book_dir):
        raise FileNotFoundError(f"Directory not found: {book_dir}")
    
    book_files = [f for f in os.listdir(book_dir) if f.endswith(".txt")]
    
    documents = []
    
    for book_file in book_files:
        file_path = os.path.join(book_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)
            
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    print("\n---Document Chunks Information---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample document chunk: {docs[0].page_content}\n")
    
    print("\n---Creating Embeddings---")
    
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory,
    )

else:
    print("Persistent directory exists. No initialization required...")