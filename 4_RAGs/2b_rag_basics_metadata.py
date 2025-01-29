import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_metadata")

embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings) 

query = "Where is Dracula's castle located?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)

relevant_docs = retriever.invoke(query)

print("\n---Relevant Documents---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}: {doc.page_content}")
    if doc.metadata:
        print(f"Source: {doc.metadata['source']}\n")
