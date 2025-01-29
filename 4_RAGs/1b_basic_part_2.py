import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma")

embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Where does Gandalf meet Frodo?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.5},
)

relevant_docs = retriever.invoke(query)

print("\n---Relevant Documents---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}: {doc.page_content}")
    if doc.metadata:
        print(f"Source: {doc.metadata}\n")