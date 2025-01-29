import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_metadata"
)

embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "Who is gandalf?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs = {"k":3}
)

relevant_docs = retriever.invoke(query)

combined_input = (
    query
    + "Here are some documents that might help answer the question: "
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

llm = ChatGroq(model="llama-3.2-3b-preview")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = llm.invoke(messages)

print(result.content)