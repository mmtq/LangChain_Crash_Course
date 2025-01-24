from langchain_groq import ChatGroq
# from langchain_ollama import OllamaLLM

llm = ChatGroq(model="llama-3.2-3b-preview")
# llm = OllamaLLM(model="llama3.2:1b")


result = llm.invoke("Who's Messi")

print(result.content)