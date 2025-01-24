from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatGroq(model="llama-3.2-3b-preview")

messages = [
    SystemMessage("You are an expert in social media content strategy."),
    HumanMessage("Give a short tip to create engaging posts on instagram")
]

result = llm.invoke(messages)

print(result.content)