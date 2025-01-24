from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema import SystemMessage, HumanMessage, AIMessage

llm = ChatGroq(model="llama-3.2-3b-preview")

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    
    print("AI: " + result.content)

print("-------Message History-------")
print(chat_history)