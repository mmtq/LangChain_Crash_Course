from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project and FireStore Database
3. Retrieve the Project ID
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. pip install langchain-google-firestore
6. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""


PROJECT_ID = "langchain-25f00"
SESSION_ID = "user_session-new"
COLLECTION_NAME = "chat_history"

print("Initializing firestore client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing chat history...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID, collection=COLLECTION_NAME, client=client
)
print("Chat history initialized.")
print("Current chat history:", chat_history.messages)

llm = ChatGroq(model="llama-3.2-3b-preview")

print("Start chatting with the AI. Type 'exit' to stop.")

while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)
    result = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(result.content)

    print("AI: " + result.content)