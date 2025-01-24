from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatGroq(model="llama-3.2-3b-preview")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who knows facts about {animal}"),
    ("human", "Tell me {fact_count} facts"),
])

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"animal": "pandas", "fact_count": 2})

print(result)
