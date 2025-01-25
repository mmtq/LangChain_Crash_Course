from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence


llm = ChatGroq(model="llama-3.2-3b-preview")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who knows facts about {animal}"),
    ("human", "Tell me {fact_count} facts"),
])

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_llm = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_llm], last=parse_output)
 
response = chain.invoke({"animal": "pandas", "fact_count": 2})

print(response)

