from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence


llm = ChatGroq(model="llama-3.2-3b-preview")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who knows facts about {animal}"),
    ("human", "Tell me {fact_count} facts"),
])

translation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a translation expert who translates provided text to {language}"),
    ("human", "Translate the following text to {language}: {text}"),
])

count_words = RunnableLambda(lambda x: f"There are {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "Spanish"})

chain = prompt_template | llm | StrOutputParser() | prepare_for_translation | translation_template | llm | StrOutputParser()

result = chain.invoke({"animal": "pandas", "fact_count": 1})

print(result)