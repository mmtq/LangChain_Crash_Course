from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel


llm = ChatGroq(model="llama-3.2-3b-preview")

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You're a movie critic."),
    ("human", "Write a brief summary of the movie {movie}"),
])

def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages([
        ("system", "You're a movie critic."),
        ("human", "Analyze the plot: {plot} . What are the strengths and weaknesses?")        
    ])
    
    return plot_template.format_prompt(plot=plot)

def analyze_characters(characters):
    characters_template = ChatPromptTemplate.from_messages([
        ("system", "You're a movie critic."),
        ("human", "Analyze the characters: {characters} . What are the strengths and weaknesses?")        
    ])
    
    return characters_template.format_prompt(characters=characters)

# Combine analyses into a final verdict
def combine_verdicts(plot_analysis, characters_analysis):
    return f"Plot: {plot_analysis}\nCharacters: {characters_analysis}"

#Simplify branches with LCEL
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | llm | StrOutputParser()
)

characters_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | llm | StrOutputParser()
)

chain = (
    summary_template | llm | StrOutputParser() |
    RunnableParallel(branches={"plot": plot_branch_chain, "characters": characters_branch_chain}) |
    RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

result = chain.invoke({"movie": "Avengers: Endgame"})

print(result)