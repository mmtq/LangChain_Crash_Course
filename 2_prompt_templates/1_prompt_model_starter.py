from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


llm = ChatGroq(model="llama-3.2-3b-preview")

# -----------------------------------------------------------
# Prompt with String Template
# -----------------------------------------------------------

# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max."

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({
#     "tone": "energetic",
#     "company": "Google",
#     "position": "Software Engineer",
#     "skill": "Python"
# })

# result = llm.invoke(prompt)

# print(result.content)

# -----------------------------------------------------------
# Prompt with System and Human Messages (using Tuples)
# -----------------------------------------------------------


messages = [
    ("system", "You're a comedian, who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({
    "topic": "cars",
    "joke_count": 5
})

result = llm.invoke(prompt)

print(result.content)