import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        huggingfacehub_api_token=hf_token
    )
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant who explains answers clearly."),
    ("human", "{question}")
])

chain = prompt | llm

query = input("Enter your query: ")
response = chain.invoke({"question": query})

print("\nAI Response:\n")
print(response.content)