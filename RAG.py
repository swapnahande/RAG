import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        huggingfacehub_api_token=HF_TOKEN
    )
)



documents = [
    Document(
        page_content="Machine learning is a subset of artificial intelligence.",
        metadata={"source": "Doc1"}
    ),
    Document(
        page_content="Deep learning uses neural networks with multiple layers.",
        metadata={"source": "Doc2"}
    ),
    Document(
        page_content="RAG combines retrieval with generation to answer questions.",
        metadata={"source": "Doc3"}
    ),
    Document(
        page_content="LangChain helps build applications using LLMs.",
        metadata={"source": "Doc4"}
    )
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use the provided context first.
If the answer is clearly found in the context, answer using the context and cite the source names exactly in italics.
If the answer is not found in the context, answer from your own knowledge and write in italics: Source: General knowledge.

Rules:
1. Do not make up document citations.
2. If context supports the answer, end with: Source: <doc names>
3. If context does not support the answer, end with: Source: General knowledge
4. Keep the answer simple and clear.
"""
    ),
    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{query}"
    )
])

def rag_pipeline(query):
    docs = retriever.invoke(query)

    context_parts = []
    sources = []

    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        sources.append(src)
        context_parts.append(f"[{src}] {doc.page_content}")

    context = "\n".join(context_parts)

    chain = prompt | llm
    resp = chain.invoke({
        "context": context,
        "query": query
    })

    return resp.content

while True:
    print("------------------------------------------------")
    query = input("Enter your query or 'exit' to quit:  ")
    if query.lower() == "exit":
        print("Exiting...")
        exit()
    result = rag_pipeline(query)
    print("\nAnswer:\n")
    print(result)