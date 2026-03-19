import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    st.error("HF_TOKEN not found in .env file")
    st.stop()

st.set_page_config(page_title="Simple RAG", page_icon="🤖")
st.title("🤖 Simple RAG")

@st.cache_resource
def load_llm():
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            huggingfacehub_api_token=hf_token
        )
    )

@st.cache_resource
def load_vectorstore():
    documents = [
        Document(
            page_content="Artificial Intelligence refers to the ability of machines to perform tasks that normally require human intelligence.",
            metadata={"source": "Doc1"}
        ),
        Document(
            page_content="Machine Learning is a subset of Artificial Intelligence that focuses on training machines to learn from data and make predictions or decisions without being explicitly programmed.",
            metadata={"source": "Doc2"}
        ),
        Document(
            page_content="Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to model complex patterns in data.",
            metadata={"source": "Doc3"}
        )
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)

llm = load_llm()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
Answer the question in simple language and in detail.

Use the provided context first.
If the answer is available from the context, answer using it and cite the exact source names at the end.
If the answer is not available from the context, answer from your own knowledge and end with:
Source: General knowledge

Do not make up citations.
"""
    ),
    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{query}"
    )
])

def rag_pipeline(query):
    retrieved_docs = retriever.invoke(query)

    context_parts = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[{source}] {doc.page_content}")

    context = "\n".join(context_parts)

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "query": query
    })

    return response.content, retrieved_docs

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Enter your query...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result, docs = rag_pipeline(query)
            st.markdown(result)

            with st.expander("Retrieved Documents"):
                for doc in docs:
                    st.write(f"**{doc.metadata.get('source', 'Unknown')}**: {doc.page_content}")

    st.session_state.messages.append({"role": "assistant", "content": result})