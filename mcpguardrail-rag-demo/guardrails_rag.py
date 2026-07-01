import os
import google.generativeai as genai

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

loader = PyPDFLoader(
    "pdf/1-Week RAG Deep Dive Learning Plan.pdf"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(
    documents
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(
    ":memory:"
)

client.create_collection(
    collection_name="rag",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5
    }
)


def input_guardrail(question):

    blocked = [
        "api key",
        "password",
        "system prompt",
        "ignore previous instructions",
        "delete database"
    ]

    question = question.lower()

    for word in blocked:

        if word in question:

            return False

    return True


def retrieval_guardrail(results):

    return len(results) > 0


def output_guardrail(answer):

    blocked = [
        "definitely",
        "guaranteed",
        "certainly"
    ]

    answer = answer.lower()

    for word in blocked:

        if word in answer:

            return False

    return True


while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    if not input_guardrail(question):

        print("Input Guardrail Triggered")

        continue

    results = retriever.invoke(
        question
    )

    if not retrieval_guardrail(results):

        print("Retrieval Guardrail Triggered")

        continue

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    prompt = f"""
Use only the provided context to answer the question.

If the answer is not present in the context, respond with:

The document does not contain enough information.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(
        prompt
    )

    answer = response.text

    if not output_guardrail(answer):

        print("Output Guardrail Triggered")
        print("Please verify the response using the original document.")

    print("\nAnswer:\n")
    print(answer)