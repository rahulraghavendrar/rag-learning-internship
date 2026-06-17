from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


loader = PyPDFLoader(
    "1-Week RAG Deep Dive Learning Plan.pdf"
)

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
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
    collection_name="lab5",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


vectorstore = QdrantVectorStore(
    client=client,
    collection_name="lab5",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)


question = (
    "What is the goal of Day 3?"
)


retrieved_docs = retriever.invoke(
    question
)


contexts = [

    doc.page_content

    for doc in retrieved_docs
]


generated_answer = """
Improve retrieval quality beyond simple similarity.
"""


ground_truth = """
Improve retrieval quality beyond simple similarity.
"""


evaluation_data = {

    "question": [
        question
    ],

    "answer": [
        generated_answer
    ],

    "contexts": [
        contexts
    ],

    "ground_truth": [
        ground_truth
    ]
}


dataset = Dataset.from_dict(
    evaluation_data
)


result = evaluate(
    dataset,

    metrics=[

        faithfulness,

        answer_relevancy,

        context_precision,

        context_recall
    ]
)


print(result)