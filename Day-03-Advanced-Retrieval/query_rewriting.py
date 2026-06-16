from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_qdrant import (
    QdrantVectorStore
)

from langchain_openai import (
    ChatOpenAI
)

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    VectorParams
)

import os


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


print("\nSTEP 1 : LOADING PDF\n")

loader = PyPDFLoader(
    "1-Week RAG Deep Dive Learning Plan.pdf"
)

documents = loader.load()


print("\nSTEP 2 : CHUNKING\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(
    documents
)


print("\nSTEP 3 : EMBEDDINGS\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("\nSTEP 4 : QDRANT\n")

client = QdrantClient(
    ":memory:"
)

client.create_collection(
    collection_name="rewriter",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rewriter",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)


print("\nSTEP 5 : LLM REWRITER\n")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)


while True:

    query = input(
        "\nAsk Question: "
    )

    if query.lower() == "exit":

        break


    rewrite_prompt = f"""
Rewrite the following user query
to make it more suitable for
document retrieval.

Query:

{query}

Return only the rewritten query.
"""

    rewritten_query = (
        llm.invoke(
            rewrite_prompt
        ).content
    )

    print("\nOriginal Query:")

    print(query)

    print("\nRewritten Query:")

    print(rewritten_query)


    results = retriever.invoke(
        rewritten_query
    )

    print("\nTOP RESULTS\n")

    for doc in results:

        print(doc.page_content)

        print(
            "\n" + "=" * 60
        )