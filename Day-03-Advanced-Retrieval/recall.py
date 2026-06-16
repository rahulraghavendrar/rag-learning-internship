from google.colab import files

uploaded = files.upload()


from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_qdrant import (
    QdrantVectorStore
)

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    VectorParams
)


print("\nSTEP 1 : LOADING PDF\n")

loader = PyPDFLoader(
    "1-Week RAG Deep Dive Learning Plan.pdf"
)

documents = loader.load()

print(
    "Pages Loaded:",
    len(documents)
)


print("\nSTEP 2 : CHUNKING\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(
    documents
)

print(
    "Chunks Created:",
    len(chunks)
)


print("\nSTEP 3 : EMBEDDINGS\n")

embedding_model = HuggingFaceEmbeddings(
    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"
)


print("\nSTEP 4 : QDRANT\n")

client = QdrantClient(
    ":memory:"
)

client.create_collection(
    collection_name="recall_demo",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="recall_demo",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)

print("Chunks Stored")


TOP_K = 5

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)


questions = [

    (
        "What is the goal of Day 1?",
        "rag pipeline"
    ),

    (
        "What chunking strategies are covered?",
        "semantic chunking"
    ),

    (
        "What retrieval approach is introduced in Day 3?",
        "bm25"
    ),

    (
        "Name one evaluation framework.",
        "ragas"
    ),

    (
        "Name one advanced rag variant.",
        "agentic"
    )

]


print("\nSTEP 5 : CALCULATING RECALL\n")

correct = 0

total = len(
    questions
)


for query, keyword in questions:

    print("\nQuestion:")

    print(query)

    results = retriever.invoke(
        query
    )

    found = False

    for doc in results:

        if keyword.lower() in (
            doc.page_content.lower()
        ):

            found = True

            break

    if found:

        print(
            "Relevant Chunk Retrieved"
        )

        correct += 1

    else:

        print(
            "Relevant Chunk NOT Retrieved"
        )


recall = (
    correct / total
)

print("\n" + "=" * 60)

print(
    f"Recall@{TOP_K}: {recall:.4f}"
)

print(
    f"Recall Percentage: {recall*100:.2f}%"
)

print("=" * 60)