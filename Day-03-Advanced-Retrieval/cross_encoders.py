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

from sentence_transformers import (
    CrossEncoder
)


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
    collection_name="reranking",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="reranking",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k":5}
)

print("Retriever Ready")


print("\nSTEP 5 : CROSS ENCODER\n")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker Loaded")


while True:

    query = input(
        "\nAsk Question: "
    )

    if query.lower() == "exit":

        break


    retrieved_docs = retriever.invoke(
        query
    )


    print("\nRETRIEVAL RESULTS\n")

    for i, doc in enumerate(
        retrieved_docs
    ):

        print(
            f"\nResult {i+1}"
        )

        print(
            doc.page_content[:200]
        )

        print(
            "\n" + "-"*50
        )


    pairs = []

    for doc in retrieved_docs:

        pairs.append(
            (
                query,
                doc.page_content
            )
        )


    scores = reranker.predict(
        pairs
    )


    reranked = sorted(
        zip(
            scores,
            retrieved_docs
        ),
        reverse=True,
        key=lambda x: x[0]
    )


    print("\nRERANKED RESULTS\n")


    for rank, (score, doc) in enumerate(
        reranked[:5]
    ):

        print(
            f"\nRank {rank+1}"
        )

        print(
            "Cross Encoder Score:",
            round(score,4)
        )

        print()

        print(
            doc.page_content
        )

        print(
            "\n" + "="*60
        )