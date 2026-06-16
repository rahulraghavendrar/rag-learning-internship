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

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    VectorParams
)

from rank_bm25 import BM25Okapi


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


chunk_texts = [
    chunk.page_content
    for chunk in chunks
]


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


tokenized_chunks = [

    text.lower().split()

    for text in chunk_texts
]

bm25 = BM25Okapi(
    tokenized_chunks
)


client = QdrantClient(
    ":memory:"
)

client.create_collection(
    collection_name="hybrid",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="hybrid",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":5}
)


while True:

    query = input(
        "\nAsk Question: "
    )

    if query.lower() == "exit":

        break


    print("\nBM25 RETRIEVAL\n")

    tokenized_query = (
        query.lower().split()
    )

    bm25_scores = (
        bm25.get_scores(
            tokenized_query
        )
    )

    bm25_ranked = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )

    bm25_indices = [

        idx

        for idx, score in bm25_ranked[:5]
    ]

    print("BM25 Chunk IDs:")

    print(bm25_indices)


    print("\nDENSE RETRIEVAL\n")

    dense_results = (
        retriever.invoke(query)
    )

    dense_indices = []

    for doc in dense_results:

        dense_indices.append(
            doc.metadata["chunk_id"]
        )

    print("Dense Chunk IDs:")

    print(dense_indices)


    print("\nINTERSECTION RESULTS\n")

    intersection_ids = set(
        bm25_indices
    ).intersection(
        set(dense_indices)
    )

    if len(intersection_ids) == 0:

        print(
            "No Common Chunks Found"
        )

    else:

        for idx in intersection_ids:

            print(
                "\nChunk ID:",
                idx
            )

            print()

            print(
                chunk_texts[idx]
            )

            print(
                "\n" + "="*60
            )