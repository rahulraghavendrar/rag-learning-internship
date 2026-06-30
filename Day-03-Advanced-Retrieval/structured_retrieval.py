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
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue
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


print("\nSTEP 3 : ADDING METADATA\n")

for chunk in chunks:

    text = chunk.page_content.lower()

    if "day 1" in text:

        chunk.metadata["day"] = "Day 1"

    elif "day 2" in text:

        chunk.metadata["day"] = "Day 2"

    elif "day 3" in text:

        chunk.metadata["day"] = "Day 3"

    elif "day 4" in text:

        chunk.metadata["day"] = "Day 4"

    elif "day 5" in text:

        chunk.metadata["day"] = "Day 5"

    else:

        chunk.metadata["day"] = "Unknown"

print("Metadata Added")


print("\nSTEP 4 : EMBEDDINGS\n")

embedding_model = HuggingFaceEmbeddings(
    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"
)


print("\nSTEP 5 : QDRANT\n")

client = QdrantClient(
    ":memory:"
)

client.create_collection(
    collection_name="structured_rag",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="structured_rag",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)

print("Chunks Stored")


print("\nSTRUCTURED RETRIEVAL READY\n")


while True:

    query = input(
        "\nAsk Question: "
    )

    if query.lower() == "exit":

        break


    query_lower = query.lower()


    detected_day = None


    if (
        "chunking" in query_lower
        or
        "semantic chunking" in query_lower
        or
        "recursive chunking" in query_lower
    ):

        detected_day = "Day 2"


    elif (
        "retrieval" in query_lower
        or
        "bm25" in query_lower
        or
        "reranking" in query_lower
    ):

        detected_day = "Day 3"


    elif (
        "evaluation" in query_lower
        or
        "mrr" in query_lower
        or
        "ndcg" in query_lower
    ):

        detected_day = "Day 4"


    elif (
        "rag"
        in query_lower
    ):

        detected_day = "Day 1"


    elif (
        "agentic"
        in query_lower
        or
        "graph rag"
        in query_lower
    ):

        detected_day = "Day 5"


    print("\nDETECTED ROUTE\n")

    print(
        "Selected:",
        detected_day
    )


    if detected_day is None:

        print(
            "\nNo Route Found"
        )

        continue

    metadata_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.day",
                match=MatchValue(
                    value=detected_day
                )
            )
        ]
    )
    results=client.query_points(
        collection_name="structured_rag",
        query=embedding_model.embed_query(
            query
        ),
        metadata_filter=metadata_filter,
        limit=3
    )
    for result in results.points:

        print(
            result.payload["page_content"]
        )

        print()

        print(
            "Day:",
            result.payload["metadata"]["day"]
        )

        print()