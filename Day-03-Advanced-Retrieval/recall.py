from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder

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

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(":memory:")

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

vectorstore.add_documents(chunks)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 50}
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

TARGET_RECALL = 0.85
RELEVANCE_THRESHOLD = 5.0

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(query)

    pairs = [
        (query, doc.page_content)
        for doc in retrieved_docs
    ]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(scores, retrieved_docs),
        key=lambda x: x[0],
        reverse=True
    )

    relevant_docs = [
        (score, doc)
        for score, doc in reranked
        if score >= RELEVANCE_THRESHOLD
    ]

    total_relevant = len(relevant_docs)

    if total_relevant == 0:
        print("\nNo Relevant Chunks Found")
        continue

    selected_k = total_relevant

    for k in range(
        1,
        total_relevant + 1
    ):

        recall = k / total_relevant

        if recall >= TARGET_RECALL:
            selected_k = k
            break

    final_results = relevant_docs[:selected_k]

    print(f"\nTarget Recall : {TARGET_RECALL}")
    print(f"Selected K    : {selected_k}")
    print(f"Actual Recall : {recall:.4f}")

    for rank, (score, doc) in enumerate(
        final_results,
        start=1
    ):

        print(f"\nRank {rank}")
        print(
            f"Chunk ID : {doc.metadata['chunk_id']}"
        )
        print(
            f"Score    : {score:.4f}"
        )
        print()
        print(doc.page_content)
        print("\n" + "=" * 60)