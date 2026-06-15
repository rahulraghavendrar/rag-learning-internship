from langchain_community.document_loaders import PyPDFLoader

from langchain_experimental.text_splitter import SemanticChunker

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


print("\nSTEP 1 : LOADING PDF\n")

loader = PyPDFLoader("internship.pdf")

documents = loader.load()

print("Pages Loaded:", len(documents))


print("\nSTEP 2 : LOADING EMBEDDING MODEL\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


print("\nSTEP 3 : SEMANTIC CHUNKING\n")

semantic_splitter = SemanticChunker(
    embeddings=embedding_model
)

chunks = semantic_splitter.split_documents(
    documents
)

print("Semantic Chunks Created:", len(chunks))


for i, chunk in enumerate(chunks[:5]):

    print(f"\nChunk {i+1}")

    print(chunk.page_content)

    print("-" * 50)


print("\nSTEP 4 : CREATING QDRANT COLLECTION\n")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="semantic_docs",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Collection Created")


print("\nSTEP 5 : STORING CHUNKS IN QDRANT\n")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="semantic_docs",
    embedding=embedding_model
)

vectorstore.add_documents(chunks)

print("Chunks Stored")


print("\nSTEP 6 : CREATING RETRIEVER\n")

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

print("Retriever Ready")


while True:

    query = input(
        "\nAsk Question (type exit to stop): "
    )

    if query.lower() == "exit":
        break

    results = retriever.invoke(query)

    print("\nTOP MATCHING CHUNKS\n")

    for i, doc in enumerate(results):

        print(f"\nResult {i+1}")

        print("\nContent:")

        print(doc.page_content)

        print("\nSource:")

        print(doc.metadata.get("source"))

        print("\n" + "-" * 60)