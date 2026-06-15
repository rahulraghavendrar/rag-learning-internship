from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from google.colab import files
uploaded=files.upload()


print("\nSTEP 1 : PARSING PDF USING UNSTRUCTURED\n")

elements = partition_pdf(
    filename="1-Week RAG Deep Dive Learning Plan.pdf"
)

print("Elements Found:", len(elements))


print("\nSTEP 2 : CONVERTING ELEMENTS TO LANGCHAIN DOCUMENTS\n")

documents = []

for i, element in enumerate(elements):

    doc = Document(
        page_content=str(element),
        metadata={
            "type": element.category,
            "element_id": i + 1
        }
    )

    documents.append(doc)

print("Documents Created:", len(documents))


print("\nSTEP 3 : LOADING EMBEDDING MODEL\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


print("\nSTEP 4 : CREATING QDRANT COLLECTION\n")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="structured_docs",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Collection Created")


print("\nSTEP 5 : STORING DOCUMENTS IN QDRANT\n")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="structured_docs",
    embedding=embedding_model
)

vectorstore.add_documents(documents)

print("Documents Stored")


print("\nSTEP 6 : CREATING RETRIEVER\n")

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

print("Retriever Ready")


while True:

    query = input(
        "\nAsk A Question (type exit to stop): "
    )

    if query.lower() == "exit":
        break

    results = retriever.invoke(query)

    print("\nTOP MATCHING DOCUMENTS\n")

    for doc in results:

        print("\nType:")
        print(doc.metadata["type"])

        print("\nElement ID:")
        print(doc.metadata["element_id"])

        print("\nContent:")
        print(doc.page_content)

        print("\n" + "-" * 60)