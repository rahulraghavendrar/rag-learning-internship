from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


print("\nSTEP 1 : LOADING PDFs\n")

all_documents = []

pdf_files = [
    "internship.pdf",
    "company_policy.pdf",
    "ai_projects.pdf",
    "employee_handbook.pdf",
    "training_program.pdf"
]

for pdf in pdf_files:

    loader = PyPDFLoader(pdf)

    documents = loader.load()

    all_documents.extend(documents)

print("Documents Loaded:", len(all_documents))


print("\nSTEP 2 : SPLITTING DOCUMENTS\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_documents)

print("Chunks Created:", len(chunks))


print("\nSTEP 3 : LOADING EMBEDDING MODEL\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


print("\nSTEP 4 : CREATING QDRANT COLLECTION\n")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="naive_rag",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Collection Created")


print("\nSTEP 5 : STORING CHUNKS IN QDRANT\n")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="naive_rag",
    embedding=embedding_model
)

vectorstore.add_documents(chunks)

print("Chunks Stored")


print("\nSTEP 6 : CREATING RETRIEVER\n")

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

print("Retriever Ready")


print("\nNAIVE RAG SYSTEM READY\n")

while True:

    query = input(
        "\nAsk a Question (type exit to stop): "
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