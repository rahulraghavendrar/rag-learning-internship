from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient


print("\nSTEP 1 : LOADING PDF\n")

loader = PyPDFLoader("dummy.pdf")

documents = loader.load()

print(f"Documents Loaded: {len(documents)}")


print("\nSTEP 2 : SPLITTING DOCUMENTS\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Chunks Created: {len(chunks)}")


print("\nSTEP 3 : LOADING EMBEDDING MODEL\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


print("\nSTEP 4 : CREATING QDRANT VECTOR STORE\n")

client = QdrantClient(":memory:")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    client=client,
    collection_name="rag_day1_demo"
)

print("Vector Store Created")


print("\nSTEP 5 : CREATING RETRIEVER\n")

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

print("Retriever Ready")


print("\nSTEP 6 : USER QUERY\n")

query = input("Enter your question: ")


print("\nSTEP 7 : RETRIEVING RELEVANT CHUNKS\n")

results = retriever.invoke(query)

for i, doc in enumerate(results):

    print(f"\nChunk {i+1}")

    print("\nContent:")

    print(doc.page_content)

    print("\nMetadata:")

    print(doc.metadata)

    print("\n" + "-" * 60)