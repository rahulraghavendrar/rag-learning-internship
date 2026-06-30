from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client=QdrantClient(":memory:")

client.create_collection(
    collection_name="rag",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore=QdrantVectorStore(
    client=client,
    collection_name="rag",
    embedding=embedding_model
)

loader=PyPDFLoader("pdf/1-Week RAG Deep Dive Learning Plan.pdf")

documents=loader.load()

chunks=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
).split_documents(documents)

vectorstore.add_documents(chunks)

retriever=vectorstore.as_retriever(
    search_kwargs={"k":5}
)

def retrieve(question:str)->str:

    docs=retriever.invoke(question)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )