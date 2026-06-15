from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (VectorParams,Distance)

from google.colab import files
uploaded=files.upload()

#STEP 1 Load the PDF
loader=PyPDFLoader("1-Week RAG Deep Dive Learning Plan.pdf")
documents=loader.load()

#STEP 2 Split the documents
splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
chunks=splitter.split_documents(documents)

#STEP 3 Store in QDrant

client=QdrantClient(":memory:")
client.create_collection(
    collection_name="Sliding window",
    vectors_config=VectorParams(size=384,distance=Distance.COSINE)
)
#STEP 4 Create the embedding model
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=QdrantVectorStore(
    client=client,
    collection_name="Sliding window",
    embedding=embeddings
)
vectorstore.add_documents(chunks)

retriever=vectorstore.as_retriever(
    search_kwargs={"k":3}
)
query=input("Ask your question")
results=retriever.invoke(query)
print("Results are:")
for result in results:
  print(result.page_content)
