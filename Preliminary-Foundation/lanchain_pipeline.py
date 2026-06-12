from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

# ------------------------------------
# STEP 1
# LOAD PDF
# ------------------------------------

loader = PyPDFLoader(
    "dummy.pdf"
)

documents = loader.load()

print("Documents Loaded:")
print(len(documents))

print()

# ------------------------------------
# STEP 2
# SPLIT DOCUMENTS
# ------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(
    documents
)

print("Chunks Created:")
print(len(chunks))

print()

# ------------------------------------
# STEP 3
# LOAD EMBEDDING MODEL
# ------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")

print()

# ------------------------------------
# STEP 4
# CREATE EMBEDDINGS
# ------------------------------------

texts = []

for chunk in chunks:

    texts.append(
        chunk.page_content
    )

embeddings = (
    embedding_model
    .embed_documents(texts)
)

print("Embeddings Generated")

print()

# ------------------------------------
# STEP 5
# INSPECT RESULTS
# ------------------------------------

print("First Chunk:\n")

print(
    chunks[0].page_content
)

print()

print("Metadata:\n")

print(
    chunks[0].metadata
)

print()

print(
    "Embedding Dimension:",
    len(embeddings[0])
)