# LangChain Fundamentals

## What is LangChain?

LangChain is a framework that connects all the components of an AI application together.

Without LangChain:

PDF → Chunking → Embeddings → Vector Database → Retrieval

must be manually coded.

With LangChain:

LangChain provides ready-made components that connect these steps together.

---

# LangChain RAG Pipeline

PDF
↓
Loader
↓
Documents
↓
Text Splitter
↓
Chunks
↓
Embeddings
↓
Vector Store
↓
Retriever
↓
Relevant Chunks

---

# 1. Document

Represents a piece of text.

Example:

```python
Document(
    page_content="Internship duration is 2 months",
    metadata={
        "source":"internship.pdf"
    }
)
```

Important Fields:

* page_content → actual text
* metadata → additional information

---

# 2. PyPDFLoader

Used to load PDFs.

Example:

```python
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
```

Function:

PDF
↓
Documents

Each page becomes a LangChain Document object.

---

# 3. RecursiveCharacterTextSplitter

Used to split large documents into smaller chunks.

Example:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

Important Parameters:

* chunk_size → maximum chunk length
* chunk_overlap → overlapping context between chunks

Function:

Documents
↓
Chunks

---

# 4. HuggingFaceEmbeddings

Used to generate embeddings.

Example:

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Functions:

```python
embed_query()
```

Used for user questions.

```python
embed_documents()
```

Used for document chunks.

Function:

Text
↓
Embeddings

---

# 5. Qdrant Vector Store

Stores embeddings in Qdrant through LangChain.

Example:

```python
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    client=client,
    collection_name="internship_docs"
)
```

Function:

Chunks
↓
Embeddings
↓
Qdrant

---

# 6. Retriever

Used to perform semantic search.

Example:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)
```

Query:

```python
results = retriever.invoke(query)
```

Function:

Question
↓
Embedding
↓
Qdrant Search
↓
Top-K Chunks

---

# Important Concepts

## k

```python
search_kwargs={"k":3}
```

Returns Top 3 most relevant chunks.

Equivalent to:

```python
limit=3
```

in raw Qdrant.

---

## Similarity Search

Retriever uses embeddings and cosine similarity to find the most relevant chunks.

---

# Complete LangChain Pipeline

PDF
↓
PyPDFLoader
↓
Documents
↓
RecursiveCharacterTextSplitter
↓
Chunks
↓
HuggingFaceEmbeddings
↓
Embeddings
↓
Qdrant Vector Store
↓
Retriever
↓
Relevant Chunks

---

# Key Takeaways

1. LangChain is a framework, not an LLM.

2. Documents store text and metadata.

3. PyPDFLoader converts PDFs into Documents.

4. RecursiveCharacterTextSplitter creates chunks.

5. HuggingFaceEmbeddings converts text into vectors.

6. Qdrant stores vectors efficiently.

7. Retrievers perform semantic search.

8. LangChain automates the RAG pipeline while hiding much of the low-level implementation.
