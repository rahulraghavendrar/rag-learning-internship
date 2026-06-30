# Preliminary Foundation Notes

## Goal

Build a strong understanding of the concepts behind Retrieval-Augmented Generation (RAG) before building complete RAG systems using LangChain and Vector Databases.

---

# 1. Embeddings

Embeddings are numerical representations of text that capture semantic meaning.

Example:

Text:

Internship duration is 2 months

↓

Embedding:

[0.12, -0.44, 0.89, ...]

Similar meanings produce similar vectors.

Example:

```text
"Interns receive a stipend"

and

"Interns are paid monthly"
```

produce embeddings that are close together.

---

# 2. all-MiniLM-L6-v2

Model used:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Meaning:

```text
all
│
├── Trained on many sentence datasets

MiniLM
│
├── Lightweight Transformer

L6
│
├── 6 Transformer Layers

v2
│
└── Improved Version
```

Output:

```text
384-Dimensional Embedding
```

---

# 3. Embedding Generation Pipeline

```text
Text

↓

Tokenizer

↓

Token IDs

↓

Embedding Layer

↓

Transformer Layer 1

↓

Transformer Layer 2

↓

Transformer Layer 3

↓

Transformer Layer 4

↓

Transformer Layer 5

↓

Transformer Layer 6

↓

Pooling

↓

Sentence Embedding
```

---

# 4. encode()

Used to convert text into embeddings.

```python
embedding = model.encode(text)
```

Example:

```text
Internship duration is 2 months

↓

[0.12, -0.44, 0.89, ...]
```

---

# 5. Cosine Similarity

Measures similarity between two embeddings.

Formula:

cos(θ) = (A · B) / (||A|| ||B||)

Interpretation:

```text
1.0  → Nearly identical

0.8  → Very similar

0.5  → Related

0.0  → Unrelated

-1.0 → Opposite
```

Used in:

* Semantic Search
* Vector Databases
* Retrieval Systems
* RAG

---

# 6. Chunking

Large documents should not be embedded as one large block.

Workflow:

```text
Document

↓

Chunking

↓

Chunks

↓

Embeddings
```

---

## Fixed Size Chunking

Splits text based on character or word count.

Example:

```python
chunk_size = 500
```

Advantages:

* Simple
* Fast

Disadvantages:

* May break sentences
* Context loss

---

## Semantic Chunking

Groups text based on meaning.

Workflow:

```text
Sentence

↓

Embedding

↓

Cosine Similarity

↓

Meaning-Based Groups
```

Advantages:

* Better context preservation
* Better retrieval quality

Disadvantages:

* More computationally expensive

---

# 7. Manual Retrieval

Retrieval without a vector database.

Workflow:

```text
Documents

↓

Embeddings

↓

Stored In Python

↓

Query

↓

Query Embedding

↓

Cosine Similarity

↓

Ranking

↓

Best Match
```

Important lesson:

Retrieval quality directly affects answer quality.

---

# 8. Vector Databases

Purpose:

Store embeddings and perform fast similarity search.

Workflow:

```text
Documents

↓

Embeddings

↓

Vector Database

↓

Similarity Search

↓

Results
```

Examples:

* ChromaDB
* Qdrant
* Pinecone
* Weaviate
* FAISS

---

# 9. ChromaDB Basics

## Client

```python
client = chromadb.Client()
```

Creates a database connection.

---

## Collection

```python
collection = client.create_collection(...)
```

Similar to a SQL table.

Stores:

* Documents
* Embeddings
* Metadata
* IDs

---

## add()

```python
collection.add(...)
```

Stores documents and embeddings.

---

## query()

```python
collection.query(...)
```

Performs similarity search.

Workflow:

```text
Question

↓

Embedding

↓

Similarity Search

↓

Results
```

---

# 10. Qdrant Basics

Qdrant is a production-grade vector database.

Stores:

* Vectors
* Metadata (Payloads)
* IDs

---

## Collection

Similar to a table.

Example:

```python
collection_name="internship_docs"
```

---

## Point

A single record.

Contains:

```text
ID
Vector
Payload
```

---

## Payload

Additional information stored alongside vectors.

Example:

```python
payload={
  "text":"Internship duration is 2 months"
}
```

---

# Important Qdrant Libraries

## QdrantClient

```python
from qdrant_client import QdrantClient
```

Connects to Qdrant.

---

## VectorParams

```python
from qdrant_client.models import VectorParams
```

Configures vector dimensions and similarity metric.

Example:

```python
VectorParams(
    size=384,
    distance=Distance.COSINE
)
```

---

## PointStruct

```python
from qdrant_client.models import PointStruct
```

Creates records.

---

## Distance

```python
Distance.COSINE
```

Similarity metrics available:

* COSINE
* DOT
* EUCLID
* MANHATTAN

For RAG:

```python
Distance.COSINE
```

is most common.

---

# Important Qdrant Functions

## tolist()

```python
embedding.tolist()
```

Converts:

```text
NumPy Array

↓

Python List
```

Required by Qdrant.

---

## create_collection()

```python
client.create_collection(...)
```

Creates a collection.

---

## upsert()

```python
client.upsert(...)
```

Stores vectors.

Think:

```text
Insert / Update
```

---

## query_points()

```python
client.query_points(...)
```

Performs semantic search.

---

# 11. RAG Foundations

RAG stands for:

Retrieval-Augmented Generation

Workflow:

```text
Question

↓

Retrieve Information

↓

Provide Context To LLM

↓

Generate Answer
```

---

# Why Not Fine-Tuning?

Fine-tuning:

```text
Knowledge

↓

Model Weights
```

Problem:

Knowledge updates require retraining.

---

# Why RAG?

RAG keeps knowledge external.

```text
Documents

↓

Vector Database

↓

Retrieve

↓

LLM
```

Knowledge updates require only re-indexing.

---

# RAG vs Fine-Tuning vs Prompt Engineering

## Prompt Engineering

Knowledge supplied through prompts.

Advantages:

* Easy
* Fast

Disadvantages:

* Limited by context window

---

## Fine-Tuning

Knowledge stored in model weights.

Advantages:

* Learns behavior

Disadvantages:

* Expensive
* Hard to update

---

## RAG

Knowledge stored externally.

Advantages:

* Easy updates
* Lower cost
* Scalable
* Reduces hallucinations

---

# Core RAG Pipeline

## Index

```text
Documents
↓
Chunking
↓
Embeddings
↓
Vector Database
```

---

## Retrieve

```text
Question
↓
Embedding
↓
Similarity Search
↓
Relevant Chunks
```

---

## Augment

```text
Question

+

Retrieved Context
```

---

## Generate

```text
Prompt
↓
LLM
↓
Answer
```

---

# 12. Retrieval Concepts

## Top-K Retrieval

Example:

```python
k=3
```

Return:

```text
Top 3 Results
```

---

## Context Window

LLMs have limited input capacity.

Therefore:

```text
Retrieve Relevant Chunks

Instead Of

Entire Documents
```

---

## Hallucinations

Occurs when the LLM generates unsupported information.

RAG reduces hallucinations by providing factual context.

---

## Precision

Measures:

```text
How Accurate Retrieved Results Are
```

---

## Recall

Measures:

```text
How Many Relevant Results Were Retrieved
```

---

# 13. LangChain Fundamentals

LangChain connects RAG components together.

Pipeline:

```text
PDF
↓
Loader
↓
Documents
↓
Splitter
↓
Chunks
↓
Embeddings
↓
Vector Store
↓
Retriever
```

---

# Document

Represents text and metadata.

Example:

```python
Document(
    page_content="Internship duration is 2 months",
    metadata={"source":"pdf"}
)
```

---

# PyPDFLoader

Loads PDFs.

```python
loader = PyPDFLoader("file.pdf")
```

Converts:

```text
PDF

↓

Documents
```

---

# RecursiveCharacterTextSplitter

Creates chunks.

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

Important Parameters:

* chunk_size
* chunk_overlap

---

# HuggingFaceEmbeddings

LangChain wrapper around embedding models.

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Important Functions:

## embed_query()

Used for user questions.

---

## embed_documents()

Used for document chunks.

---

# QdrantVectorStore

LangChain wrapper around Qdrant.

```python
QdrantVectorStore.from_documents(...)
```

Creates:

```text
Chunks
↓
Embeddings
↓
Qdrant
```

---

# Retriever

Creates a retrieval interface.

```python
retriever = vectorstore.as_retriever()
```

Search:

```python
retriever.invoke(query)
```

Workflow:

```text
Question
↓
Embedding
↓
Similarity Search
↓
Top-K Chunks
```

---

# Important Libraries Used

## sentence_transformers

```python
from sentence_transformers import SentenceTransformer
```

Generate embeddings.

---

## sklearn.metrics.pairwise

```python
from sklearn.metrics.pairwise import cosine_similarity
```

Compute cosine similarity.

---

## chromadb

```python
import chromadb
```

Vector database.

---

## qdrant_client

```python
from qdrant_client import QdrantClient
```

Qdrant database client.

---

## langchain_core.documents

```python
from langchain_core.documents import Document
```

LangChain document object.

---

## langchain_community.document_loaders

```python
from langchain_community.document_loaders import PyPDFLoader
```

Load PDFs.

---

## langchain_text_splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Chunk documents.

---

## langchain_huggingface

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

Generate embeddings inside LangChain.

---

## langchain_qdrant

```python
from langchain_qdrant import QdrantVectorStore
```

Connect LangChain with Qdrant.

---

# Complete Preliminary Foundation Pipeline

```text
PDF

↓

PyPDFLoader

↓

Documents

↓

Chunking

↓

Embeddings

↓

Qdrant

↓

Retriever

↓

Relevant Chunks

↓

LLM

↓

Answer
```

---

# Key Takeaways

✅ Embeddings represent semantic meaning.

✅ Cosine similarity measures semantic similarity.

✅ Chunking improves retrieval quality.

✅ Vector databases store embeddings efficiently.

✅ ChromaDB and Qdrant enable semantic search.

✅ RAG combines retrieval and generation.

✅ LangChain automates the RAG workflow.

✅ Qdrant + LangChain forms the retrieval layer of modern RAG systems.

✅ Retrieval quality directly impacts answer quality.

✅ Understanding these fundamentals is essential before building full RAG applications.
