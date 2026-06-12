# Preliminary Foundation Notes

## Goal

Learn the core concepts behind Retrieval-Augmented Generation (RAG) before using frameworks like LangChain.

---

# 1. Embeddings

Embeddings are numerical representations of text.

Example:

```text
"Internship duration is 2 months"
```

↓

```text
[0.12, -0.44, 0.89, ...]
```

Similar meanings produce similar vectors.

---

## all-MiniLM-L6-v2

Used through:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Meaning

```text
all
│
├── Trained on many datasets

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
384-dimensional embedding
```

---

## Embedding Generation Pipeline

```text
Text

↓

Tokenizer

↓

Token IDs

↓

Embedding Layer

↓

6 Transformer Layers

↓

Pooling

↓

Sentence Embedding
```

---

## encode()

```python
embedding = model.encode(text)
```

Converts text into a 384-dimensional vector.

---

# 2. Cosine Similarity

Measures semantic similarity between embeddings.

Interpretation:

```text
1.0  → Almost identical

0.8  → Very similar

0.5  → Related

0.0  → Unrelated

-1.0 → Opposite
```

Used in:

* Semantic Search
* Vector Databases
* RAG Retrieval

---

# 3. Chunking

Large documents should not be embedded as a single block.

Instead:

```text
PDF

↓

Chunking

↓

Chunks

↓

Embeddings
```

---

## Fixed Size Chunking

Split text by word count.

Example:

```python
chunk_size = 20
```

Advantages:

* Simple
* Fast

Disadvantages:

* Can split sentences
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

Group Similar Sentences
```

Advantages:

* Better context preservation
* Better retrieval quality

---

# 4. Manual Retrieval

Implemented retrieval without a vector database.

Workflow:

```text
Documents

↓

Embeddings

↓

Stored In Python List

↓

Query

↓

Query Embedding

↓

Cosine Similarity

↓

Rank Results

↓

Best Match
```

Important lesson:

Retrieval quality depends heavily on document wording and context.

---

# 5. Vector Databases

Problem:

Manual retrieval becomes slow for thousands or millions of documents.

Solution:

Vector Databases.

Purpose:

```text
Documents

↓

Embeddings

↓

Storage

↓

Similarity Search

↓

Retrieved Results
```

Examples:

* ChromaDB
* Qdrant
* Pinecone
* FAISS
* Weaviate

---

# 6. ChromaDB Basics

### Client

```python
client = chromadb.Client()
```

Database connection.

### Collection

```python
collection = client.create_collection(...)
```

Similar to a SQL table.

Stores:

* Documents
* Embeddings
* IDs
* Metadata

---

### Add Documents

```python
collection.add(...)
```

Process:

```text
Document

↓

Embedding

↓

Stored
```

---

### Query Documents

```python
collection.query(...)
```

Process:

```text
Question

↓

Embedding

↓

Similarity Search

↓

Top Result
```

---

# Complete RAG Pipeline

```text
PDF

↓

Chunking

↓

Embeddings

↓

Vector Database

↓

User Query

↓

Query Embedding

↓

Similarity Search

↓

Retrieved Chunks

↓

LLM

↓

Final Answer
```

---

# Key Interview Questions

### What is an embedding?

A numerical representation of text meaning.

---

### What is cosine similarity?

A metric used to measure similarity between embeddings.

---

### Why do we need chunking?

Large documents must be split into smaller meaningful pieces before embedding.

---

### What is retrieval?

Finding the most relevant document or chunk for a query.

---

### What is a vector database?

A system that stores embeddings and performs similarity search.

---

### What is ChromaDB?

An open-source vector database used for semantic retrieval.

---

# Foundation Completed

✅ Embeddings

✅ Cosine Similarity

✅ Fixed Size Chunking

✅ Semantic Chunking

✅ Manual Retrieval

✅ ChromaDB

✅ Vector Databases

Next:

➡ LangChain

➡ Qdrant

➡ Advanced Retrieval

➡ Full RAG Pipelines

# Day 04 - Qdrant Basics

## Goal

Learn how a vector database stores embeddings and performs semantic retrieval.

Qdrant is the first production-grade vector database I learned before moving to LangChain.

---

# What is Qdrant?

Qdrant is a Vector Database.

Unlike SQL databases which store rows and columns, Qdrant stores:

* Embeddings (Vectors)
* Documents
* Metadata (Payloads)

Workflow:

```text
Document
↓
Embedding
↓
Qdrant

Question
↓
Embedding
↓
Qdrant Search
↓
Top Matches
```

---

# Why Do We Need Qdrant?

Without Qdrant:

```text
Documents
↓
Embeddings
↓
Python List
↓
Compare One By One
```

This becomes slow for thousands or millions of documents.

With Qdrant:

```text
Documents
↓
Embeddings
↓
Qdrant
↓
Fast Similarity Search
```

---

# Important Concepts

## Collection

Similar to a SQL table.

Example:

```python
collection_name="internship_docs"
```

Stores:

* Points
* Vectors
* Payloads

---

## Point

One record inside a collection.

Contains:

```text
ID
Vector
Payload
```

Example:

```python
PointStruct(
    id=1,
    vector=[...],
    payload={...}
)
```

---

## Vector

The embedding generated from text.

Example:

```text
Internship duration is 2 months
```

↓

```python
[0.12, -0.44, 0.89, ...]
```

---

## Payload

Extra information stored alongside the vector.

Example:

```python
payload={
    "text":"Internship duration is 2 months"
}
```

---

# Libraries Used

## SentenceTransformer

```python
from sentence_transformers import SentenceTransformer
```

Purpose:

```text
Text
↓
Embedding
```

---

## QdrantClient

```python
from qdrant_client import QdrantClient
```

Purpose:

```text
Connect To Qdrant
```

---

## VectorParams

```python
from qdrant_client.models import VectorParams
```

Purpose:

Configure collection vectors.

Example:

```python
VectorParams(
    size=384,
    distance=Distance.COSINE
)
```

Meaning:

* Embedding size = 384
* Similarity = Cosine

---

## PointStruct

```python
from qdrant_client.models import PointStruct
```

Purpose:

Create records for Qdrant.

---

## Distance

```python
from qdrant_client.models import Distance
```

Purpose:

Choose similarity metric.

Example:

```python
Distance.COSINE
```

Other options:

* COSINE
* DOT
* EUCLID
* MANHATTAN

For RAG:

```python
Distance.COSINE
```

is usually used.

---

# Important Functions

## encode()

```python
model.encode(text)
```

Converts text into embeddings.

Example:

```text
Text
↓
Vector
```

---

## tolist()

```python
embedding.tolist()
```

Converts:

```python
numpy array
```

↓

```python
python list
```

Qdrant requires Python lists.

---

## create_collection()

```python
client.create_collection(...)
```

Creates a collection.

Think:

```text
SQL
↓
CREATE TABLE
```

---

## upsert()

```python
client.upsert(...)
```

Stores points inside Qdrant.

Think:

```text
Save To Database
```

---

## query_points()

```python
client.query_points(...)
```

Retrieves the most similar vectors.

Think:

```text
Question
↓
Embedding
↓
Search
↓
Top Matches
```

---

# Top-K Retrieval

Example:

```python
limit=3
```

Meaning:

```text
Return Top 3 Results
```

Common values:

* Top 3
* Top 5
* Top 10

---

# Complete Pipeline

```text
Documents
↓
Embeddings
↓
Qdrant Collection
↓
User Question
↓
Query Embedding
↓
query_points()
↓
Top Matching Documents
```

---

# Key Takeaways

✅ Qdrant is a vector database.

✅ Collections store vectors and payloads.

✅ Points are individual records.

✅ Embeddings are stored as vectors.

✅ Payloads store original text and metadata.

✅ upsert() stores data.

✅ query_points() retrieves data.

✅ Qdrant performs semantic search using vector similarity.

✅ This forms the Retrieve step of a RAG pipeline.
