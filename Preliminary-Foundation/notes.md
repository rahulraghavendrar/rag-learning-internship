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
