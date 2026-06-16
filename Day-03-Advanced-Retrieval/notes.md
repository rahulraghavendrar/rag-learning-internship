# Day 03 - Advanced Retrieval & Reranking

## Goal

Improve retrieval quality beyond simple vector similarity search.

Until Day 2 we built:

```text
PDF
↓
Chunking
↓
Embeddings
↓
Qdrant
↓
Retriever
```

This is called Dense Retrieval.

Day 3 focuses on improving retrieval quality using:

* BM25
* Hybrid Search
* Query Expansion
* Query Rewriting
* Metadata Filtering
* Structured Retrieval
* Reranking
* Retrieval Metrics

---

# BM25 Retrieval

## What is BM25?

BM25 is a keyword-based retrieval algorithm.

Unlike Qdrant:

```text
Question
↓
Embeddings
↓
Similarity Search
```

BM25 performs:

```text
Question
↓
Keyword Matching
↓
Ranking
```

---

## Example

Document:

```text
Internship duration is 2 months.
```

Question:

```text
internship duration
```

BM25 finds:

```text
internship
duration
```

inside the document.

Result:

```text
High Score
```

---

Question:

```text
How long is the internship?
```

BM25 only sees:

```text
internship
```

Result:

```text
Lower Score
```

because BM25 does not understand semantic meaning.

---

# Dense Retrieval

What we used in Day 1 and Day 2.

Pipeline:

```text
Question
↓
Embedding
↓
Qdrant
↓
Similarity Search
```

Example:

```text
How long is the internship?
```

and

```text
Internship duration is 2 months.
```

are considered similar because embeddings understand meaning.

---

# BM25 vs Dense Retrieval

| BM25                     | Dense Retrieval                 |
| ------------------------ | ------------------------------- |
| Uses keywords            | Uses embeddings                 |
| Exact word matching      | Semantic matching               |
| Fast                     | Slightly slower                 |
| No embeddings required   | Embeddings required             |
| Good for technical terms | Good for semantic understanding |

---

# Why BM25 Still Matters

Suppose the document contains:

```text
BM25
```

Question:

```text
What is BM25?
```

BM25 retrieval is often better because:

```text
Exact Match
```

is present.

Dense retrieval may sometimes miss rare technical words.

---

# Real Production Systems

Most production RAG systems use:

```text
BM25
+
Dense Retrieval
```

called:

```text
Hybrid Search
```

which will be the next topic.

---

# Libraries Used

## rank_bm25

```python
from rank_bm25 import BM25Okapi
```

Purpose:

```text
Keyword Based Retrieval
```

---

## PyPDFLoader

```python
from langchain_community.document_loaders import PyPDFLoader
```

Purpose:

```text
PDF
↓
Documents
```

---

## RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Purpose:

```text
Documents
↓
Chunks
```

---

# Functions Used

## BM25Okapi()

```python
bm25 = BM25Okapi(tokenized_chunks)
```

Creates a BM25 index.

---

## get_scores()

```python
scores = bm25.get_scores(
    tokenized_query
)
```

Returns relevance scores.

---

# Key Takeaways

1. BM25 is a keyword-based retrieval algorithm.

2. Dense Retrieval uses embeddings.

3. BM25 performs exact matching.

4. Dense Retrieval performs semantic matching.

5. Most production systems combine BM25 and Dense Retrieval.

Next:

➡ Hybrid Search
