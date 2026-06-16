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

# Hybrid Search Variants

After learning Hybrid Search, an important question arises:

```text
How should BM25 and Dense Retrieval results be combined?
```

There are two common approaches:

1. Union-Based Hybrid Retrieval
2. Intersection-Based Hybrid Retrieval

---

# Union-Based Hybrid Retrieval

Most beginner implementations use:

```text
BM25 Results

+

Dense Results
```

Mathematically:

```text
A ∪ B
```

Example:

BM25 returns:

```text
Chunk 1
Chunk 2
Chunk 3
```

Dense Retrieval returns:

```text
Chunk 2
Chunk 3
Chunk 5
```

Union:

```text
Chunk 1
Chunk 2
Chunk 3
Chunk 5
```

All unique chunks are returned.

---

## Workflow

```text
Question
↓
BM25 Search

and

Dense Search
↓
Combine Results
↓
Remove Duplicates
↓
Final Chunks
```

---

## Advantages

* Higher Recall
* More Documents Retrieved
* Lower Chance Of Missing Relevant Information

---

## Disadvantages

* More Noise
* More Irrelevant Chunks
* Larger Context Sent To LLM

---

# Intersection-Based Hybrid Retrieval

Instead of returning everything:

```text
BM25 Results

∩

Dense Results
```

Mathematically:

```text
A ∩ B
```

Only chunks retrieved by BOTH systems are returned.

---

## Example

BM25 returns:

```text
Chunk 1
Chunk 2
Chunk 3
```

Dense Retrieval returns:

```text
Chunk 2
Chunk 3
Chunk 5
```

Intersection:

```text
Chunk 2
Chunk 3
```

Only the common chunks survive.

---

# Workflow

```text
Question
↓
BM25 Search
↓
Top Chunks

and

Dense Search
↓
Top Chunks

↓

Find Common Chunks

↓

Final Results
```

---

# Why Use Intersection Retrieval?

A chunk returned by both systems has passed:

```text
Keyword Matching
```

and

```text
Semantic Matching
```

Therefore:

```text
Higher Confidence
```

that the chunk is relevant.

---

# How We Implemented It

Instead of comparing IDs:

```python
intersection_ids = set(
    bm25_indices
).intersection(
    set(dense_indices)
)
```

we used:

```python
bm25_texts.intersection(
    dense_texts
)
```

because:

* Simpler
* No metadata dependency
* Easier to understand

---

# Union vs Intersection

| Feature           | Union  | Intersection |
| ----------------- | ------ | ------------ |
| Recall            | High   | Lower        |
| Precision         | Medium | High         |
| Number Of Results | More   | Fewer        |
| Noise             | More   | Less         |
| Context Size      | Larger | Smaller      |
| Confidence        | Medium | High         |

---

# When To Use Union?

Use Union Retrieval when:

```text
Missing Information
is more dangerous than
retrieving extra information.
```

Examples:

* Research Assistants
* Legal Search
* Medical Search
* Knowledge Discovery

Goal:

```text
High Recall
```

---

# When To Use Intersection?

Use Intersection Retrieval when:

```text
Only highly relevant
chunks should be returned.
```

Examples:

* Customer Support
* FAQ Systems
* Enterprise Chatbots
* Internal Documentation Search

Goal:

```text
High Precision
```

---

# Production Systems

Many production RAG systems use:

```text
Union Retrieval
```

for initial retrieval.

Then:

```text
Reranking
```

is applied later.

Some systems use:

```text
Intersection Retrieval
```

when precision is more important than recall.

---

# Libraries Used

## BM25Okapi

```python
from rank_bm25 import BM25Okapi
```

Purpose:

```text
Keyword Search
```

---

## QdrantVectorStore

```python
from langchain_qdrant import QdrantVectorStore
```

Purpose:

```text
Dense Retrieval
```

---

## intersection()

```python
bm25_texts.intersection(
    dense_texts
)
```

Purpose:

```text
Find Common Chunks
```

between BM25 and Dense Retrieval.

---

# Interview Questions

### What is Hybrid Search?

Hybrid Search combines BM25 and Dense Retrieval to improve retrieval quality.

---

### What is Union-Based Hybrid Retrieval?

Union Retrieval returns all unique chunks retrieved by either BM25 or Dense Retrieval.

---

### What is Intersection-Based Hybrid Retrieval?

Intersection Retrieval returns only chunks retrieved by both BM25 and Dense Retrieval.

---

### Which has higher recall?

```text
Union Retrieval
```

---

### Which has higher precision?

```text
Intersection Retrieval
```

---

# Key Takeaways

1. Hybrid Search combines BM25 and Dense Retrieval.

2. Hybrid Search can be implemented using Union or Intersection.

3. Union Retrieval prioritizes Recall.

4. Intersection Retrieval prioritizes Precision.

5. Intersection Retrieval returns only chunks agreed upon by both retrieval systems.

6. Production RAG systems often start with Union Retrieval and later apply Reranking.

Next:

➡ Query Expansion
