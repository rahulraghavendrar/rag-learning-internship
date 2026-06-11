# Day 2 - Manual Vector Search

## Summary

Today I built my first retrieval system without using any vector database.

Instead of using ChromaDB, Pinecone, FAISS, or Qdrant, I manually stored document embeddings inside a Python list and used cosine similarity to retrieve the most relevant document for a user query.

This helped me understand the fundamental retrieval process that powers Retrieval-Augmented Generation (RAG).

---

# What Problem Are We Solving?

Suppose we have multiple documents:

* Internship duration is 2 months
* Interns receive a stipend of 15000 rupees per month
* Applications close on June 30

Now a user asks:

"How much money do interns receive?"

The goal is to find which document best answers the question.

Traditional keyword search compares words directly.

Semantic search compares meanings.

To compare meanings, both documents and questions must first be converted into embeddings.

---

# Retrieval Pipeline

The retrieval process follows this pipeline:

Documents

↓

Embeddings

↓

Stored in Memory

↓

User Query

↓

Query Embedding

↓

Cosine Similarity

↓

Rank Results

↓

Return Best Match

This is the same fundamental pipeline used by modern RAG systems.

---

# Step 1 - Store Documents

The documents were stored inside a Python list.

Example:

documents = [
"Internship duration is 2 months",
"Interns receive a stipend of 15000 rupees per month",
"Applications close on June 30"
]

These documents act as a small database.

---

# Step 2 - Convert Documents Into Embeddings

The Sentence Transformer model converts every document into a 384-dimensional vector.

Example:

Document

↓

Embedding

↓

[0.12, -0.45, 0.89, ...]

After encoding:

document_embeddings

contains one embedding vector for every document.

---

# Step 3 - Encode User Query

The query:

"How much money do interns receive?"

is converted into another embedding vector.

Example:

Query

↓

Embedding

↓

[0.67, -0.23, 0.91, ...]

Now both documents and queries exist in the same semantic space.

---

# Step 4 - Calculate Similarity

For each document:

Cosine Similarity

is calculated between:

Query Embedding

and

Document Embedding

This produces a similarity score.

Example:

Document A -> 0.32

Document B -> 0.89

Document C -> 0.17

Higher score means more similar meaning.

---

# Step 5 - Rank Documents

All documents are stored together with their similarity scores.

Example:

[
("Document A", 0.32),
("Document B", 0.89),
("Document C", 0.17)
]

The list is sorted in descending order.

Highest score appears first.

---

# Step 6 - Retrieve Best Match

The document at index 0 becomes the retrieved result.

Example:

scores[0]

This document is considered the most relevant answer to the query.

---

# Important Observation

Initially I expected:

"Stipend is 15000 rupees"

to match:

"How much money do interns receive?"

However retrieval quality depends heavily on wording and semantic context.

When the document was changed to:

"Interns receive a stipend of 15000 rupees per month"

retrieval became much more accurate.

This demonstrated an important lesson:

Embeddings are powerful but retrieval quality depends on document wording and context.

---

# Internal Architecture

Documents

↓

SentenceTransformer

↓

Embeddings

↓

Python List

↓

User Query

↓

Embedding

↓

Cosine Similarity

↓

Ranking

↓

Best Match

This is essentially a simplified vector database.

---

# What I Learned

1. Retrieval means finding the most relevant document.

2. Documents and queries must be converted into embeddings.

3. Cosine similarity measures semantic similarity.

4. Similarity scores can be ranked.

5. A vector database is essentially a system that stores embeddings and retrieves the most similar vectors efficiently.

6. Today's implementation manually recreated the retrieval process used by RAG systems.

7. Retrieval quality depends on the wording of the stored documents.

---

# Key Takeaway

Today I implemented the core retrieval mechanism behind Retrieval-Augmented Generation (RAG) using only Python lists, embeddings, and cosine similarity.

Understanding this process makes it easier to learn vector databases such as ChromaDB, FAISS, Pinecone, and Qdrant because they all perform the same task at a larger scale.
