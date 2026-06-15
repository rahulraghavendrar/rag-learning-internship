# Day 02 - Chunking Strategies & Text Processing

## Goal

Master document preparation and chunking for Retrieval-Augmented Generation (RAG).

Chunking is one of the most important components of a RAG system because retrieval quality depends heavily on how documents are split before being stored in a vector database.

---

# What is Chunking?

Chunking is the process of splitting a large document into smaller pieces called chunks.

Example:

```text
Internship Handbook

Page 1:
Internship duration is 2 months.

Page 2:
Interns receive a stipend of 15000 rupees.

Page 3:
Mentors conduct weekly reviews.
```

Instead of embedding the entire document as one block:

```text
Document
↓
One Giant Embedding
```

we split it:

```text
Document
↓
Chunking
↓
Chunk 1
Chunk 2
Chunk 3
↓
Embeddings
```

---

# Why Do We Need Chunking?

## 1. Context Window Limitation

LLMs cannot read unlimited text.

```text
1000 Page PDF
↓
Impossible To Send Completely
```

Instead:

```text
Question
↓
Retriever
↓
Top Relevant Chunks
↓
LLM
```

---

## 2. Better Retrieval Accuracy

Without chunking:

```text
Entire PDF
↓
One Embedding
```

The embedding represents multiple topics.

With chunking:

```text
Chunk
↓
One Topic
↓
Better Embedding
```

Result:

Better retrieval quality.

---

## 3. Lower Cost

Smaller chunks:

```text
Less Context
↓
Fewer Tokens
↓
Lower Cost
```

---

## 4. Faster Responses

```text
Less Text
↓
Faster Retrieval
↓
Faster Generation
```

---

# Important Concepts

## Context Window

A context window is the maximum amount of text an LLM can process at one time.

```text
Question
+
Retrieved Chunks
↓
Context Window
↓
LLM
```

Good chunking helps fit relevant information into the context window.

---

## Relevance

Relevance means:

```text
How useful a chunk is
for answering a query
```

Example:

Question:

```text
What stipend do interns receive?
```

Relevant Chunk:

```text
Interns receive a stipend of 15000 rupees.
```

---

## Latency

Latency means:

```text
How long retrieval takes
```

Large chunks:

* More context
* Slower retrieval

Small chunks:

* Faster retrieval
* Less context

---

# Chunk Coherence

Chunk coherence measures:

```text
How well the information
inside a chunk belongs
to the same topic
```

---

## High Coherence

```text
Internship duration is 2 months.

Interns receive a stipend.

Certificates are issued after completion.
```

Topic:

```text
Internship Program
```

Result:

✅ High Coherence

---

## Low Coherence

```text
Internship duration is 2 months.

Employees wear ID cards.

Qdrant is a vector database.

LangChain supports retrievers.
```

Multiple unrelated topics.

Result:

❌ Low Coherence

---

# Why Chunk Coherence Matters

High coherence:

```text
Focused Embedding
↓
Better Similarity Search
↓
Better Retrieval
```

Low coherence:

```text
Mixed Topics
↓
Confused Embedding
↓
Poor Retrieval
```

---

# Chunking Strategy Comparisons

## Fixed Size Chunking

Split using character count.

Example:

```python
chunk_size=500
```

Advantages:

* Simple
* Fast

Disadvantages:

* Can split sentences
* Can lose context

Coherence:

Medium

---

## Semantic Chunking

Split based on meaning.

```text
Sentence
↓
Embedding
↓
Similarity
↓
Group Similar Sentences
```

Advantages:

* Preserves meaning
* Excellent retrieval quality

Disadvantages:

* Slower
* More expensive

Coherence:

Very High

---

## Recursive Chunking

Splits text hierarchically.

```text
Paragraph
↓
Sentence
↓
Words
↓
Characters
```

Advantages:

* Preserves structure
* Works well for PDFs
* Most common LangChain splitter

Disadvantages:

* Slightly slower than fixed-size chunking

Coherence:

High

---

## Sliding Window Chunking

Creates overlapping chunks.

Advantages:

* Preserves context

Disadvantages:

* Redundant information
* Larger storage requirements

Coherence:

High

---

## Structure-Aware Chunking

Uses document structure.

```text
Heading
↓
Paragraph
↓
Table
↓
Code Block
```

Advantages:

* Preserves formatting
* Excellent for documentation

Disadvantages:

* Requires parsing logic

Coherence:

Very High

---

# Recursive Chunking Deep Dive

## Problem With Fixed Size Chunking

Suppose:

```text
Internship duration is 2 months.

Interns receive a stipend of 15000 rupees.

Mentors conduct weekly reviews.
```

Fixed-size chunking may split in the middle of a sentence:

```text
Chunk 1

Internship duration is 2 mo
```

```text
Chunk 2

nths. Interns receive...
```

Problems:

❌ Broken sentences

❌ Lost context

❌ Lower coherence

---

## How Recursive Chunking Works

Instead of splitting blindly, it tries:

```text
Paragraph
↓
Sentence
↓
Word
↓
Character
```

Only when necessary.

This preserves document structure.

---

## Why Recursive Chunking Is Popular

Most production RAG systems use:

```python
RecursiveCharacterTextSplitter
```

because it:

✅ Preserves context

✅ Produces coherent chunks

✅ Works well on PDFs

✅ Works well on documentation

✅ Improves retrieval quality

---

# Recursive Chunking Example

File:

```text
recursive_chunking_demo.py
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Internship Program

The internship duration is 2 months.

Interns receive a stipend of 15000 rupees.

Mentors conduct weekly reviews.

Certificates are issued after completion.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i+1}")

    print(chunk)
```

---

# Recursive Chunking + Qdrant Workflow

```text
PDF
↓
PyPDFLoader
↓
RecursiveCharacterTextSplitter
↓
Chunks
↓
Embeddings
↓
Qdrant
↓
Retriever
↓
Top Matching Chunks
```

This is the same architecture used in Lab 1, but now with Recursive Chunking.

---

# Libraries Used

## langchain_text_splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Purpose:

```text
Large Text
↓
Smaller Chunks
```

---

# Functions Used

## RecursiveCharacterTextSplitter()

Creates the splitter object.

Example:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
```

---

### chunk_size

```python
chunk_size=100
```

Maximum characters per chunk.

---

### chunk_overlap

```python
chunk_overlap=20
```

Characters repeated between chunks.

Purpose:

```text
Preserve Context
```

---

## split_text()

```python
chunks = splitter.split_text(text)
```

Input:

```text
One Large String
```

Output:

```python
[
 chunk1,
 chunk2,
 chunk3
]
```

---

## split_documents()

```python
chunks = splitter.split_documents(documents)
```

Input:

```python
[
 Document(...),
 Document(...)
]
```

Output:

```python
[
 Chunk1,
 Chunk2,
 Chunk3
]
```

Used when working with PDFs.

---

# Day 2 Progress

Completed:

✅ What is Chunking?

✅ Why Chunking Matters

✅ Context Window

✅ Relevance

✅ Latency

✅ Chunk Coherence

✅ Fixed Size Chunking

✅ Semantic Chunking Overview

✅ Recursive Chunking Overview

✅ Recursive Chunking Deep Dive

✅ Recursive Chunking with PDFs

✅ Recursive Chunking with Qdrant

✅ Sliding Window Chunking Overview

✅ Structure-Aware Chunking Overview

✅ Chunk Size Experiment

✅ LangChain Splitter Basics

Next:

➡ Sliding Window Chunking Deep Dive

➡ Structure-Aware Chunking

➡ Semantic Chunking Deep Dive

➡ Chunking Evaluation

➡ Retrieval Accuracy Comparison

➡ Lab 2
