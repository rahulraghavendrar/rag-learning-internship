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

# Sliding Window Chunking Deep Dive

## Problem Sliding Window Solves

Suppose a document contains:

```text
Sentence 1:
Internship duration is 2 months.

Sentence 2:
Interns receive a stipend of 15000 rupees.

Sentence 3:
Payment is made monthly.

Sentence 4:
Certificates are issued after completion.
```

If normal chunking is used:

### Chunk 1

```text
Sentence 1

Sentence 2
```

### Chunk 2

```text
Sentence 3

Sentence 4
```

Question:

```text
How is the stipend paid?
```

The information is split across chunks.

Chunk 1 contains:

```text
Stipend Information
```

Chunk 2 contains:

```text
Payment Information
```

The retriever may not retrieve both chunks.

---

## How Sliding Window Chunking Works

Sliding Window Chunking introduces overlap between chunks.

Instead of:

```text
Chunk 1

A
B
C

Chunk 2

D
E
F
```

It creates:

```text
Chunk 1

A
B
C

Chunk 2

B
C
D

Chunk 3

C
D
E

Chunk 4

D
E
F
```

Notice:

```text
B
C
```

appear in both Chunk 1 and Chunk 2.

This repeated information is called:

```text
Overlap
```

---

## Why Overlap Matters

Without overlap:

```text
Important Information
↓
Split Between Chunks
↓
Lost Context
```

With overlap:

```text
Important Information
↓
Appears In Multiple Chunks
↓
Better Retrieval
```

This improves question answering quality.

---

## Real World Example

Chunk 1:

```text
Interns receive a stipend of 15000 rupees.
```

Chunk 2:

```text
Payment is made monthly.
```

Question:

```text
How is the stipend paid?
```

Without overlap:

```text
Retriever May Return Only Chunk 1
```

Answer becomes incomplete.

With overlap:

```text
Interns receive a stipend of 15000 rupees.

Payment is made monthly.
```

Both pieces of information stay together.

---

## Advantages

✅ Better context preservation

✅ Better retrieval quality

✅ Better question answering

✅ Handles chunk boundaries effectively

---

## Disadvantages

❌ More chunks generated

❌ Increased storage requirements

❌ More embeddings

❌ Slower indexing

---

## Sliding Window vs Recursive Chunking

| Feature               | Recursive | Sliding Window |
| --------------------- | --------- | -------------- |
| Preserves Structure   | ✅         | ❌              |
| Preserves Context     | Good      | Excellent      |
| Storage Cost          | Lower     | Higher         |
| Duplicate Information | No        | Yes            |
| Retrieval Quality     | High      | Very High      |

---

## LangChain Implementation

LangChain does not provide a separate Sliding Window Splitter.

Instead:

```python
RecursiveCharacterTextSplitter
```

acts as a sliding window splitter when:

```python
chunk_overlap > 0
```

Example:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=30
)
```

---

## Sliding Window Example

File:

```text
sliding_window_demo.py
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Internship duration is 2 months.

Interns receive a stipend of 15000 rupees.

Payment is made monthly.

Certificates are issued after completion.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=30
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i+1}")

    print(chunk)

    print("-"*50)
```

---

## Understanding chunk_overlap

Example:

```python
chunk_size=100
chunk_overlap=20
```

Chunk 1:

```text
Characters 1 → 100
```

Chunk 2:

```text
Characters 81 → 180
```

Characters:

```text
81 → 100
```

appear in both chunks.

This preserves context between adjacent chunks.

---

## Sliding Window + PDF Workflow

```text
PDF
↓
PyPDFLoader
↓
RecursiveCharacterTextSplitter
(chunk_overlap > 0)
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

---

## Sliding Window + Qdrant Workflow

```text
PDF
↓
Sliding Window Chunking
↓
Chunks
↓
Embeddings
↓
Qdrant
↓
Semantic Search
↓
Top-K Chunks
```

The retrieval pipeline remains the same as Day 1.

Only the chunking strategy changes.

---

## When Should Sliding Window Be Used?

Best For:

✅ Question Answering Systems

✅ Customer Support Chatbots

✅ Long PDF Retrieval

✅ RAG Applications

Avoid When:

❌ Storage is limited

❌ Indexing speed is critical

❌ Duplicate chunks are undesirable

---

## Key Takeaways

1. Sliding Window Chunking introduces overlap between chunks.

2. Overlap helps preserve context near chunk boundaries.

3. Better context usually leads to better retrieval quality.

4. The tradeoff is additional storage and indexing cost.

5. In LangChain, Sliding Window Chunking is implemented using:

```python
RecursiveCharacterTextSplitter(
    chunk_overlap > 0
)
```

6. Sliding Window Chunking is one of the most common strategies used in production RAG systems.

# Structure-Aware Chunking Deep Dive

## What is Structure-Aware Chunking?

Structure-Aware Chunking splits documents based on their logical structure rather than character counts.

Instead of using:

```text
Character Count
Paragraph Length
Sentence Length
```

it uses:

```text
Headings
Subheadings
Tables
Lists
Code Blocks
Sections
```

to create chunks.

---

# Why Do We Need Structure-Aware Chunking?

Consider a company policy PDF:

```text
Internship Policy

Duration

The internship lasts 2 months.

Stipend

Interns receive 15000 rupees.

Certification

Certificates are issued after completion.
```

A normal splitter may create:

```text
Chunk 1

Internship Policy

Duration

The internship lasts 2 months.

Stipend
```

```text
Chunk 2

Interns receive 15000 rupees.

Certification

Certificates are issued.
```

The topic "Stipend" gets separated from its content.

---

# Structure-Aware Chunking Output

Instead:

### Chunk 1

```text
Duration

The internship lasts 2 months.
```

### Chunk 2

```text
Stipend

Interns receive 15000 rupees.
```

### Chunk 3

```text
Certification

Certificates are issued after completion.
```

Now each chunk represents one complete topic.

---

# Why Is This Better?

The embedding becomes focused.

Example:

```text
Stipend

Interns receive 15000 rupees.
```

Embedding Meaning:

```text
Stipend Topic
```

Instead of:

```text
Duration
+
Stipend
+
Certification
```

inside a single embedding.

This improves:

✅ Chunk Coherence

✅ Retrieval Quality

✅ Question Answering Accuracy

---

# Real World Use Cases

Structure-Aware Chunking is commonly used for:

### Documentation

```text
Installation

Configuration

Deployment
```

---

### Research Papers

```text
Abstract

Methodology

Results

Conclusion
```

---

### Company Policies

```text
Attendance Policy

Leave Policy

Dress Code
```

---

### Technical Documentation

```text
API Reference

Authentication

Endpoints

Examples
```

---

# Built-In Approaches

## HTMLHeaderTextSplitter

Used for HTML documents.

Library:

```python
from langchain_text_splitters import HTMLHeaderTextSplitter
```

Splits using:

```html
<h1>
<h2>
<h3>
```

tags.

---

### Example

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers = [
    ("h1","Main Header"),
    ("h2","Section Header")
]

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers
)
```

---

### Workflow

```text
HTML
↓
Headers
↓
Chunks
↓
Embeddings
↓
Vector Database
```

---

# Can HTMLHeaderTextSplitter Split PDFs?

No.

Reason:

```text
PDF
```

does not contain:

```html
<h1>
<h2>
<h3>
```

tags.

It only works on HTML documents.

---

# Unstructured Library

For PDFs, the most common production solution is:

```python
from unstructured.partition.pdf import partition_pdf
```

---

## What Does Unstructured Do?

It understands document elements such as:

```text
Title

NarrativeText

Table

List

Header

Footer
```

instead of treating everything as plain text.

---

# Example

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="internship.pdf"
)
```

Output:

```text
Title

Internship Program
```

```text
Title

Duration
```

```text
NarrativeText

The internship lasts 2 months.
```

```text
Title

Stipend
```

```text
NarrativeText

Interns receive 15000 rupees.
```

---

# Why Unstructured Is Important

Compared to:

### Fixed Size Chunking

```text
Split By Character Count
```

### Recursive Chunking

```text
Split By Paragraphs
Sentences
Words
```

### Sliding Window Chunking

```text
Split With Overlap
```

Unstructured performs:

```text
Document Understanding
```

before chunk creation.

---

# Structure-Aware RAG Pipeline

```text
PDF
↓
Unstructured
↓
Title
Paragraph
Table
Code
↓
Chunks
↓
Embeddings
↓
Qdrant
↓
Retriever
↓
Top-K Results
```

---

# Libraries Used

## partition_pdf

```python
from unstructured.partition.pdf import partition_pdf
```

Purpose:

```text
PDF
↓
Structured Elements
```

---

## HTMLHeaderTextSplitter

```python
from langchain_text_splitters import HTMLHeaderTextSplitter
```

Purpose:

```text
HTML
↓
Header-Based Chunks
```

---

# Functions Used

## partition_pdf()

```python
elements = partition_pdf(
    filename="internship.pdf"
)
```

Input:

```text
PDF
```

Output:

```python
[
 Title,
 NarrativeText,
 Table,
 ListItem
]
```

---

## HTMLHeaderTextSplitter()

```python
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers
)
```

Creates a splitter that uses HTML headers.

---

## split_text()

```python
documents = splitter.split_text(html_text)
```

Input:

```text
HTML
```

Output:

```python
[
 Document(...),
 Document(...)
]
```

---

# Advantages

✅ Excellent Chunk Coherence

✅ Preserves Document Meaning

✅ Better Retrieval Quality

✅ Works Very Well For PDFs

✅ Common In Production RAG Systems

---

# Disadvantages

❌ More Complex

❌ Requires Document Parsing

❌ Slower Than Fixed Size Chunking

❌ Some PDFs Have Poor Structure

---

# Key Takeaways

1. Structure-Aware Chunking uses document structure instead of character counts.

2. HTMLHeaderTextSplitter is used for HTML documents.

3. HTMLHeaderTextSplitter cannot directly process PDFs.

4. Unstructured is the most common solution for PDF Structure-Aware Chunking.

5. Structure-Aware Chunking usually produces highly coherent chunks.

6. Better chunk coherence generally leads to better retrieval quality.

7. This is one of the most commonly used chunking strategies in production RAG systems.

# Semantic Chunking Deep Dive

## What is Semantic Chunking?

Semantic Chunking splits text based on meaning rather than character count, document structure, or overlap.

Instead of asking:

```text
How many characters are present?
```

it asks:

```text
Do these sentences talk about the same topic?
```

If the meaning is similar:

```text
Same Chunk
```

If the meaning changes:

```text
Create New Chunk
```

---

# Why Do We Need Semantic Chunking?

Consider the following document:

```text
Internship duration is 2 months.

Interns receive a stipend of 15000 rupees.

Mentors conduct weekly reviews.

Qdrant is a vector database.

Embeddings convert text into vectors.

Cosine similarity measures semantic similarity.
```

---

## Fixed Size Chunking Might Produce

```text
Chunk 1

Internship duration is 2 months.

Interns receive a stipend.
```

```text
Chunk 2

Mentors conduct reviews.

Qdrant is a vector database.
```

```text
Chunk 3

Embeddings convert text into vectors.
```

Notice:

```text
Chunk 2
```

contains:

```text
Internship Topic
+
Vector Database Topic
```

Result:

❌ Low Coherence

---

## Semantic Chunking Produces

```text
Chunk 1

Internship duration is 2 months.

Interns receive a stipend.

Mentors conduct weekly reviews.
```

```text
Chunk 2

Qdrant is a vector database.

Embeddings convert text into vectors.

Cosine similarity measures semantic similarity.
```

Result:

✅ High Coherence

---

# How Semantic Chunking Works

## Step 1

Split the document into sentences.

```text
Sentence 1

Sentence 2

Sentence 3
```

---

## Step 2

Generate embeddings.

```text
Sentence
↓
Embedding
```

Example:

```python
embedding = model.encode(sentence)
```

---

## Step 3

Compare sentence similarity.

```text
Sentence A
↔
Sentence B
↓
Cosine Similarity
```

---

## Step 4

Decision

If similarity is high:

```text
Same Chunk
```

If similarity is low:

```text
New Chunk
```

---

# Semantic Chunking Workflow

```text
Document
↓
Sentence Split
↓
Embeddings
↓
Cosine Similarity
↓
Group Similar Sentences
↓
Chunks
```

---

# Why Semantic Chunking Is Powerful

Unlike Fixed Size Chunking:

```text
Split Every 500 Characters
```

Semantic Chunking asks:

```text
Do These Sentences Mean Similar Things?
```

This usually creates:

✅ Better Chunks

✅ Better Retrieval

✅ Better Answers

---

# LangChain SemanticChunker

LangChain provides a built-in semantic chunker.

Library:

```python
from langchain_experimental.text_splitter import SemanticChunker
```

Purpose:

```text
Document
↓
Embeddings
↓
Semantic Similarity
↓
Topic-Based Chunks
```

---

# Example

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=embedding_model
)

chunks = semantic_splitter.split_documents(
    documents
)
```

---

# Semantic Chunking + PDF Workflow

```text
PDF
↓
PyPDFLoader
↓
SemanticChunker
↓
Semantic Chunks
↓
Embeddings
↓
Qdrant
↓
Retriever
```

---

# Semantic Chunking + Qdrant Workflow

```text
PDF
↓
Semantic Chunking
↓
Chunks
↓
Embeddings
↓
Qdrant
↓
Top-K Retrieval
```

The retrieval pipeline remains identical.

Only chunk creation changes.

---

# Libraries Used

## SemanticChunker

```python
from langchain_experimental.text_splitter import SemanticChunker
```

Purpose:

```text
Meaning-Based Chunking
```

---

## HuggingFaceEmbeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

Purpose:

```text
Text
↓
Embedding
```

Used internally by SemanticChunker.

---

# Functions Used

## SemanticChunker()

```python
semantic_splitter = SemanticChunker(
    embeddings=embedding_model
)
```

Creates the semantic splitter.

---

## split_documents()

```python
chunks = semantic_splitter.split_documents(
    documents
)
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

---

# Comparison of Chunking Strategies

| Strategy        | Splits By                   | Retrieval Quality |
| --------------- | --------------------------- | ----------------- |
| Fixed Size      | Character Count             | Medium            |
| Recursive       | Paragraph → Sentence → Word | High              |
| Sliding Window  | Overlap                     | High              |
| Structure-Aware | Headers / Sections          | High              |
| Semantic        | Meaning                     | Very High         |

---

# Advantages

✅ Excellent Chunk Coherence

✅ Excellent Retrieval Quality

✅ Topic-Based Chunks

✅ Better Question Answering

✅ Works Very Well For RAG

---

# Disadvantages

❌ Slower Than Fixed Size Chunking

❌ Requires Embedding Generation

❌ More Expensive

❌ More Complex

---

# Interview Questions

### What is Semantic Chunking?

Semantic Chunking splits text based on meaning using embeddings and similarity scores instead of character counts.

---

### Why is Semantic Chunking useful?

Because it creates highly coherent chunks that usually improve retrieval quality and question-answering performance.

---

### Does LangChain support Semantic Chunking?

Yes.

LangChain provides:

```python
from langchain_experimental.text_splitter import SemanticChunker
```

for built-in semantic chunking.

---

# Key Takeaways

1. Semantic Chunking groups sentences based on meaning.

2. It uses embeddings and similarity scores.

3. LangChain provides SemanticChunker for semantic chunking.

4. Semantic Chunking usually produces the highest chunk coherence.

5. Higher chunk coherence often leads to better retrieval quality.

6. Semantic Chunking is one of the most powerful chunking strategies used in modern RAG systems.

# Lab 2 - Comparing Chunking Strategies

## Goal

Compare different chunking strategies on the same dataset and measure retrieval quality.

The purpose of this lab is not to determine which chunking strategy creates the fewest chunks or the largest chunks.

The purpose is to determine:

```text
Which chunking strategy retrieves the most relevant information?
```

---

# Objective

Compare:

1. Recursive Chunking
2. Sliding Window Chunking
3. Semantic Chunking

using:

* Same PDFs
* Same Embedding Model
* Same Vector Database
* Same Questions

This ensures a fair comparison.

---

# Dataset

The dataset consists of:

```text
internship.pdf

company_policy.pdf

employee_handbook.pdf

training_program.pdf

ai_projects.pdf
```

All chunking strategies use the exact same documents.

---

# Evaluation Pipeline

```text
PDFs
↓
Chunking Strategy
↓
Chunks
↓
Embeddings
↓
Qdrant
↓
Retriever
↓
Questions
↓
Retrieved Chunks
↓
Evaluation
↓
Accuracy Score
```

---

# Chunking Strategies Tested

## Recursive Chunking

Uses:

```python
RecursiveCharacterTextSplitter
```

Splits hierarchically:

```text
Paragraph
↓
Sentence
↓
Word
↓
Character
```

---

## Sliding Window Chunking

Uses:

```python
RecursiveCharacterTextSplitter
```

with:

```python
chunk_overlap > 0
```

Creates overlapping chunks.

Purpose:

```text
Preserve Context
```

---

## Semantic Chunking

Uses:

```python
SemanticChunker
```

Splits based on:

```text
Meaning
```

instead of:

```text
Character Count
```

---

# Questions Used

Example:

```python
questions = [

("How long is the internship?",
 "2 months"),

("What stipend do interns receive?",
 "15000"),

("How are mentors assigned?",
 "mentor"),

("How many training modules exist?",
 "module"),

("What AI projects are available?",
 "project")
]
```

Each question contains:

```text
Question
+
Expected Keyword
```

---

# How Retrieval Is Evaluated

Question:

```text
What stipend do interns receive?
```

Expected Keyword:

```text
15000
```

Retrieved Chunk:

```text
Interns receive a stipend of 15000 rupees.
```

Result:

```text
Correct Retrieval
```

Score:

```text
1
```

---

Question:

```text
What stipend do interns receive?
```

Retrieved Chunk:

```text
Internship duration is 2 months.
```

Result:

```text
Incorrect Retrieval
```

Score:

```text
0
```

---

# Comparison Function

The main comparison function is:

```python
evaluate_strategy()
```

Purpose:

```text
Chunking Strategy
↓
Store In Qdrant
↓
Ask Questions
↓
Retrieve Chunks
↓
Calculate Accuracy
```

---

# Important Functions Used

## PyPDFLoader()

```python
loader = PyPDFLoader(pdf)
```

Purpose:

```text
PDF
↓
Documents
```

---

## RecursiveCharacterTextSplitter()

```python
RecursiveCharacterTextSplitter()
```

Purpose:

```text
Text
↓
Recursive Chunks
```

---

## SemanticChunker()

```python
SemanticChunker()
```

Purpose:

```text
Text
↓
Meaning-Based Chunks
```

---

## add_documents()

```python
vectorstore.add_documents()
```

Purpose:

```text
Chunks
↓
Embeddings
↓
Qdrant
```

Stores chunks in the vector database.

---

## as_retriever()

```python
vectorstore.as_retriever()
```

Creates a retriever object.

---

## retriever.invoke()

```python
retriever.invoke(question)
```

Purpose:

```text
Question
↓
Similarity Search
↓
Top-K Chunks
```

This is the actual retrieval function used in the comparison.

---

# Accuracy Formula

```text
Accuracy

=

Correct Retrievals
------------------
Total Questions

× 100
```

Example:

```text
5 Questions

4 Correct
```

Accuracy:

```text
80%
```

---

# Example Results

```text
Recursive Chunking

Accuracy = 80%
```

```text
Sliding Window Chunking

Accuracy = 100%
```

```text
Semantic Chunking

Accuracy = 100%
```

---

# What Is Actually Being Compared?

We are comparing:

```text
Retrieval Quality
```

NOT:

```text
Number Of Chunks
```

NOT:

```text
Chunk Size
```

NOT:

```text
LLM Answer Quality
```

The only thing being measured is:

```text
Did the retriever find the correct chunk?
```

---

# Why This Lab Is Important

Chunking is one of the biggest factors affecting RAG performance.

Better chunking usually leads to:

✅ Better Chunk Coherence

✅ Better Retrieval

✅ Better Context

✅ Better Answers

---

# Limitations Of This Lab

This lab uses:

```text
Keyword Matching
```

to determine correctness.

Example:

```python
if keyword in retrieved_text
```

This is simple but not perfect.

---

# Professional Retrieval Metrics

Real production RAG systems use:

## Recall@K

Measures:

```text
Was the correct chunk present
in the top K results?
```

---

## Precision@K

Measures:

```text
How many retrieved chunks
were actually relevant?
```

---

## MRR

Mean Reciprocal Rank

Measures:

```text
How high the correct chunk
appears in the ranking.
```

---

## NDCG

Measures:

```text
Ranking Quality
```

for multiple relevant documents.

---

# Key Takeaways

1. Chunking quality directly affects retrieval quality.

2. Recursive, Sliding Window, and Semantic Chunking can be compared fairly using the same dataset.

3. Retrieval quality is measured by checking whether the correct information is retrieved.

4. The comparison function used in this lab is:

```python
evaluate_strategy()
```

5. The retrieval function used in this lab is:

```python
retriever.invoke()
```

6. Accuracy is calculated using:

```text
Correct Retrievals / Total Questions
```

7. Professional RAG systems use Recall@K, Precision@K, MRR, and NDCG instead of simple keyword matching.
