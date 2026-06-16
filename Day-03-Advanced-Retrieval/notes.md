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

# Query Expansion

## What is Query Expansion?

Query Expansion is a retrieval improvement technique where additional related terms are added to the user's query before retrieval.

Goal:

Improve retrieval recall by searching with multiple related terms.

---

## Basic Workflow

User Query
↓
Expand Query
↓
Retrieve Documents
↓
Better Results

---

## Example

Original Query:

What is RAG?

Expanded Query:

What is RAG?

Retrieval Augmented Generation

RAG Architecture

Knowledge Retrieval System

Retrieval Pipeline

---

# Why Query Expansion?

Suppose a document contains:

Internship duration is 2 months.

User asks:

How long is the internship?

The document contains:

duration

The query contains:

long

Query expansion helps bridge this gap.

---

# Benefits

* Improves Recall
* Finds More Relevant Chunks
* Helps BM25 Retrieval
* Helps Hybrid Retrieval

---

# Types of Query Expansion

## 1. Manual Expansion

Developer manually defines related terms.

Example:

```python
query_dictionary = {
    "retrieval": [
        "search",
        "document retrieval"
    ]
}
```

Advantages:

* Easy to implement
* No LLM required

Disadvantages:

* Limited vocabulary
* Requires maintenance

---

## 2. LLM-Based Expansion

An LLM generates multiple related search queries.

Example:

User Query:

What is RAG?

Generated Queries:

What is Retrieval Augmented Generation?

Explain RAG Architecture

Knowledge Retrieval Systems

Retrieval Pipelines

Advantages:

* Dynamic
* Understands semantics
* Scales well

Disadvantages:

* Additional LLM cost
* Slower retrieval

---

## 3. Knowledge-Based Expansion

Uses:

* WordNet
* Domain Dictionaries
* Knowledge Graphs

Advantages:

* Structured

Disadvantages:

* Requires external resources

---

# Multi Query Retrieval

LangChain provides built-in LLM query expansion through:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
```

---

## Workflow

User Query
↓
LLM
↓
Generate Multiple Queries
↓
Retrieve For Each Query
↓
Merge Results
↓
Final Results

---

# Example

User Query:

What is retrieval?

LLM Generated Queries:

What is document retrieval?

How does search work?

Explain retrieval systems.

Information retrieval concepts.

---

# Libraries Used

## MultiQueryRetriever

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
```

Purpose:

Generate multiple search queries using an LLM.

---

# Functions Used

## MultiQueryRetriever.from_llm()

```python
MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)
```

Purpose:

Creates a retriever capable of generating multiple search queries before retrieval.

---

# Query Expansion vs Hybrid Search

Hybrid Search improves:

How retrieval happens.

Query Expansion improves:

What is searched.

---

# Key Takeaways

1. Query Expansion adds related search terms.
2. It improves retrieval recall.
3. Query Expansion can be Manual, LLM-Based, or Knowledge-Based.
4. MultiQueryRetriever is LangChain's built-in implementation.
5. Production RAG systems commonly perform Query Expansion before retrieval.

# Query Rewriting

## What is Query Rewriting?

Query Rewriting is a retrieval improvement technique where the user's query is rewritten into a better version before retrieval.

Goal:

Improve retrieval quality by aligning the user's language with the document language.

---

## Basic Workflow

User Query
↓
LLM Rewriter
↓
Better Query
↓
Retriever
↓
Results

---

## Example 1

User Query:

How long is the internship?

Rewritten Query:

What is the duration of the internship?

---

## Example 2

User Query:

What do interns get paid?

Rewritten Query:

What stipend is provided to interns?

---

## Why Query Rewriting?

Documents often contain formal terminology.

Example:

Document:

Internship duration is 2 months.

User Query:

How long is the internship?

The document contains:

duration

The user uses:

long

Query rewriting helps align the two.

---

# Query Expansion vs Query Rewriting

## Query Expansion

One Query
↓
Many Queries

Example:

What is RAG?

↓

What is RAG?

What is Retrieval Augmented Generation?

Explain RAG Architecture.

Knowledge Retrieval Systems.

Goal:

Improve Recall

---

## Query Rewriting

One Query
↓
One Better Query

Example:

How long is the internship?

↓

What is the duration of the internship?

Goal:

Improve Query Quality

---

# Why Production Systems Use Query Rewriting

Users often ask questions using:

* Informal language
* Abbreviations
* Incomplete sentences
* Different terminology

Examples:

internship length?

what's bm25?

how much do interns earn?

Query rewriting converts these into retrieval-friendly queries.

---

# Query Rewriting Architecture

User Query
↓
LLM Rewriter
↓
Rewritten Query
↓
Hybrid Retrieval
↓
Retrieved Chunks

---

# Example RAG Pipeline

PDF
↓
Chunking
↓
Embeddings
↓
Qdrant

User Query
↓
Query Rewriter
↓
Hybrid Retrieval
↓
Reranker
↓
LLM

---

# Libraries Used

## ChatOpenAI

```python
from langchain_openai import ChatOpenAI
```

Purpose:

Used as the LLM responsible for rewriting the query.

---

## QdrantVectorStore

```python
from langchain_qdrant import QdrantVectorStore
```

Purpose:

Stores embeddings and performs dense retrieval.

---

## HuggingFaceEmbeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

Purpose:

Converts text into embeddings for storage and retrieval.

---

# New Functions Used

## ChatOpenAI()

```python
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)
```

Purpose:

Creates an LLM object.

Parameters:

### model

```python
model="gpt-3.5-turbo"
```

Specifies which model to use.

---

### temperature

```python
temperature=0
```

Makes responses deterministic.

Useful for query rewriting because we want consistent rewrites.

---

## invoke()

```python
response = llm.invoke(
    prompt
)
```

Purpose:

Sends a prompt to the LLM.

Input:

```text
Prompt
```

Output:

```python
AIMessage
```

---

## content

```python
response.content
```

Purpose:

Extracts the generated text from the AIMessage object.

Example:

```python
response = llm.invoke(prompt)

print(response.content)
```

Output:

```text
What is the duration of the internship?
```

---

## as_retriever()

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)
```

Purpose:

Converts Qdrant into a retriever object.

---

## retriever.invoke()

```python
results = retriever.invoke(
    rewritten_query
)
```

Purpose:

Performs retrieval using the rewritten query.

Returns:

Top matching chunks.

---

# Advantages

* Better Retrieval Quality
* Aligns User Language With Documents
* Helps Dense Retrieval
* Helps Hybrid Retrieval
* Easy To Add To Existing Pipelines

---

# Disadvantages

* Requires An LLM
* Additional Cost
* Additional Latency
* Poor Rewrites Can Hurt Retrieval

---

# Interview Questions

### What is Query Rewriting?

Query Rewriting transforms a user query into a better retrieval-friendly query before retrieval.

---

### What is the difference between Query Expansion and Query Rewriting?

Query Expansion generates multiple related queries.

Query Rewriting generates one improved query.

---

### Why is Query Rewriting useful?

Because users and documents often use different terminology.

Query Rewriting helps align them.

---

### Which comes first in a modern RAG pipeline?

Common order:

User Query
↓
Query Rewriting
↓
Query Expansion
↓
Hybrid Retrieval
↓
Reranking

---

# Key Takeaways

1. Query Rewriting improves the quality of a query before retrieval.

2. Query Rewriting produces one improved query.

3. Query Expansion produces multiple related queries.

4. Query Rewriting is commonly powered by an LLM.

5. Query Rewriting is frequently used before Hybrid Retrieval.

6. Production RAG systems often combine Query Rewriting, Query Expansion, Hybrid Retrieval and Reranking.

# Metadata Filtering

## What is Metadata Filtering?

Metadata Filtering is a retrieval technique where retrieval is restricted using metadata conditions before similarity search is performed.

Instead of:

Question
↓
Retrieve From All Chunks

we perform:

Question
+
Metadata Filter
↓
Retrieve Relevant Chunks

---

# Why Metadata Filtering?

Without metadata filtering:

```text
Search Entire Collection
```

Example:

```text
Day 1
Day 2
Day 3
Day 4
Day 5
```

Question:

```text
What chunking strategies are covered?
```

The retriever searches every chunk.

---

With metadata filtering:

```text
Question
+
Day = Day 2
```

Only Day 2 chunks are searched.

---

# What is Metadata?

Metadata means:

```text
Data About Data
```

Example Chunk:

```text
Recursive Chunking preserves structure.
```

Metadata:

```python
{
    "day": "Day 2",
    "topic": "Chunking",
    "page": 5
}
```

---

# Why Metadata Matters

Metadata allows:

* Faster Retrieval
* Better Precision
* Less Noise
* Domain-Specific Search

---

# Example Enterprise Use Cases

## HR Assistant

Documents:

```text
HR Policies
Legal Documents
Engineering Docs
```

Question:

```text
What is the leave policy?
```

Filter:

```python
{
    "department": "HR"
}
```

Only HR documents are searched.

---

## Training Portal

Question:

```text
What chunking strategies exist?
```

Filter:

```python
{
    "day": "Day 2"
}
```

Only Day 2 chunks are searched.

---

# Metadata Filtering Workflow

Documents
↓
Chunking
↓
Metadata Added
↓
Qdrant

Question
+
Filter
↓
Retrieve
↓
Results

---

# Metadata Stored In Qdrant

Remember from Preliminary Day:

Qdrant stores:

```text
Vector
Payload
```

Metadata is stored inside:

```text
Payload
```

Example:

```python
{
    "page_content": "...",
    "metadata": {
        "day": "Day 2"
    }
}
```

---

# Libraries Used

## QdrantClient

```python
from qdrant_client import QdrantClient
```

Purpose:

Connects to Qdrant.

---

## Filter

```python
from qdrant_client.models import Filter
```

Purpose:

Defines retrieval conditions.

---

## FieldCondition

```python
from qdrant_client.models import FieldCondition
```

Purpose:

Specifies which metadata field should be checked.

---

## MatchValue

```python
from qdrant_client.models import MatchValue
```

Purpose:

Performs exact value matching.

---

# New Functions Used

## Filter()

```python
search_filter = Filter(...)
```

Purpose:

Creates retrieval constraints.

Example:

```python
Filter(
    must=[
        ...
    ]
)
```

Meaning:

```text
All conditions must be satisfied.
```

---

## FieldCondition()

```python
FieldCondition(
    key="metadata.day"
)
```

Purpose:

Specifies the metadata field to check.

---

## MatchValue()

```python
MatchValue(
    value="Day 2"
)
```

Purpose:

Matches chunks containing:

```text
Day 2
```

inside metadata.

---

## query_points()

```python
client.query_points(...)
```

Purpose:

Performs vector search directly in Qdrant.

Used instead of:

```python
client.search(...)
```

in newer Qdrant versions.

---

# Metadata Filtering Architecture

User Query
↓
Metadata Filter
↓
Qdrant Search
↓
Filtered Chunks

---

# Advantages

* Faster Retrieval
* Better Precision
* Smaller Search Space
* Enterprise Friendly

---

# Disadvantages

* Requires Metadata Creation
* Additional Design Complexity
* Bad Metadata Leads To Poor Filtering

---

# Interview Questions

### What is Metadata?

Information describing a chunk.

---

### Where is metadata stored in Qdrant?

Inside payloads.

---

### Why use metadata filtering?

To reduce the search space and improve retrieval quality.

---

### What Qdrant objects are used?

* Filter
* FieldCondition
* MatchValue

---

# Key Takeaways

1. Metadata is data about a chunk.

2. Metadata is stored inside Qdrant payloads.

3. Metadata filtering reduces the search space.

4. Metadata filtering improves retrieval precision.

5. Enterprise RAG systems heavily use metadata filtering.

6. Metadata filtering is usually performed before retrieval.

# Structured Retrieval

## What is Structured Retrieval?

Structured Retrieval is a retrieval technique where the system first understands the user's intent and then applies retrieval logic before searching.

Instead of:

Question
↓
Retriever
↓
Results

we perform:

Question
↓
Intent Detection
↓
Metadata Filter Selection
↓
Retriever
↓
Results

---

# Why Structured Retrieval?

Traditional retrieval searches across the entire vector database.

Example:

Question:

What chunking strategies are covered?

Without Structured Retrieval:

Search:

Day 1

Day 2

Day 3

Day 4

Day 5

---

However, we know:

Chunking

belongs to:

Day 2

Only.

Structured Retrieval narrows the search space before retrieval.

---

# Structured Retrieval Workflow

User Query
↓
Intent Detection
↓
Route Selection
↓
Metadata Filter
↓
Retriever
↓
Results

---

# Example

User Query:

What chunking strategies exist?

Intent Detected:

Chunking

Route Selected:

Day 2

Filter Applied:

```python
{
    "day": "Day 2"
}
```

Retriever searches only Day 2 chunks.

---

# Another Example

User Query:

Explain BM25 Retrieval

Intent:

Retrieval

Route:

Day 3

Filter:

```python
{
    "day": "Day 3"
}
```

Only Day 3 chunks are searched.

---

# Structured Retrieval vs Metadata Filtering

## Metadata Filtering

Human provides the filter.

Example:

Question

*

Day 2

↓

Retrieve

---

## Structured Retrieval

System determines the filter.

Example:

Question

↓

Detect Topic

↓

Day 2

↓

Retrieve

---

# Why Enterprises Use Structured Retrieval

Large enterprises may have:

* HR Documents
* Legal Documents
* Engineering Docs
* Training Material
* Financial Reports

Without Structured Retrieval:

Search Everything

With Structured Retrieval:

Determine Department

↓

Search Relevant Department Only

---

# Benefits

## Better Precision

Only relevant chunks are searched.

---

## Lower Latency

Smaller search space.

---

## Lower LLM Cost

Fewer irrelevant chunks retrieved.

---

## Better Scalability

Works well for millions of chunks.

---

# Architecture

PDF
↓
Chunking
↓
Embeddings
↓
Qdrant

User Query
↓
Intent Detection
↓
Metadata Filter
↓
Retriever
↓
Results

---

# Libraries Used

## Filter

```python
from qdrant_client.models import Filter
```

Purpose:

Creates metadata filters.

---

## FieldCondition

```python
from qdrant_client.models import FieldCondition
```

Purpose:

Specifies metadata fields.

---

## MatchValue

```python
from qdrant_client.models import MatchValue
```

Purpose:

Performs exact matching.

---

# New Concepts Learned

## Intent Detection

Determining what the user is asking about.

Example:

```text
chunking
```

↓

```text
Day 2
```

---

## Route Selection

Choosing which subset of documents should be searched.

---

## Rule-Based Routing

Example:

```python
if "chunking" in query:
    day = "Day 2"
```

This is the simplest form of structured retrieval.

---

# Functions Used

## query_points()

```python
client.query_points(...)
```

Purpose:

Performs filtered vector search in Qdrant.

---

## embed_query()

```python
embedding_model.embed_query(
    query
)
```

Purpose:

Converts the query into an embedding vector.

---

# Advantages

* Better Precision
* Faster Retrieval
* Lower Search Cost
* Better Enterprise Scalability

---

# Disadvantages

* Requires Metadata
* Requires Routing Logic
* More Complex Architecture

---

# Interview Questions

### What is Structured Retrieval?

Structured Retrieval combines intent detection and metadata filtering before retrieval.

---

### How is Structured Retrieval different from Metadata Filtering?

Metadata Filtering requires the user to provide filters.

Structured Retrieval determines filters automatically.

---

### Why use Structured Retrieval?

To improve precision, reduce latency and scale retrieval systems.

---

# Key Takeaways

1. Structured Retrieval applies routing before retrieval.

2. Structured Retrieval narrows the search space.

3. Structured Retrieval usually uses metadata filters.

4. Structured Retrieval improves retrieval precision.

5. Enterprise RAG systems commonly use Structured Retrieval.

6. Structured Retrieval often combines Intent Detection + Metadata Filtering.

Next:

➡ Bi-Encoders
➡ Cross-Encoders
➡ Reranking
# Bi-Encoders vs Cross-Encoders

## Why Do We Need Reranking?

Suppose retrieval returns:

Chunk A

Chunk B

Chunk C

Chunk D

Chunk E

All chunks are relevant.

However:

Chunk C

may actually be the best answer.

The retriever does not always rank results perfectly.

This is where reranking is used.

---

# Retrieval vs Reranking

## Retrieval

Goal:

Find candidate chunks quickly.

Uses:

Bi-Encoders

Examples:

* all-MiniLM-L6-v2
* BGE Embeddings
* OpenAI Embeddings

---

## Reranking

Goal:

Improve the ordering of retrieved chunks.

Uses:

Cross-Encoders

Examples:

* cross-encoder/ms-marco-MiniLM-L-6-v2
* Cohere Rerank
* BGE Reranker

---

# What is a Bi-Encoder?

A Bi-Encoder encodes:

Query

and

Document

separately.

Workflow:

Query
↓
Encoder
↓
Query Embedding

Document
↓
Encoder
↓
Document Embedding

↓

Cosine Similarity

↓

Score

---

# Example

Query:

What is BM25?

Document:

BM25 is a ranking algorithm.

Workflow:

```text
Query
↓
Embedding

Document
↓
Embedding

↓

Cosine Similarity
```

---

# Advantages of Bi-Encoders

* Very Fast
* Scales To Millions Of Chunks
* Ideal For Retrieval

---

# Disadvantages of Bi-Encoders

* Limited Understanding Of Query-Document Relationships
* May Rank Chunks Incorrectly

---

# What is a Cross-Encoder?

A Cross-Encoder processes:

Query

and

Document

together.

Workflow:

Query

*

Document

↓

Transformer

↓

Relevance Score

---

# Example

Input:

What is BM25?

[SEP]

BM25 is a ranking algorithm.

↓

Cross Encoder

↓

Relevance Score

---

# Why Cross-Encoders Are Better

Cross-Encoders examine:

* Word Relationships
* Context Relationships
* Exact Relevance

between:

Query

and

Document

simultaneously.

---

# Why Not Use Cross-Encoders For Retrieval?

Suppose:

1,000,000 Chunks

For every query:

Cross Encoder must evaluate:

Query + Chunk

for all chunks.

This is computationally expensive.

---

# Production Solution

Use:

Bi-Encoder

↓

Top 20 Chunks

↓

Cross Encoder

↓

Top 5 Chunks

This process is called:

Reranking

---

# Reranking Workflow

User Query
↓
Retriever
↓
Top K Chunks
↓
Cross Encoder
↓
Relevance Scores
↓
Sorted Results

---

# Lab 4 Implementation

Pipeline:

PDF
↓
Chunking
↓
Embeddings
↓
Qdrant

User Query
↓
Dense Retrieval
↓
Top 10 Chunks
↓
Cross Encoder
↓
Top 5 Reranked Chunks

---

# Libraries Used

## CrossEncoder

```python
from sentence_transformers import CrossEncoder
```

Purpose:

Provides reranking capabilities.

---

# New Functions Used

## CrossEncoder()

```python
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

Purpose:

Loads the reranking model.

---

## predict()

```python
scores = reranker.predict(
    pairs
)
```

Input:

```python
[
    (query, document)
]
```

Output:

```python
[
    9.25,
    7.81,
    2.14
]
```

Higher score:

More relevant document.

---

# Bi-Encoder vs Cross-Encoder

| Feature          | Bi-Encoder | Cross-Encoder |
| ---------------- | ---------- | ------------- |
| Speed            | Very Fast  | Slow          |
| Scalability      | Excellent  | Poor          |
| Retrieval        | Yes        | No            |
| Reranking        | No         | Yes           |
| Accuracy         | Good       | Excellent     |
| Production Usage | Retrieval  | Reranking     |

---

# Advantages Of Reranking

* Better Ordering Of Results
* Improved Precision
* Better Context For LLM
* Reduces Retrieval Errors

---

# Disadvantages

* Additional Latency
* Additional Compute Cost
* Not Suitable For Searching Millions Of Chunks Directly

---

# Interview Questions

### What is a Bi-Encoder?

A model that independently encodes queries and documents into embeddings.

---

### What is a Cross-Encoder?

A model that processes queries and documents together to generate a relevance score.

---

### Why are Cross-Encoders used after retrieval?

Because they are more accurate but computationally expensive.

---

### What is reranking?

Reordering retrieved documents using a more accurate relevance model.

---

# Key Takeaways

1. Bi-Encoders power retrieval.

2. Cross-Encoders power reranking.

3. Retrieval finds candidate chunks.

4. Reranking improves chunk ordering.

5. Most production RAG systems use Bi-Encoder Retrieval + Cross-Encoder Reranking.

Next:

➡ Recall@K

➡ MRR

➡ NDCG
