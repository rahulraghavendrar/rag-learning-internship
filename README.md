# RAG Learning Internship

## Status

### ✅ Preliminary Foundation Completed

Topics Covered:

* Embeddings
* Cosine Similarity
* Fixed Size Chunking
* Semantic Chunking
* Manual Retrieval
* ChromaDB
* Qdrant
* RAG Fundamentals
* Top-K Retrieval
* Precision & Recall
* Hallucinations
* Context Window
* LangChain Fundamentals
* PyPDFLoader
* RecursiveCharacterTextSplitter
* HuggingFaceEmbeddings
* Qdrant Vector Store
* Retriever

---

## ✅ Day 01 Completed - RAG Foundations & Architecture

### Topics Covered

* What is RAG?

* Why Not Fine-Tuning?

* RAG vs Fine-Tuning vs Prompt Engineering

* Core RAG Pipeline

  * Index
  * Retrieve
  * Augment
  * Generate

* Vector Databases

* Embedding Models

* Similarity Search

* Indexing

* Real World RAG Applications

* Lewis et al. (2020) RAG Paper Overview

---

### Lab 1 - Naive RAG Pipeline

Built a complete Naive RAG System.

Components:

* PyPDFLoader
* Chunking
* HuggingFace Embeddings
* Qdrant Vector Database
* Retriever

Deliverables:

* Multi-PDF Search
* Semantic Retrieval
* Top-K Retrieval
* Chunk ID Tracking
* Chunk Deletion Support

---

## ✅ Day 02 Completed - Chunking Strategies & Text Processing

### Topics Covered

#### Chunking Fundamentals

* What is Chunking?
* Why Chunking Matters
* Context Window
* Relevance
* Latency
* Chunk Coherence

---

### Fixed Size Chunking

Concepts:

* Character-Based Splitting
* Chunk Size Selection
* Tradeoffs

Pros:

* Simple
* Fast

Cons:

* Semantic Breaks
* Context Loss

---

### Recursive Chunking

Concepts:

* Hierarchical Splitting

```text
Paragraph
↓
Sentence
↓
Word
↓
Character
```

Libraries:

* RecursiveCharacterTextSplitter

Applications:

* PDFs
* Documentation
* Production RAG Systems

---

### Sliding Window Chunking

Concepts:

* Overlapping Chunks
* Context Preservation
* chunk_overlap

Applications:

* Question Answering
* Long PDFs
* Customer Support Systems

---

### Structure-Aware Chunking

Concepts:

* Headers
* Sections
* Tables
* Code Blocks

Libraries:

* HTMLHeaderTextSplitter
* Unstructured

Applications:

* PDFs
* Research Papers
* Documentation
* Company Policies

---

### Semantic Chunking

Concepts:

* Meaning-Based Splitting
* Embeddings
* Similarity Scores
* Topic-Based Chunks

Libraries:

* SemanticChunker
* SentenceTransformers

Applications:

* High Accuracy Retrieval
* Advanced RAG Systems

---

### Chunking Strategy Comparison

Compared:

* Fixed Size Chunking
* Recursive Chunking
* Sliding Window Chunking
* Structure-Aware Chunking
* Semantic Chunking

Studied:

* Chunk Coherence
* Retrieval Quality
* Retrieval Accuracy
* Chunking Tradeoffs

---

### Lab 2 - Chunking Strategy Evaluation

Built a complete chunking comparison system.

Pipeline:

```text
PDFs
↓
Chunking Strategy
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
Accuracy Evaluation
```

Strategies Compared:

* Recursive Chunking
* Sliding Window Chunking
* Semantic Chunking

Evaluation Method:

* Keyword Matching Accuracy

Important Functions:

* evaluate_strategy()
* retriever.invoke()

Deliverables:

* Retrieval Quality Comparison
* Retrieval Accuracy Comparison
* Chunking Evaluation Framework

---

## Roadmap Progress

✅ Preliminary Foundation

✅ Day 01 - RAG Foundations & Architecture

✅ Day 02 - Chunking Strategies & Text Processing

⬜ Day 03 - Advanced Retrieval & Reranking

⬜ Day 04 - RAG Evaluation & Metrics

⬜ Day 05 - Advanced RAG Architectures & Capstone Project

---

## Upcoming Topics

### Day 03 - Advanced Retrieval & Reranking

Topics:

* BM25 Retrieval
* Dense Retrieval
* Hybrid Search
* Query Expansion
* Query Rewriting
* Metadata Filtering
* Structured Retrieval
* Reranking Models

  * Bi-Encoders
  * Cross-Encoders

Labs:

* Lab 3 - Hybrid Retrieval (BM25 + Dense Search)
* Lab 4 - Reranking Pipeline

Metrics:

* Recall@K
* MRR
* NDCG

---

### Day 04 - RAG Evaluation & Metrics

Topics:

* Precision
* Recall
* MRR
* NDCG
* BLEU
* ROUGE
* BERTScore
* Faithfulness
* Relevance
* Consistency

Tools:

* RAGAS
* TruLens
* Langfuse

Lab:

* Evaluation Pipeline

---

### Day 05 - Advanced RAG Architectures & Capstone

Topics:

* Naive RAG
* Advanced RAG
* Hybrid RAG
* Agentic RAG
* Graph RAG
* Multi-Modal RAG

Capstone:

* End-to-End Production RAG System
* Hybrid Retrieval
* Reranking
* Evaluation
* FastAPI Deployment
* Qdrant Integration

---

## Current Status

```text
Foundation        ✅ Completed
Day 01            ✅ Completed
Day 02            ✅ Completed
Day 03            ✅ Completed
Day 04            ✅ Completed
Day 05            ✅ Completed
```
---

## ✅ MCP Integration

### Objective

Expose the retrieval step of the RAG pipeline as an MCP Tool using FastMCP.

### Topics Covered

- Model Context Protocol (MCP)
- MCP Server
- MCP Tool Registration
- FastMCP
- MCP Inspector
- Tool Invocation
- LangChain Integration
- Qdrant Retrieval through MCP

---

### MCP Pipeline

```text
Question
↓
MCP Tool
(retrieve_documents)
↓
LangChain Retriever
↓
Qdrant Vector Database
↓
Top-K Retrieved Chunks
