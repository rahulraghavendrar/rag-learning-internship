# Day 01 - RAG Foundations & Architecture

## Goal

Understand the Retrieval-Augmented Generation (RAG) pipeline end-to-end and build a basic retrieval system using embeddings, vector databases, and semantic search.

---

# What is RAG?

RAG stands for:

Retrieval-Augmented Generation

RAG combines:

1. Retrieval of relevant information
2. Generation of answers using an LLM

Instead of expecting the LLM to memorize all knowledge, RAG retrieves information from external sources and provides it to the model before generating an answer.

Traditional LLM:

Question
↓
LLM
↓
Answer

RAG:

Question
↓
Retrieve Information
↓
Provide Context
↓
LLM
↓
Answer

---

# Why Not Just Fine-Tune?

Fine-tuning stores knowledge inside model weights.

Workflow:

Documents
↓
Fine-Tune Model
↓
Knowledge Stored In Weights

Problems:

* Expensive
* Requires retraining for updates
* Knowledge becomes outdated
* Difficult to maintain large knowledge bases

Example:

If a company policy changes, the model must be retrained.

---

# Why RAG Is Preferred

RAG keeps knowledge external.

Workflow:

Documents
↓
Embeddings
↓
Vector Database
↓
Retrieval
↓
LLM

Benefits:

* Easy to update
* No retraining required
* Scalable
* Lower cost
* Reduces hallucinations

When documents change:

Documents
↓
Re-index
↓
Done

No model retraining is needed.

---

# RAG vs Fine-Tuning vs Prompt Engineering

## Prompt Engineering

Approach:

Question + Prompt

Advantages:

* Fastest implementation
* No training required

Disadvantages:

* Limited context window
* Not suitable for large knowledge bases
* Knowledge must fit inside prompts

---

## Fine-Tuning

Approach:

Store knowledge inside model weights.

Advantages:

* Learns specific behavior
* Useful for domain adaptation

Disadvantages:

* Expensive
* Hard to update
* Requires retraining

---

## RAG

Approach:

Store knowledge externally and retrieve it when needed.

Advantages:

* Easy updates
* Lower cost
* Scalable
* Better factual grounding

Disadvantages:

* Requires retrieval infrastructure
* Retrieval quality impacts answer quality

---

# Core RAG Pipeline

## Step 1: Index

Prepare documents for retrieval.

Documents
↓
Chunking
↓
Embeddings
↓
Vector Database

This stage happens before users ask questions.

---

## Step 2: Retrieve

User asks a question.

Question
↓
Embedding
↓
Similarity Search
↓
Relevant Chunks

---

## Step 3: Augment

Combine:

Question
+
Retrieved Chunks

Example:

Question:

How long is the internship?

Retrieved Context:

The internship duration is 2 months.

---

## Step 4: Generate

Question + Context
↓
LLM
↓
Answer

Example:

Answer:

The internship duration is 2 months.

---

# Vector Databases

A vector database stores embeddings and performs semantic search.

Examples:

* Qdrant
* ChromaDB
* Pinecone
* Weaviate
* FAISS

Purpose:

Documents
↓
Embeddings
↓
Storage
↓
Similarity Search
↓
Retrieved Results

---

# Embedding Models

Embedding models convert text into vectors.

Example:

Text:

Internship duration is 2 months

↓

Embedding

[0.12, -0.44, 0.89, ...]

Examples:

* all-MiniLM-L6-v2
* BGE Models
* OpenAI Embeddings

---

# Similarity Search

Similarity search retrieves documents with similar meaning.

Example:

Question:

How much money do interns receive?

Document:

Interns receive a stipend.

Even though the wording differs, semantic similarity allows the correct document to be retrieved.

---

# Indexing

Indexing is the process of preparing data for retrieval.

Workflow:

Documents
↓
Chunking
↓
Embeddings
↓
Vector Database

Without indexing, retrieval is impossible.

---

# Real World Applications of RAG

## Documentation Search

Examples:

* API Documentation
* Product Manuals
* Technical Guides

---

## Enterprise Knowledge Bases

Examples:

* HR Policies
* Internal Procedures
* Employee Handbooks

---

## Customer Support

Examples:

* FAQ Assistants
* Product Support Bots

---

## Domain Experts

Examples:

* Legal Assistants
* Medical Assistants
* Financial Advisors

---

## Research Assistants

Examples:

* Literature Search
* Research Paper Retrieval

---

# Lab 1 - Naive RAG Pipeline

Objective:

Build a retrieval system without generation.

Pipeline:

PDFs
↓
Loader
↓
Chunking
↓
Embeddings
↓
Qdrant
↓
Retriever
↓
Top Matching Chunks

Components Used:

* PyPDFLoader
* RecursiveCharacterTextSplitter
* HuggingFaceEmbeddings
* QdrantVectorStore
* Retriever

Example Dataset:

* internship.pdf
* company_policy.pdf
* ai_projects.pdf
* employee_handbook.pdf
* training_program.pdf

Sample Query:

What stipend do interns receive?

Retrieved Result:

Interns receive a stipend of 15000 rupees per month.

---

# Reading

Paper:

Retrieval-Augmented Generation for Knowledge-Intensive Tasks

Authors:

Patrick Lewis et al.

Year:

2020

Main Contribution:

Introduced the modern Retrieval-Augmented Generation architecture that combines retrieval systems with language models.

---

# Key Takeaways

1. RAG combines retrieval and generation.
2. RAG keeps knowledge outside the model.
3. Fine-tuning stores knowledge inside model weights.
4. Prompt engineering relies only on prompts.
5. RAG is easier to update and maintain.
6. The RAG pipeline is:

Index → Retrieve → Augment → Generate

7. Vector databases store embeddings and enable semantic search.
8. Retrieval quality directly impacts answer quality.
9. A naive RAG system performs retrieval without generation.
10. LangChain and Qdrant can be used to implement retrieval pipelines efficiently.
