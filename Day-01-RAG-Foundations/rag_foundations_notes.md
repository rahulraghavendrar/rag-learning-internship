# Day 01 - RAG Foundations & Architecture

## Goal

Understand the Retrieval-Augmented Generation (RAG) pipeline from end to end and learn why modern AI applications use RAG instead of relying solely on fine-tuning or prompt engineering.

---

# What is RAG?

RAG stands for:

Retrieval-Augmented Generation

It is a technique where an LLM is provided with relevant external information before generating an answer.

Traditional LLM:

Question
↓
LLM
↓
Answer

RAG:

Question
↓
Retrieve Relevant Information
↓
Provide Context To LLM
↓
Generate Answer

---

# Why Not Just Fine-Tune?

Fine-tuning stores knowledge inside the model weights.

Example:

Company Policy
↓
Fine-Tune Model
↓
Knowledge Stored In Weights

Problem:

If the policy changes:

* Retraining is required
* Expensive
* Time consuming
* Difficult to maintain

---

# Why RAG Is Preferred

RAG keeps knowledge outside the model.

Example:

Company Policy PDF
↓
Vector Database
↓
Retrieve Relevant Information
↓
LLM

If information changes:

* Update the documents
* Re-index the vector database
* No retraining required

This makes RAG more flexible and scalable.

---

# RAG vs Fine-Tuning vs Prompt Engineering

## Prompt Engineering

Approach:

Question + Large Prompt

Advantages:

* Fastest to implement
* No training required

Disadvantages:

* Context window limitations
* Difficult to scale for large knowledge bases

---

## Fine-Tuning

Approach:

Knowledge stored inside model weights

Advantages:

* Model learns specific behavior
* Useful for style and task adaptation

Disadvantages:

* Expensive
* Requires retraining for updates
* Knowledge becomes outdated

---

## RAG

Approach:

Knowledge stored externally and retrieved when needed

Advantages:

* Easy updates
* Lower cost
* More scalable
* Reduces hallucinations

Disadvantages:

* Requires retrieval infrastructure
* Retrieval quality affects final answer

---

# Core RAG Pipeline

## 1. Index

Prepare documents for retrieval.

Documents
↓
Chunking
↓
Embeddings
↓
Vector Database

---

## 2. Retrieve

User asks a question.

Question
↓
Embedding
↓
Similarity Search
↓
Relevant Chunks

---

## 3. Augment

Combine:

Question
+
Retrieved Chunks

into a prompt.

Example:

Question:
How long is the internship?

Context:
The internship duration is 2 months.

---

## 4. Generate

The LLM generates an answer using the retrieved context.

Question + Context
↓
LLM
↓
Answer

---

# Real World Use Cases

## Documentation Search

Examples:

* Company documentation
* API documentation
* Product manuals

---

## Enterprise Knowledge Base

Examples:

* HR policies
* Internal procedures
* Employee handbooks

---

## Customer Support

Examples:

* FAQ systems
* Product support assistants

---

## Domain Experts

Examples:

* Legal assistants
* Medical knowledge assistants
* Financial advisory systems

---

## Research Assistants

Examples:

* Research paper search
* Literature review systems

---

# Vector Databases

A vector database stores embeddings and performs semantic search.

Examples:

* Qdrant
* ChromaDB
* Pinecone
* Weaviate
* FAISS

---

# Embedding Models

Embedding models convert text into vectors.

Example:

Sentence:

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

Similarity search finds documents with similar meaning.

Question:

How much money do interns receive?

Document:

Interns receive a stipend.

Even though the words are different, embeddings capture the semantic meaning and similarity search retrieves the correct document.

---

# Indexing

Indexing is the process of preparing documents for retrieval.

Pipeline:

Documents
↓
Chunking
↓
Embeddings
↓
Vector Database

Without indexing:

No retrieval is possible.

---

# Key Takeaways

1. RAG combines retrieval and generation.

2. RAG keeps knowledge outside the model.

3. Fine-tuning stores knowledge inside model weights.

4. Prompt engineering relies only on prompts.

5. RAG is easier to update and maintain.

6. The core pipeline is:

Index → Retrieve → Augment → Generate

7. Vector databases store embeddings and enable semantic search.

8. Retrieval quality is one of the most important factors in RAG performance.
