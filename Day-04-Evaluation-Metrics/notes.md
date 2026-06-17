# Day 04 - RAG Evaluation & Metrics

## Goal

Learn how to rigorously measure the quality of a RAG system.

Until now we have built:

```text
PDF
↓
Chunking
↓
Embeddings
↓
Qdrant
↓
BM25
↓
Hybrid Retrieval
↓
Query Expansion
↓
Cross Encoder Reranking
↓
Final Retrieved Chunks
```

The question now becomes:

```text
How do we know if the RAG system is actually good?
```

This is the purpose of Day 4.

---

# Evaluation Categories

Day 4 contains three major categories:

## Retrieval Metrics

Measure retrieval quality.

Examples:

* Recall@K
* Precision@K
* MRR
* NDCG

---

## Generation Metrics

Measure answer quality.

Examples:

* BLEU
* ROUGE
* BERTScore
* Semantic Similarity

---

## RAG-Specific Metrics

Measure overall RAG performance.

Examples:

* Faithfulness
* Relevance
* Consistency

---

# Why Evaluation Matters

Suppose:

Question:

```text
What is Hybrid Retrieval?
```

Retriever returns:

```text
Chunk A
Chunk B
Chunk C
```

LLM generates:

```text
Hybrid Retrieval combines
BM25 and Dense Retrieval.
```

How do we know:

* Retrieval was correct?
* Ranking was correct?
* Generated answer was correct?

Evaluation metrics answer these questions.

---

# Retrieval Metrics

Retrieval metrics evaluate the quality of document retrieval before generation occurs.

Workflow:

```text
Question
↓
Retriever
↓
Top K Chunks
↓
Evaluation Metrics
```

---

# Recall@K

## What is Recall?

Recall measures:

```text
Did the retriever find the relevant chunks?
```

---

## Recall Formula

Recall@K =

Retrieved Relevant Chunks

/

Total Relevant Chunks

---

## Example

Relevant Chunks:

```text
Chunk 3
Chunk 8
Chunk 10
```

Retriever returns:

```text
Chunk 1
Chunk 3
Chunk 5
Chunk 8
Chunk 20
```

Retrieved Relevant Chunks:

```text
Chunk 3
Chunk 8
```

Recall:

```text
2 / 3

66.67%
```

---

## Interpretation

High Recall:

```text
Most relevant chunks are retrieved.
```

Low Recall:

```text
Important chunks are being missed.
```

---

## What Recall Measures

Recall answers:

```text
Can the retriever find the correct chunks?
```

---

## What Recall Does NOT Measure

Recall does not care about ranking position.

Example:

Relevant Chunk at:

```text
Rank 1
```

and

```text
Rank 10
```

Both count as retrieved.

---

# Precision@K

## What is Precision?

Precision measures:

```text
How many retrieved chunks are actually useful?
```

---

## Precision Formula

Precision@K =

Retrieved Relevant Chunks

/

Retrieved Chunks

---

## Example

Retrieved:

```text
Chunk 1
Chunk 3
Chunk 5
Chunk 8
Chunk 20
```

Relevant:

```text
Chunk 3
Chunk 8
```

Precision:

```text
2 / 5

40%
```

---

## Interpretation

High Precision:

```text
Most retrieved chunks are useful.
```

Low Precision:

```text
Retriever returns lots of noise.
```

---

# Recall vs Precision

Recall:

```text
Coverage
```

Precision:

```text
Purity
```

---

## Example

Total Relevant Chunks:

```text
10
```

Retrieved Chunks:

```text
20
```

Relevant Retrieved:

```text
8
```

Recall:

```text
8 / 10

80%
```

Precision:

```text
8 / 20

40%
```

---

# Why Cross Encoder Improves Precision

Before Reranking:

```text
Relevant
Relevant
Noise
Noise
Noise
```

After Reranking:

```text
Relevant
Relevant
Relevant
Noise
Noise
```

Precision increases.

---

# MRR (Mean Reciprocal Rank)

## What is MRR?

MRR measures:

```text
How early was the first relevant chunk retrieved?
```

---

## MRR Formula

MRR =

1

/

Rank Of First Relevant Chunk

---

## Example

Results:

```text
Rank 1 → Chunk A

Rank 2 → Chunk B

Rank 3 → BM25 Chunk

Rank 4 → Chunk D
```

First Relevant Chunk:

```text
Rank 3
```

MRR:

```text
1 / 3

0.333
```

---

## Perfect MRR

Relevant Chunk:

```text
Rank 1
```

MRR:

```text
1.0
```

Perfect score.

---

## Why MRR Matters

Recall cannot distinguish:

```text
Rank 1

vs

Rank 10
```

Both count as retrieved.

MRR rewards:

```text
Earlier Retrieval
```

---

## Why MRR Is Important

Cross Encoder Reranking should improve:

```text
MRR
```

because relevant chunks move toward the top.

---

# NDCG

## What is NDCG?

NDCG stands for:

```text
Normalized Discounted Cumulative Gain
```

It evaluates:

```text
Overall Ranking Quality
```

rather than only the first relevant result.

---

## Example

Results:

```text
Rank 1 → Highly Relevant

Rank 2 → Relevant

Rank 3 → Slightly Relevant

Rank 4 → Irrelevant
```

NDCG rewards:

```text
Relevant chunks appearing early.
```

---

# Why NDCG Exists

Suppose:

System A

```text
Relevant
Relevant
Relevant
Noise
Noise
```

System B

```text
Noise
Noise
Relevant
Relevant
Relevant
```

Recall:

```text
Same
```

MRR:

```text
May be Similar
```

NDCG:

```text
System A Wins
```

because overall ranking quality is better.

---

## What NDCG Measures

NDCG evaluates:

* Ranking Quality
* Relevance Ordering
* Position Of Relevant Chunks

---

## Interpretation

High NDCG:

```text
Relevant chunks appear near the top.
```

Low NDCG:

```text
Relevant chunks appear lower in the ranking.
```

---

# Libraries Used

## sklearn.metrics

```python
from sklearn.metrics import ndcg_score
```

Purpose:

```text
Calculate NDCG
```

---

# Functions Used

## retriever.invoke()

```python
results = retriever.invoke(
    query
)
```

Purpose:

Retrieve top K chunks from Qdrant.

---

## ndcg_score()

```python
ndcg_score(
    [y_true],
    [y_score]
)
```

Purpose:

Calculate ranking quality.

---

# evaluation_metrics.py

This file combines:

* Recall@K
* Precision@K
* MRR
* NDCG

into a single evaluation pipeline.

Workflow:

```text
Question
↓
Retriever
↓
Top K Chunks
↓
Recall
↓
Precision
↓
MRR
↓
NDCG
```

---

# Key Takeaways

1. Retrieval metrics evaluate retrieval quality.

2. Recall measures retrieval coverage.

3. Precision measures retrieval purity.

4. MRR measures how early the first relevant result appears.

5. NDCG measures overall ranking quality.

6. Cross Encoder Reranking should improve Precision, MRR and NDCG.

7. Retrieval metrics are evaluated before generation.

---

# Day 04 Progress

Completed:

✅ Evaluation Overview

✅ Retrieval Metrics Overview

✅ Recall@K

✅ Precision@K

✅ MRR

✅ NDCG

✅ evaluation_metrics.py

# Generation Metrics

Until now we evaluated:

```text
Question
↓
Retriever
↓
Retrieved Chunks
```

using:

* Recall@K
* Precision@K
* MRR
* NDCG

However, RAG systems do not stop at retrieval.

A complete RAG pipeline is:

```text
Question
↓
Retriever
↓
Retrieved Chunks
↓
LLM
↓
Generated Answer
```

Now we must evaluate:

```text
How good is the generated answer?
```

This is where Generation Metrics are used.

---

# Why Generation Metrics Matter

Question:

```text
What is BM25?
```

Reference Answer:

```text
BM25 is a ranking algorithm used in information retrieval.
```

Generated Answer:

```text
BM25 is a retrieval ranking algorithm.
```

The generated answer is correct.

However:

```text
How do we measure this automatically?
```

Generation metrics answer this question.

---

# Generation Metrics Covered

1. BLEU

2. ROUGE

3. BERTScore

4. Semantic Similarity

---

# BLEU

BLEU stands for:

```text
Bilingual Evaluation Understudy
```

Originally designed for:

```text
Machine Translation
```

but later adopted for text generation.

---

# What BLEU Measures

BLEU measures:

```text
Word Overlap
```

between:

```text
Reference Answer

and

Generated Answer
```

---

# Example

Reference:

```text
BM25 is a ranking algorithm
```

Generated:

```text
BM25 is a ranking algorithm
```

BLEU:

```text
1.0
```

Perfect match.

---

Generated:

```text
BM25 is used for retrieval
```

BLEU:

```text
Lower
```

because fewer words overlap.

---

# Advantages Of BLEU

* Simple
* Fast
* Easy To Calculate

---

# Limitations Of BLEU

Reference:

```text
BM25 is a ranking algorithm.
```

Generated:

```text
BM25 is an information retrieval ranking method.
```

Meaning:

```text
Correct
```

BLEU:

```text
May Still Be Low
```

because wording changed.

---

# Key Point

BLEU evaluates:

```text
Word Matching
```

not:

```text
Meaning
```

---

# ROUGE

ROUGE stands for:

```text
Recall-Oriented Understudy for Gisting Evaluation
```

Usually used for:

```text
Summarization
```

---

# What ROUGE Measures

ROUGE measures:

```text
How much of the reference answer
appears in the generated answer
```

---

# Example

Reference:

```text
BM25 is a ranking algorithm used in retrieval systems.
```

Generated:

```text
BM25 is a ranking algorithm.
```

ROUGE:

```text
High
```

because important information is retained.

---

# BLEU vs ROUGE

BLEU:

```text
Precision-Oriented
```

ROUGE:

```text
Recall-Oriented
```

---

# Easy Memory Trick

BLEU:

```text
Generated Answer
↓
How Much Matches Reference
```

ROUGE:

```text
Reference Answer
↓
How Much Appears In Generated
```

---

# Why BLEU And ROUGE Are Not Enough

Modern LLMs often generate:

```text
Different Words

Same Meaning
```

Example:

Reference:

```text
BM25 is a ranking algorithm.
```

Generated:

```text
BM25 ranks documents using relevance scores.
```

Meaning:

```text
Same
```

BLEU:

```text
Low
```

ROUGE:

```text
Low
```

This motivated newer metrics.

---

# BERTScore

BERTScore evaluates:

```text
Meaning Similarity
```

instead of exact word overlap.

---

# How BERTScore Works

Workflow:

```text
Reference Answer
↓
Embeddings

Generated Answer
↓
Embeddings

↓

Similarity Calculation
```

This is very similar to:

```text
Embeddings
↓
Similarity Search
↓
Qdrant Retrieval
```

which we already learned.

---

# Example

Reference:

```text
BM25 is a ranking algorithm.
```

Generated:

```text
BM25 ranks documents based on relevance.
```

Words differ.

Meaning is similar.

BERTScore:

```text
High
```

---

# Advantages Of BERTScore

* Measures semantic meaning
* Better suited for LLM outputs
* Handles paraphrasing well

---

# Semantic Similarity

Semantic Similarity is the most intuitive metric.

It asks:

```text
Do these two sentences mean the same thing?
```

---

# Workflow

Reference Answer
↓
Embedding

Generated Answer
↓
Embedding

↓

Cosine Similarity

---

# Example

Reference:

```text
RAG combines retrieval and generation.
```

Generated:

```text
RAG retrieves information and generates answers.
```

Meaning:

```text
Very Similar
```

Semantic Similarity:

```text
High
```

---

# Why Semantic Similarity Is Familiar

This is exactly the same concept used in:

* Embeddings
* Vector Databases
* Dense Retrieval
* Hybrid Retrieval
* Qdrant Similarity Search

The only difference is:

Instead of comparing:

```text
Query

vs

Document
```

we compare:

```text
Reference Answer

vs

Generated Answer
```

---

# Libraries Used

## rouge_score

```python
from rouge_score import rouge_scorer
```

Purpose:

Calculate ROUGE metrics.

---

## bert_score

```python
from bert_score import score
```

Purpose:

Calculate BERTScore.

---

## sentence_transformers

```python
from sentence_transformers import SentenceTransformer
```

Purpose:

Generate embeddings.

---

## cosine_similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
```

Purpose:

Calculate semantic similarity.

---

# Functions Used

## rouge_scorer.RougeScorer()

Creates a ROUGE evaluator.

---

## score()

Calculates:

```text
Precision
Recall
F1
```

for BERTScore.

---

## encode()

Converts text into embeddings.

Used throughout previous RAG modules.

---

## cosine_similarity()

Measures similarity between embeddings.

---

# Generation Metrics Summary

BLEU:

```text
Word Overlap
```

ROUGE:

```text
Reference Coverage
```

BERTScore:

```text
Semantic Similarity Using Transformers
```

Semantic Similarity:

```text
Embedding Similarity
```

---

# RAG Metrics

Traditional metrics compare:

```text
Reference Answer

vs

Generated Answer
```

Modern RAG systems need additional evaluation.

Why?

Because RAG introduces:

```text
Question
↓
Retriever
↓
Context
↓
LLM
↓
Answer
```

Now we must evaluate:

* Was the answer grounded in the retrieved context?
* Did it answer the question?
* Is it stable?

These are called RAG Metrics.

---

# RAG Metrics Covered

1. Faithfulness

2. Relevance

3. Consistency

---

# Faithfulness

Faithfulness measures:

```text
Did the answer come from the retrieved context?
```

---

# Example

Retrieved Context:

```text
Internship duration is 2 months.
```

Answer:

```text
Internship duration is 2 months.
```

Faithfulness:

```text
High
```

---

Answer:

```text
Internship duration is 6 months.
```

Faithfulness:

```text
Low
```

because the answer was not supported by context.

---

# Why Faithfulness Matters

Faithfulness is the primary defense against:

```text
Hallucinations
```

---

# Relevance

Relevance measures:

```text
Did the answer actually answer the question?
```

---

# Example

Question:

```text
What is BM25?
```

Answer:

```text
BM25 is a retrieval ranking algorithm.
```

Relevance:

```text
High
```

---

Answer:

```text
Qdrant stores embeddings.
```

Relevance:

```text
Low
```

---

# Consistency

Consistency measures:

```text
Does the system give similar answers repeatedly?
```

---

# Example

Run the same question multiple times.

Answer 1:

```text
RAG combines retrieval and generation.
```

Answer 2:

```text
RAG retrieves information and generates answers.
```

Consistency:

```text
High
```

because meaning remains stable.

---

# Why Consistency Matters

Low consistency may indicate:

* Prompt instability
* Retrieval instability
* Model instability

---

# RAG Metrics Workflow

Question
↓
Retrieved Context
↓
Generated Answer
↓
Faithfulness
↓
Relevance
↓
Consistency

---

# Libraries Used

## SentenceTransformer

Used to generate embeddings.

---

## cosine_similarity

Used to compare:

* Context vs Answer
* Question vs Answer
* Answer vs Answer

---

# RAG Metrics Summary

Faithfulness:

```text
Answer vs Context
```

Relevance:

```text
Answer vs Question
```

Consistency:

```text
Answer vs Answer
```

---

# Day 04 Progress

Completed:

✅ Evaluation Overview

✅ Recall@K

✅ Precision@K

✅ MRR

✅ NDCG

✅ evaluation_metrics.py

✅ BLEU

✅ ROUGE

✅ BERTScore

✅ Semantic Similarity

✅ generation_metrics.py

✅ Faithfulness

✅ Relevance

✅ Consistency

✅ rag_metrics.py

# Lab 5 - RAG Evaluation Pipeline

## Goal

Build a complete RAG system and evaluate it using an evaluation framework.

This lab combines concepts from:

```text
Preliminary Day
---------------
Embeddings
Vector Databases
Qdrant

Day 01
------
PyPDFLoader
RAG Pipeline
Retriever

Day 02
------
Chunking
Recursive Chunking

Day 03
------
Retrieval
Hybrid Search
Cross Encoder Reranking

Day 04
------
Evaluation Metrics
RAG Evaluation
```

---

# Why Do We Need Lab 5?

Until now we evaluated individual components.

Examples:

```text
Recall
Precision
MRR
NDCG
```

evaluate retrieval.

---

```text
BLEU
ROUGE
BERTScore
Semantic Similarity
```

evaluate generated answers.

---

```text
Faithfulness
Relevance
Consistency
```

evaluate RAG behavior.

---

However, in a real project we need:

```text
One Complete Evaluation Pipeline
```

that evaluates the entire RAG system.

---

# What Is A RAG Evaluation Pipeline?

A RAG Evaluation Pipeline measures the quality of:

```text
Question
↓
Retriever
↓
Retrieved Context
↓
LLM
↓
Generated Answer
```

instead of evaluating each component separately.

---

# Evaluation Framework Chosen

For Lab 5 we use:

```text
RAGAS
```

---

## Why RAGAS?

Compared to other frameworks:

### Langfuse

Purpose:

```text
Production Monitoring
```

Tracks:

* Requests
* Latency
* Cost
* Traces

Best suited for deployed applications.

---

### LlamaIndex Evaluation

Purpose:

```text
Evaluation Inside LlamaIndex
```

Useful when using the LlamaIndex ecosystem.

---

### RAGAS

Purpose:

```text
RAG Evaluation
```

Specifically designed for:

* Faithfulness
* Relevance
* Context Quality

Most beginner-friendly framework for learning RAG evaluation.

---

# Lab 5 Architecture

Workflow:

```text
1-Week RAG Deep Dive Learning Plan.pdf
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
Question
↓
Retrieved Context
↓
Generated Answer
↓
RAGAS Evaluation
```

---

# Step 1 - Load PDF

We use:

```python
PyPDFLoader
```

Purpose:

```text
PDF
↓
LangChain Documents
```

This is the same loader used since Day 1.

---

# Step 2 - Chunk Documents

We use:

```python
RecursiveCharacterTextSplitter
```

Purpose:

```text
Large Document
↓
Smaller Chunks
```

This is the chunking strategy we studied in Day 2.

---

# Step 3 - Create Embeddings

Model:

```python
all-MiniLM-L6-v2
```

Purpose:

```text
Text
↓
Vector Embeddings
```

This is the same embedding model used throughout the internship.

---

# Step 4 - Store In Qdrant

We use:

```python
QdrantVectorStore
```

Purpose:

```text
Embeddings
↓
Vector Database
```

This allows semantic retrieval.

---

# Step 5 - Retrieve Context

We use:

```python
retriever.invoke()
```

Purpose:

```text
Question
↓
Top Matching Chunks
```

This is the same retrieval workflow used in:

* Day 1
* Day 3

---

# Step 6 - Create Evaluation Dataset

RAGAS expects:

```text
Question
Answer
Contexts
Ground Truth
```

---

## Question

The user query.

Example:

```text
What is the goal of Day 3?
```

---

## Contexts

Retrieved chunks.

Example:

```text
Improve retrieval quality beyond simple similarity.
```

---

## Generated Answer

Answer produced by the RAG system.

Example:

```text
Improve retrieval quality beyond simple similarity.
```

---

## Ground Truth

Correct answer.

Example:

```text
Improve retrieval quality beyond simple similarity.
```

---

# Step 7 - Run RAGAS

Function:

```python
evaluate()
```

Purpose:

```text
Dataset
↓
Metrics
↓
Scores
```

RAGAS automatically calculates multiple evaluation metrics.

---

# Metrics Used In Lab 5

## Faithfulness

Measures:

```text
Answer
vs
Retrieved Context
```

Question:

```text
Did the answer come from the retrieved context?
```

---

## Answer Relevancy

Measures:

```text
Question
vs
Answer
```

Question:

```text
Did the answer actually answer the question?
```

---

## Context Precision

Measures:

```text
Retrieved Context
```

Question:

```text
How much retrieved context was useful?
```

Similar to:

```text
Precision@K
```

---

## Context Recall

Measures:

```text
Retrieved Context
```

Question:

```text
Did retrieval miss important information?
```

Similar to:

```text
Recall@K
```

---

# Relationship To Previous Metrics

## Retrieval Metrics

```text
Recall@K
Precision@K
MRR
NDCG
```

Evaluate retrieval quality.

---

## Generation Metrics

```text
BLEU
ROUGE
BERTScore
Semantic Similarity
```

Evaluate answer quality.

---

## RAG Metrics

```text
Faithfulness
Relevance
Consistency
```

Evaluate RAG behavior.

---

## RAGAS

Combines these ideas into one framework.

---

# New Libraries Used

## datasets

```python
from datasets import Dataset
```

Purpose:

Create evaluation datasets.

---

## ragas

```python
from ragas import evaluate
```

Purpose:

Run RAG evaluations.

---

## ragas.metrics

```python
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
```

Purpose:

Import evaluation metrics.

---

# Functions Used

## Dataset.from_dict()

Creates a dataset from Python dictionaries.

Example:

```python
dataset = Dataset.from_dict(data)
```

---

## evaluate()

Runs RAGAS evaluation.

Example:

```python
result = evaluate(
    dataset,
    metrics=[...]
)
```

Output:

```text
Metric Scores
```

---

# Bottleneck Analysis

One of the most important uses of evaluation.

Suppose:

```text
Faithfulness = 0.95
Answer Relevancy = 0.92
Context Recall = 0.35
```

Interpretation:

```text
Retriever Is Weak
```

because context recall is low.

---

Another Example:

```text
Context Recall = 0.95
Faithfulness = 0.40
```

Interpretation:

```text
LLM Is Hallucinating
```

because retrieval is good but answers are not grounded.

---

# Why Lab 5 Is Important

This is the first lab that evaluates the entire RAG pipeline.

Instead of asking:

```text
Is retrieval good?
```

or

```text
Is generation good?
```

we ask:

```text
Is the complete RAG system good?
```

---

# Key Takeaways

1. Lab 5 combines everything learned from Days 1–4.

2. RAGAS is a framework designed specifically for RAG evaluation.

3. Evaluation requires:

   * Question
   * Context
   * Answer
   * Ground Truth

4. RAGAS automatically computes:

   * Faithfulness
   * Answer Relevancy
   * Context Precision
   * Context Recall

5. Evaluation helps identify bottlenecks in retrieval and generation.

6. Lab 5 is the capstone project for Day 04.

---

# Day 04 Progress

Completed:

✅ Retrieval Metrics

* Recall@K
* Precision@K
* MRR
* NDCG

✅ Generation Metrics

* BLEU
* ROUGE
* BERTScore
* Semantic Similarity

✅ RAG Metrics

* Faithfulness
* Relevance
* Consistency

✅ Lab 5

* Complete RAG Evaluation Pipeline
* RAGAS Framework
* Bottleneck Analysis


