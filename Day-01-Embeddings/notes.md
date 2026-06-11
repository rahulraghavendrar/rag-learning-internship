# Day 1 - Embeddings and Cosine Similarity

## Required Libraries

### Google Colab

```python
!pip install sentence-transformers
!pip install scikit-learn
```

### Imports Used

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

---

# Summary of What I Learned Today

Today I learned the foundation of Retrieval-Augmented Generation (RAG).

Before a machine can perform semantic search, it must convert text into a numerical representation called an embedding.

An embedding is a vector that captures the meaning of a sentence. Similar meanings produce vectors that are located close together in a high-dimensional semantic space.

I learned:

* What embeddings are
* Why computers need embeddings
* How Sentence Transformers generate embeddings
* How the `all-MiniLM-L6-v2` model works
* How the `encode()` function generates embeddings
* How cosine similarity measures semantic similarity between embeddings

The complete flow is:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding Layer
 ↓
Transformer Layers
 ↓
Pooling
 ↓
Embedding Vector
 ↓
Cosine Similarity
 ↓
Similarity Score
```

---

# About all-MiniLM-L6-v2

The model used today was:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Breaking down the name:

```text
all
│
├── Trained on many sentence similarity datasets

MiniLM
│
├── Lightweight Transformer model

L6
│
├── Contains 6 Transformer layers

v2
│
└── Improved Version
```

This model produces embeddings with:

```text
384 dimensions
```

which means every sentence becomes:

```text
[0.12,
-0.44,
0.89,
...]
```

with 384 numerical values.

---

# What Does L6 Mean?

L6 means the model contains:

```text
6 Transformer Layers
```

Each layer refines the understanding of the sentence.

Simplified understanding:

### Layer 1

Learns basic word patterns.

Example:

```text
Dogs
are
loyal
animals
```

---

### Layer 2

Learns grammar and sentence structure.

Example:

```text
Subject
Verb
Object
```

relationships.

---

### Layer 3

Learns relationships between words.

Example:

```text
Dogs ↔ loyal
Dogs ↔ animals
```

---

### Layer 4

Builds contextual understanding.

The meaning of a word begins to depend on surrounding words.

---

### Layer 5

Learns semantic meaning.

Example:

```text
Dog ≈ Puppy
```

---

### Layer 6

Creates a refined contextual representation of the sentence.

This representation is later used to generate the final embedding.

---

# Internal Architecture

The complete architecture of the model is:

```text
Input Text

"Dogs are loyal animals"

        ↓

Tokenizer

        ↓

Token IDs

        ↓

Embedding Layer

        ↓

Transformer Layer 1

        ↓

Transformer Layer 2

        ↓

Transformer Layer 3

        ↓

Transformer Layer 4

        ↓

Transformer Layer 5

        ↓

Transformer Layer 6

        ↓

Pooling Layer

        ↓

Sentence Embedding

        ↓

384-Dimensional Vector
```

---

# The Tokenizer

Input:

```text
Dogs are loyal animals
```

becomes:

```text
["Dogs",
 "are",
 "loyal",
 "animals"]
```

Then internally:

```text
[6078,
 2024,
 8887,
 4174]
```

These are token IDs.

The neural network understands numbers, not words.

---

# The Embedding Layer

Each token ID is converted into a vector.

Example:

```text
6078

↓

[0.22,
0.81,
-0.33,
...]
```

At this stage, words have become numbers.

---

# Transformer Layers

The transformer uses self-attention.

Each word looks at other words in the sentence.

Example:

```text
Dogs
 ↘
  loyal

 ↘
  animals
```

The model learns which words influence each other.

This creates contextual meaning.

---

# Pooling Layer

After the transformer, each token has its own vector.

Example:

```text
Dogs    → Vector A
are     → Vector B
loyal   → Vector C
animals → Vector D
```

The pooling layer combines them.

Most commonly:

```text
(A+B+C+D)/4
```

This creates:

```text
One Sentence
=
One Vector
```

---

# Understanding the encode() Function

Code:

```python
embedding = model.encode(
    "Dogs are loyal animals"
)
```

The `encode()` function performs:

```text
Sentence

        ↓

Tokenization

        ↓

Token IDs

        ↓

Embedding Layer

        ↓

6 Transformer Layers

        ↓

Pooling

        ↓

Final Embedding
```

The returned object is a NumPy array.

Example:

```python
[
 0.12,
-0.44,
 0.89,
 ...
]
```

Length:

```python
len(embedding)
```

Output:

```text
384
```

---

# Cosine Similarity

Cosine similarity measures how similar two vectors are.

Instead of comparing words directly, it compares the direction of vectors.

Interpretation:

```text
1.0  = Nearly identical meaning

0.8  = Very similar

0.5  = Somewhat related

0.0  = Unrelated

-1.0 = Opposite
```

The formula is:

cos(θ) = (A · B) / (||A|| ||B||)

where:

* A = First embedding
* B = Second embedding

---

# Example

Sentence 1:

```text
Dogs are loyal animals
```

Sentence 2:

```text
Puppies are friendly pets
```

These produce similar vectors.

Result:

```text
Cosine Similarity ≈ 0.8
```

---

Sentence 1:

```text
Dogs are loyal animals
```

Sentence 2:

```text
I like pizza
```

These produce less similar vectors.

Result:

```text
Cosine Similarity ≈ 0.2
```

---

# Why Cosine Similarity Is Important

Cosine similarity is the foundation of:

* Semantic Search
* Vector Databases
* FAISS
* ChromaDB
* Pinecone
* Qdrant
* Retrieval-Augmented Generation (RAG)

When a user asks a question, both:

```text
Document
```

and

```text
Question
```

are converted into embeddings.

Cosine similarity is then used to find the most relevant document chunks.

---

# Key Takeaways

1. Embeddings are numerical representations of meaning.

2. Similar meanings produce nearby vectors.

3. The `all-MiniLM-L6-v2` model creates 384-dimensional embeddings.

4. The `encode()` function converts text into embeddings.

5. Cosine similarity measures semantic similarity between embeddings.

6. Cosine similarity is the core mechanism behind retrieval in RAG systems.

7. Understanding embeddings and cosine similarity is the first step toward understanding vector databases and Retrieval-Augmented Generation.

---

# Chunking

Before documents are stored inside a vector database, they are usually split into smaller pieces called chunks.

Chunking is the process of breaking a large document into smaller units that can be embedded and retrieved efficiently.

Example:

Large Document:

```text
Internship duration is 2 months.

Interns receive a stipend of 15000 rupees per month.

Applications close on June 30.

Selected candidates will work on AI projects.
```

After Chunking:

```text
Chunk 1

Internship duration is 2 months.

Interns receive a stipend of 15000 rupees per month.
```

```text
Chunk 2

Applications close on June 30.

Selected candidates will work on AI projects.
```

Each chunk receives its own embedding.

---

# Why Is Chunking Important?

Without chunking:

```text
Entire PDF

↓

Single Embedding
```

This causes multiple topics to be mixed together.

Example:

```text
Internship Details
Stipend
Application Process
Projects
Attendance
Deadlines
```

One embedding must represent all of these topics simultaneously.

Retrieval quality becomes poor.

With chunking:

```text
Document

↓

Chunk 1

Chunk 2

Chunk 3

↓

Individual Embeddings
```

The retriever can now find only the relevant chunk.

---

# Types of Chunking

## Fixed Size Chunking

The document is divided into fixed word counts.

Example:

```text
Chunk Size = 100 Words
```

Advantages:

* Simple
* Fast

Disadvantages:

* May split sentences
* May lose context

---

## Fixed Size Chunking With Overlap

Example:

```text
Chunk Size = 100

Overlap = 20
```

Chunks share some words.

Example:

```text
Chunk 1

Words 1-100
```

```text
Chunk 2

Words 81-180
```

This helps preserve context near chunk boundaries.

---

## Semantic Chunking

Semantic chunking groups text based on meaning rather than word count.

Instead of asking:

```text
Have I reached 100 words?
```

it asks:

```text
Has the topic changed?
```

Example:

```text
Internship duration is 2 months.

Interns receive a stipend of 15000 rupees.

Applications close on June 30.

Selected candidates will be contacted by email.

Internship benefits include mentorship.

The internship stipend is paid monthly.
```

Semantic chunking may produce:

```text
Chunk 1

Internship duration is 2 months.

Interns receive a stipend of 15000 rupees.

Internship benefits include mentorship.

The internship stipend is paid monthly.
```

```text
Chunk 2

Applications close on June 30.

Selected candidates will be contacted by email.
```

Notice that internship-related information remains together.

---

# Pairwise Semantic Chunking

To better understand semantic chunking, I implemented a pairwise similarity approach.

Workflow:

```text
Sentence

↓

Embedding

↓

Compare Against Other Sentences

↓

Cosine Similarity

↓

Group Similar Sentences
```

Example:

```text
Sentence 1

Internship duration is 2 months.
```

```text
Sentence 2

Interns receive a stipend of 15000 rupees.
```

These may have:

```text
Similarity = 0.82
```

and therefore belong to the same chunk.

---

# Pairwise Semantic Chunking Code Logic

Step 1:

Convert every sentence into an embedding.

```text
Sentence

↓

Embedding
```

---

Step 2:

Compare every sentence with all remaining sentences.

```text
Sentence 1

↓

Compare

↓

Sentence 2

Sentence 3

Sentence 4
```

---

Step 3:

Calculate cosine similarity.

```text
Similarity Score
```

Example:

```text
0.85
```

means highly related.

---

Step 4:

If similarity exceeds a threshold:

```python
threshold = 0.55
```

the sentence is added to the same chunk.

---

Step 5:

Display all discovered chunks.

---

# Limitation of Pairwise Semantic Chunking

Suppose:

```text
Sentence 1

Internship duration
```

```text
Sentence 2

Internship stipend
```

```text
Sentence 3

Application deadline
```

```text
Sentence 4

Internship benefits
```

Sentence 1 and Sentence 4 may be highly related even though Sentence 3 appears between them.

Basic pairwise approaches may not always capture these long-range relationships.

More advanced systems use:

* Centroid-Based Chunking
* Graph-Based Chunking
* Hierarchical Clustering
* Recursive Chunking

to preserve topic coherence.

---

# How Chunking Fits Into RAG

Complete RAG Pipeline:

```text
PDF

↓

Chunking

↓

Chunk 1
Chunk 2
Chunk 3

↓

Embeddings

↓

Vector Database

↓

User Query

↓

Query Embedding

↓

Similarity Search

↓

Retrieve Relevant Chunks

↓

LLM

↓

Final Answer
```

Chunking is one of the most important preprocessing steps in Retrieval-Augmented Generation because retrieval quality depends heavily on chunk quality.
# Day 1 - Fixed Size Chunking

## Summary

Today I learned about Fixed Size Chunking, which is one of the most fundamental preprocessing techniques used in Retrieval-Augmented Generation (RAG).

Before generating embeddings, large documents are usually divided into smaller pieces called chunks.

These chunks are then converted into embeddings and stored inside a vector database.

Chunking improves retrieval quality because embeddings represent smaller, more focused pieces of information.

---

# What Is Chunking?

Chunking is the process of dividing a large document into smaller sections.

Example:

Large Document:

```text
Internship duration is 2 months.

Interns receive a stipend of 15000 rupees per month.

Applications close on June 30.

Selected candidates will work on AI projects.
```

After Chunking:

```text
Chunk 1

Internship duration is 2 months.

Interns receive a stipend of 15000 rupees per month.
```

```text
Chunk 2

Applications close on June 30.

Selected candidates will work on AI projects.
```

Each chunk receives its own embedding.

---

# Why Do We Need Chunking?

Suppose we have a PDF containing:

```text
Internship Details

Stipend Information

Application Process

Project Information

Attendance Rules

Deadlines
```

If the entire document is converted into a single embedding:

```text
PDF

↓

One Embedding
```

multiple topics become mixed together.

This reduces retrieval accuracy.

Instead:

```text
PDF

↓

Chunk 1

Chunk 2

Chunk 3

↓

Separate Embeddings
```

Each embedding represents a specific topic.

This improves retrieval quality.

---

# What Is Fixed Size Chunking?

Fixed Size Chunking divides text based on a predefined number of words.

Example:

```python
chunk_size = 20
```

means:

```text
20 Words Per Chunk
```

Regardless of sentence boundaries.

Example:

```text
Word 1
Word 2
...
Word 20
```

becomes:

```text
Chunk 1
```

The next 20 words become:

```text
Chunk 2
```

and so on.

---

# Workflow

The complete workflow is:

```text
PDF

↓

Extract Text

↓

Split Into Words

↓

Create Chunks

↓

Store Chunks

↓

Generate Embeddings
```

---

# Code Workflow

The code follows these steps:

### Step 1

Load the PDF.

```python
reader = PdfReader(
    "internship_handbook.pdf"
)
```

---

### Step 2

Extract text from every page.

```python
for page in reader.pages:

    text += page.extract_text()
```

Result:

```text
Entire PDF Text
```

stored inside:

```python
text
```

---

### Step 3

Split the text into words.

```python
words = text.split()
```

Example:

Before:

```text
Internship duration is 2 months
```

After:

```python
[
 "Internship",
 "duration",
 "is",
 "2",
 "months"
]
```

---

### Step 4

Define chunk size.

```python
chunk_size = 20
```

Meaning:

```text
Every Chunk

↓

20 Words
```

---

### Step 5

Create chunks.

```python
for i in range(
    0,
    len(words),
    chunk_size
):
```

The loop moves through the document in steps of 20 words.

Example:

```text
0

20

40

60
```

Each value represents the starting position of a chunk.

---

### Step 6

Slice the words.

```python
chunk_words = words[
    i:i+chunk_size
]
```

Example:

```python
words[20:40]
```

returns:

```text
Words 20–39
```

---

### Step 7

Convert words back into text.

```python
chunk_text = " ".join(
    chunk_words
)
```

Before:

```python
[
 "Internship",
 "duration",
 "is",
 "2"
]
```

After:

```text
Internship duration is 2
```

---

### Step 8

Store the chunk.

```python
chunks.append(
    chunk_text
)
```

The chunk is added to the chunk list.

---

### Step 9

Display all chunks.

```python
for index, chunk in enumerate(
    chunks
):
```

Example output:

```text
Chunk 1

Internship duration is 2 months...
```

```text
Chunk 2

Applications close on June 30...
```

---

# Understanding Key Python Functions

## split()

Converts text into a list of words.

Example:

```python
text.split()
```

Output:

```python
[
 "Internship",
 "duration",
 "is",
 "2",
 "months"
]
```

---

## join()

Converts a list of words back into text.

Example:

```python
" ".join(words)
```

Output:

```text
Internship duration is 2 months
```

---

## range()

Generates a sequence of numbers.

Example:

```python
range(
    0,
    80,
    20
)
```

Produces:

```text
0

20

40

60
```

---

## append()

Adds an item to a list.

Example:

```python
chunks.append(
    chunk_text
)
```

---

## enumerate()

Adds numbering while looping.

Example:

```python
for index, chunk in enumerate(
    chunks
)
```

Produces:

```text
Chunk 1

Chunk 2

Chunk 3
```

---

# Advantages of Fixed Size Chunking

1. Simple to implement.

2. Very fast.

3. Works well for small projects.

4. Easy to understand.

---

# Limitations of Fixed Size Chunking

Fixed size chunking ignores sentence boundaries.

Example:

```text
Chunk 1

The stipend is credited
```

```text
Chunk 2

on the first working day...
```

The sentence becomes split across chunks.

This can reduce retrieval quality.

---

# Why Better Chunking Methods Exist

Because fixed size chunking can split related information, more advanced techniques were developed:

* Fixed Size Chunking with Overlap
* Sentence-Based Chunking
* Semantic Chunking
* Recursive Chunking
* Document-Aware Chunking

These methods preserve context more effectively.

---

# How Fixed Size Chunking Fits Into RAG

A typical RAG pipeline looks like:

```text
PDF

↓

Chunking

↓

Chunk 1
Chunk 2
Chunk 3

↓

Embeddings

↓

Vector Database

↓

User Query

↓

Retrieve Relevant Chunks

↓

LLM

↓

Final Answer
```

Fixed Size Chunking is usually the first chunking strategy learned because it introduces the concept of dividing large documents before embedding and retrieval.

---

# Key Takeaways

1. Chunking divides large documents into smaller pieces.

2. Fixed Size Chunking uses a predefined word count.

3. Chunking improves retrieval quality.

4. Fixed Size Chunking is simple and fast.

5. Fixed Size Chunking may split sentences and lose context.

6. More advanced chunking strategies solve these limitations.

7. Chunking is one of the first preprocessing steps in every RAG pipeline.

8. Understanding chunking is essential before learning vector databases and advanced retrieval techniques.
