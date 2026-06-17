from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer(
    "all-MiniLM-L6-v2"
)

question="""
What is RAG?
"""

context="""
RAG combines retrieval and generation.
"""

answer="""
RAG combines retrieval and generation.
"""

faithfulness=cosine_similarity(
    [model.encode(context)],
    [model.encode(answer)]
)[0][0]

relevance=cosine_similarity(
    [model.encode(question)],
    [model.encode(answer)]
)[0][0]

answer2="""
RAG retrieves information and generates responses.
"""

consistency=cosine_similarity(
    [model.encode(answer)],
    [model.encode(answer2)]
)[0][0]

print(
    f"Faithfulness: {faithfulness:.4f}"
)

print(
    f"Relevance: {relevance:.4f}"
)

print(
    f"Consistency: {consistency:.4f}"
)