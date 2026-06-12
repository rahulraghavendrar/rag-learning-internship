from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

sentence1 = "Dogs are loyal animals"
sentence2 = "Puppies are friendly pets"
sentence3 = "I like pizza"

# Convert sentences into embeddings
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)

# Similarity calculations
similarity1 = cosine_similarity(
    [embedding1],
    [embedding2]
)

similarity2 = cosine_similarity(
    [embedding1],
    [embedding3]
)

print(
    "Dogs vs Puppies:",
    similarity1[0][0]
)

print(
    "Dogs vs Pizza:",
    similarity2[0][0]
)