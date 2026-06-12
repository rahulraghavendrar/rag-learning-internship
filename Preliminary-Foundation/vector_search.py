from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Red car",
    "Interns receive a stipend of 15000 rupees per month",
    "Applications close on June 30"
]

query = "How much money do interns receive?"

document_embeddings = model.encode(documents)
query_embedding = model.encode(query)

scores = []

for i in range(len(documents)):
    similarity = cosine_similarity(
        [query_embedding],
        [document_embeddings[i]]
    )[0][0]

    scores.append((documents[i], similarity))

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

print("All Scores:")
for doc, score in scores:
    print(f"{doc} -> {score}")

print("\nBest Match:")
print(scores[0][0])