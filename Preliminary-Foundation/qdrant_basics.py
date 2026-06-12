from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="internship_docs",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

documents = [
    "Internship duration is 2 months",
    "Interns receive a stipend of 15000 rupees per month",
    "Applications close on June 30",
    "Selected interns will work on AI projects",
    "Mentors provide weekly guidance"
]

embeddings = model.encode(documents)

points = []

for i in range(len(documents)):
    point = PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "text": documents[i]
        }
    )

    points.append(point)

client.upsert(
    collection_name="internship_docs",
    points=points
)

query = input("Enter your question: ")

query_embedding = model.encode(query)

search_results = client.query_points(
    collection_name="internship_docs",
    query=query_embedding.tolist(),
    limit=3
)

print("\nTop Matches:\n")

for result in search_results.points:

    print("Score:", round(result.score, 4))
    print(result.payload["text"])
    print("-" * 50)