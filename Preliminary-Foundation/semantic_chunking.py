from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "Internship duration is 2 months.",
    "Interns receive a stipend of 15000 rupees.",
    "Applications close on June 30.",
    "Selected candidates will be contacted by email.",
    "Internship benefits include mentorship.",
    "The internship stipend is paid monthly."
]
embeddings = model.encode(sentences)
threshold = 0.55
visited = [False] * len(sentences)
chunks = []
for i in range(len(sentences)):
    if visited[i]:
        continue
    current_chunk = [sentences[i]]
    visited[i] = True
    for j in range(i + 1, len(sentences)):
        similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        print(f"S{i+1} vs S{j+1} = {similarity:.3f}")
        if similarity >= threshold:
            current_chunk.append(sentences[j])
            visited[j] = True
    chunks.append(current_chunk)
print("\nFINAL CHUNKS\n")
for index, chunk in enumerate(chunks):
    print(f"Chunk {index + 1}")
    for sentence in chunk:
        print(sentence)
    print("\n" + "-" * 50)