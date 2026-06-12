from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#example sentences
sentence="Dogs are loyal animals"
embedding=model.encode(sentence)
print("Sentence:", sentence)
print("\nEmbedding:", embedding)
print("\nEmbedding dimension:", len(embedding))
