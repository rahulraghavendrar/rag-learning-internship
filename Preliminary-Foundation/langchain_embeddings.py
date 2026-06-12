from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Internship duration is 2 months"

embedding = embedding_model.embed_query(text)

print("Embedding Length:", len(embedding))

print()

print(embedding[:10])