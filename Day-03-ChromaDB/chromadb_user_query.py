import chromadb

# Create ChromaDB client
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="internship_docs"
)

# Add documents
collection.add(
    documents=[
        "Internship duration is 2 months",
        "Interns receive a stipend of 15000 rupees per month",
        "Applications close on June 30"
    ],
    ids=[
        "doc1",
        "doc2",
        "doc3"
    ]
)

# Take query from user
query = input("Enter your question: ")

# Search collection
results = collection.query(
    query_texts=[query],
    n_results=1
)

print("\nBest Match:")
print(results["documents"][0][0])