import chromadb

client = chromadb.Client()

collection = client.create_collection(
    name="internship_docs"
)

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

query = input("Enter your question: ")

results = collection.query(
    query_texts=[query],
    n_results=1
)

print("\nBest Match:")
print(results["documents"][0][0])