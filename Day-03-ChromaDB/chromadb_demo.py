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

results = collection.query(
    query_texts=[
        "How much money do interns receive?"
    ],
    n_results=1
)

print(results["documents"])