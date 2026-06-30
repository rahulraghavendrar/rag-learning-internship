from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Document Search")

documents = [
    "RAG stands for Retrieval Augmented Generation",
    "Embeddings convert text into vectors",
    "Vector databases store embeddings"
]

@mcp.tool()
def search_docs(query: str) -> str:
    for doc in documents:
        if query.lower() in doc.lower():
            return doc

    return "No document found"

if __name__ == "__main__":
    mcp.run()