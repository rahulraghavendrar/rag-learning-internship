from fastmcp import FastMCP

from rag import retrieve

mcp = FastMCP("RAG Retrieval")

@mcp.tool
def retrieve_documents(question: str) -> str:
    """Retrieve relevant chunks from the vector database."""
    return retrieve(question)

if __name__ == "__main__":
    mcp.run()