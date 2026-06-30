---

# MCP (Model Context Protocol)

## What is MCP?

Model Context Protocol (MCP) is an open standard that allows Large Language Models (LLMs) to communicate with external tools and applications through a standardized interface.

Instead of directly calling Python functions, an LLM can invoke registered tools exposed by an MCP Server.

---

## Why MCP?

Without MCP

User
↓
LLM
↓
Python Function
↓
Qdrant

The function is tightly coupled to the application.

---

With MCP

User
↓
LLM
↓
MCP Tool
↓
Retriever
↓
Qdrant

The tool becomes reusable by any MCP-compatible client.

---

## MCP Components

### MCP Server

Hosts the available tools.

Example:

- retrieve_documents()

---

### MCP Tool

A Python function exposed using FastMCP.

Example:

```python
@mcp.tool()
def retrieve_documents(question: str):
    ...
```

---

### MCP Client

Any application capable of communicating with an MCP Server.

Examples:

- Claude Desktop
- Cursor
- VS Code
- MCP Inspector

---

## MCP in this Lab

The LangChain retriever was converted into an MCP Tool.

Pipeline:

Question
↓
retrieve_documents()
↓
LangChain Retriever
↓
Qdrant
↓
Top-K Chunks
↓
Returned Context

---

## Advantages

- Standardized Tool Interface
- Reusable Retrieval Logic
- LLM Agnostic
- Easy Integration with AI Agents
- Modular Architecture

---

## Libraries Used

- FastMCP
- LangChain
- Qdrant
- HuggingFace Embeddings
- PyPDFLoader

---

## Outcome

Successfully built an MCP Server exposing the retrieval operation as an MCP Tool.

The tool retrieves relevant chunks from the Qdrant vector database using the LangChain retriever.