from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("dummy.pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(
    documents
)

print("Number of Chunks:", len(chunks))

print()

print(chunks[0])