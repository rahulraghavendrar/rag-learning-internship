from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

print("Number of documents:", len(documents))

print()

print(documents[0])