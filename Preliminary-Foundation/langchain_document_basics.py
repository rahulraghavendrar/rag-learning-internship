from langchain_core.documents import Document
doc1=Document(
    page_content="Internship duration is 2 months"
    metadata={
        "source": "internship_info.txt",
        "page": 1
    }
)
doc2=Document(
    page_content="Interns receive a stipend of 15000 rupees per month",
    metadata={
        "source": "internship_info.txt",
        "page": 2
    }
)
print(doc1)
print(doc2)