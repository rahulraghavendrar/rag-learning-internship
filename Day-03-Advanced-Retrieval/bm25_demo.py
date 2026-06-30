from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from rank_bm25 import BM25Okapi


loader = PyPDFLoader(
    "1-Week RAG Deep Dive Learning Plan.pdf"
)

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(
    documents
)


chunk_texts = [
    chunk.page_content
    for chunk in chunks
]


tokenized_chunks = [
    text.lower().split()
    for text in chunk_texts
]


bm25 = BM25Okapi(
    tokenized_chunks
)


while True:

    query = input(
        "\nAsk Question: "
    )

    if query.lower() == "exit":

        break

    tokenized_query = (
        query.lower().split()
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_results = sorted(
        zip(scores, chunk_texts),
        reverse=True
    )

    print("\nTOP RESULTS\n")

    for score, text in ranked_results[:3]:

        print("Score:", score)

        print()

        print(text)

        print("\n" + "-" * 60)