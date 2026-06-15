from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_qdrant import (
    QdrantVectorStore
)

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    VectorParams
)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


pdf_files = [
    "internship.pdf",
    "company_policy.pdf",
    "employee_handbook.pdf",
    "training_program.pdf",
    "ai_projects.pdf"
]


all_documents = []

for pdf in pdf_files:

    loader = PyPDFLoader(pdf)

    docs = loader.load()

    all_documents.extend(docs)


questions = [
    (
        "How long is the internship?",
        "2 months"
    ),

    (
        "What stipend do interns receive?",
        "15000"
    ),

    (
        "How are mentors assigned?",
        "mentor"
    ),

    (
        "How many training modules exist?",
        "module"
    ),

    (
        "What AI projects are available?",
        "project"
    )
]


def evaluate_strategy(
        strategy_name,
        chunks
):

    print(
        f"\n{'='*60}"
    )

    print(
        strategy_name
    )

    print(
        f"{'='*60}"
    )


    client = QdrantClient(
        ":memory:"
    )


    client.create_collection(
        collection_name="test",

        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )


    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="test",
        embedding=embedding_model
    )


    vectorstore.add_documents(
        chunks
    )


    retriever = vectorstore.as_retriever(
        search_kwargs={"k":3}
    )


    correct = 0


    for question, keyword in questions:

        results = retriever.invoke(
            question
        )


        retrieved_text = ""

        for doc in results:

            retrieved_text += (
                doc.page_content.lower()
            )


        if keyword.lower() in retrieved_text:

            correct += 1


    accuracy = (
        correct /
        len(questions)
    ) * 100


    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    return accuracy


recursive_splitter = (
    RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
)

recursive_chunks = (
    recursive_splitter.split_documents(
        all_documents
    )
)


sliding_chunks = (
    RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    ).split_documents(
        all_documents
    )
)


semantic_chunks = (
    SemanticChunker(
        embeddings=embedding_model
    ).split_documents(
        all_documents
    )
)


recursive_score = (
    evaluate_strategy(
        "Recursive Chunking",
        recursive_chunks
    )
)

sliding_score = (
    evaluate_strategy(
        "Sliding Window Chunking",
        sliding_chunks
    )
)

semantic_score = (
    evaluate_strategy(
        "Semantic Chunking",
        semantic_chunks
    )
)


print("\nFINAL RESULTS\n")

print(
    f"Recursive : {recursive_score:.2f}%"
)

print(
    f"Sliding   : {sliding_score:.2f}%"
)

print(
    f"Semantic  : {semantic_score:.2f}%"
)