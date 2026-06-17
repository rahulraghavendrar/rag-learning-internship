from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from sklearn.metrics import ndcg_score


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


for i, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = i


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


client = QdrantClient(":memory:")

client.create_collection(
    collection_name="evaluation",

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


vectorstore = QdrantVectorStore(
    client=client,
    collection_name="evaluation",
    embedding=embedding_model
)

vectorstore.add_documents(
    chunks
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k":10}
)


evaluation_data = [

    {
        "question":
        "What is the goal of Day 1?",

        "keyword":
        "rag pipeline"
    },

    {
        "question":
        "Name one chunking strategy.",

        "keyword":
        "semantic"
    },

    {
        "question":
        "What retrieval method is introduced in Day 3?",

        "keyword":
        "bm25"
    },

    {
        "question":
        "Name one evaluation framework.",

        "keyword":
        "ragas"
    },

    {
        "question":
        "Name one advanced rag variant.",

        "keyword":
        "agentic"
    }
]


recall_scores = []
precision_scores = []
mrr_scores = []
ndcg_scores = []


for item in evaluation_data:

    query = item["question"]

    keyword = item["keyword"].lower()

    results = retriever.invoke(
        query
    )


    relevant_positions = []

    retrieved_relevant = 0


    for rank, doc in enumerate(
        results,
        start=1
    ):

        if keyword in (
            doc.page_content.lower()
        ):

            retrieved_relevant += 1

            relevant_positions.append(
                rank
            )


    total_relevant = 1


    recall = (
        retrieved_relevant
        /
        total_relevant
    )

    precision = (
        retrieved_relevant
        /
        len(results)
    )


    if len(relevant_positions) > 0:

        first_rank = min(
            relevant_positions
        )

        mrr = (
            1 / first_rank
        )

    else:

        mrr = 0


    y_true = []

    y_score = []


    for rank, doc in enumerate(
        results,
        start=1
    ):

        if keyword in (
            doc.page_content.lower()
        ):

            y_true.append(1)

        else:

            y_true.append(0)

        y_score.append(
            len(results) - rank + 1
        )


    ndcg = ndcg_score(
        [y_true],
        [y_score]
    )


    recall_scores.append(
        recall
    )

    precision_scores.append(
        precision
    )

    mrr_scores.append(
        mrr
    )

    ndcg_scores.append(
        ndcg
    )


average_recall = (
    sum(recall_scores)
    /
    len(recall_scores)
)

average_precision = (
    sum(precision_scores)
    /
    len(precision_scores)
)

average_mrr = (
    sum(mrr_scores)
    /
    len(mrr_scores)
)

average_ndcg = (
    sum(ndcg_scores)
    /
    len(ndcg_scores)
)


print()

print(
    f"Recall@10    : {average_recall:.4f}"
)

print(
    f"Precision@10 : {average_precision:.4f}"
)

print(
    f"MRR          : {average_mrr:.4f}"
)

print(
    f"NDCG         : {average_ndcg:.4f}"
)