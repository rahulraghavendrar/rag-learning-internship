from bert_score import score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

reference="""
RAG combines retrieval and generation to answer questions using external knowledge.
"""

generated="""
RAG retrieves relevant information and uses it to generate answers.
"""

scorer=rouge_scorer.RougeScorer(
    ["rouge1","rougeL"],
    use_stemmer=True
)

rouge=scorer.score(
    reference,
    generated
)

P,R,F1=score(
    [generated],
    [reference],
    lang="en"
)

model=SentenceTransformer(
    "all-MiniLM-L6-v2"
)

ref_embedding=model.encode(
    reference
)

gen_embedding=model.encode(
    generated
)

similarity=cosine_similarity(
    [ref_embedding],
    [gen_embedding]
)[0][0]

reference_tokens=reference.split()
generated_tokens=generated.split()

overlap=len(
    set(reference_tokens).intersection(
        set(generated_tokens)
    )
)

bleu=overlap/len(
    generated_tokens
)

print(
    f"BLEU: {bleu:.4f}"
)

print(
    f"ROUGE-1: {rouge['rouge1'].fmeasure:.4f}"
)

print(
    f"ROUGE-L: {rouge['rougeL'].fmeasure:.4f}"
)

print(
    f"BERTScore: {F1.mean():.4f}"
)

print(
    f"Semantic Similarity: {similarity:.4f}"
)