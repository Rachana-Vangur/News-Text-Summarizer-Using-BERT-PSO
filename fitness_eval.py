import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(reference, hypothesis):
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """
    scores = scorer.score(reference, hypothesis)
    return {k: v.fmeasure for k, v in scores.items()}


def redundancy_penalty(embeddings):
    """
    Mean cosine similarity between selected embeddings = redundancy.
    """
    if embeddings.shape[0] < 2:
        return 0.0
    sims = cosine_similarity(embeddings)
    tri = np.triu_indices_from(sims, k=1)
    return float(np.mean(sims[tri]))
