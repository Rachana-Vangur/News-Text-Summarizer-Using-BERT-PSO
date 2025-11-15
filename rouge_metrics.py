# from rouge_score import rouge_scorer
#
#
# def evaluate_summary(reference, generated):
#     scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
#     scores = scorer.score(reference, generated)
#     return {k: v.fmeasure for k, v in scores.items()}


# rouge_metrics.py
from rouge_score import rouge_scorer


def compute_rouge(reference, hypothesis):
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    between reference and generated summaries.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {metric: round(values.fmeasure, 4) for metric, values in scores.items()}
