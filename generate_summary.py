import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from pso_optimizer import PSOOptimizer
import re


class SummaryGenerator:
    def __init__(self, diversity_lambda=0.5):
        self.diversity_lambda = diversity_lambda

    def extract_features(self, sentences, title, embeddings, model):
        title_emb = model.embed_sentences([title])[0]
        title_sim = cosine_similarity(embeddings, title_emb.reshape(1, -1)).flatten()
        mean_emb = np.mean(embeddings, axis=0)
        global_imp = 1 - np.linalg.norm(embeddings - mean_emb, axis=1)
        redundancy = np.array(
            [np.max(cosine_similarity([emb], embeddings)) for emb in embeddings]
        )
        # Entity density feature: prefer sentences with people/places/dates/events
        entity_density = self._compute_entity_density(sentences)
        features = np.stack([title_sim, global_imp, redundancy, entity_density], axis=1)
        return MinMaxScaler().fit_transform(features)

    def _compute_entity_density(self, sentences):
        """
        Heuristic entity score per sentence.
        - Proper noun-like capitalized spans (People/Places/Orgs)
        - Dates/times (months, weekdays, years, time patterns)
        - Event keywords (election, summit, conference, protest, attack, launch, deal, meeting)
        Returns a normalized array (0..1).
        """
        months = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        weekdays = r"(Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)"
        years = r"(19|20)\\d{2}"
        timepat = r"\\b\\d{1,2}(:\\d{2})?\\s?(AM|PM|am|pm|GMT|BST|CET|IST)\\b"
        # Capitalized spans: avoid sentence-initial trivial match by requiring 2+ tokens or commas inside
        proper = r"\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+)\\b"
        events = r"\\b(election|summit|conference|protest|attack|strike|ceasefire|treaty|deal|agreement|launch|meeting|festival|tournament)\\b"

        scores = []
        for s in sentences:
            s_clean = s.strip()
            count = 0
            count += len(re.findall(proper, s_clean))
            count += len(re.findall(months, s_clean))
            count += len(re.findall(weekdays, s_clean))
            count += len(re.findall(years, s_clean))
            count += len(re.findall(timepat, s_clean))
            count += len(re.findall(events, s_clean, flags=re.IGNORECASE))
            # Penalize extremely short sentences
            if len(s_clean.split()) < 6:
                count = max(0, count - 1)
            scores.append(count)
        scores = np.array(scores, dtype=float)
        if scores.max() > 0:
            scores /= scores.max()
        return scores

    # def summarize(self, sentences, title, embeddings, max_sentences=3):
    #     features = self.extract_features(sentences, title, embeddings, self)
    #     optimizer = PSOOptimizer(lambda_div=self.diversity_lambda)
    #     weights, history = optimizer.optimize(features, embeddings, max_sentences)
    #     scores = features.dot(weights)
    #     top_idx = np.argsort(scores)[-max_sentences:]
    #     return " ".join([sentences[i] for i in sorted(top_idx)]), weights, history
    def summarize(self, sentences, title, embeddings, max_sentences=3, model=None):
        # model is passed in from main.py
        if model is None:
            raise ValueError("BERT model required for feature extraction")

        features = self.extract_features(sentences, title, embeddings, model)
        optimizer = PSOOptimizer(lambda_div=self.diversity_lambda)
        weights, history = optimizer.optimize(features, embeddings, max_sentences)
        scores = features.dot(weights)
        top_idx = np.argsort(scores)[-max_sentences:]
        return " ".join([sentences[i] for i in sorted(top_idx)]), weights, history
