import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fitness_eval import redundancy_penalty


class PSOOptimizer:
    def __init__(
        self, num_particles=30, max_iter=80, w=0.7, c1=1.5, c2=1.5, lambda_div=0.5
    ):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lambda_div = lambda_div

    def objective(self, weights, features, embeddings, target_k):
        weights = np.abs(weights)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
        weights /= np.sum(weights)
        scores = features.dot(weights)
        k = int(target_k)
        top_idx = np.argsort(scores)[-k:]
        selected_emb = embeddings[top_idx]
        score_sum = np.sum(scores[top_idx])
        diversity_penalty = redundancy_penalty(selected_emb)
        return -(score_sum - self.lambda_div * diversity_penalty)

    # def optimize(
    #     # self, features, embeddings, target_k, early_stop_tol=1e-5, verbose=False
    #     self, features, embeddings, target_k, n_particles=40, max_iter=150, w=0.8, c1=1.8, c2=1.8
    # ):
    def optimize(
        self,
        features,
        embeddings,
        target_k,
        n_particles=40,
        max_iter=150,
        w=0.8,
        c1=1.8,
        c2=1.8,
        early_stop_tol=1e-5,
        verbose=True,
    ):

        n_features = features.shape[1]
        particles = np.random.rand(self.num_particles, n_features)
        velocities = np.zeros_like(particles)
        personal_best = particles.copy()
        personal_best_scores = np.array(
            [self.objective(p, features, embeddings, target_k) for p in personal_best]
        )

        g_idx = np.argmin(personal_best_scores)
        global_best = personal_best[g_idx].copy()
        global_best_score = personal_best_scores[g_idx]

        history = []

        for it in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best[i] - particles[i])
                    + self.c2 * r2 * (global_best - particles[i])
                )
                particles[i] += velocities[i]
                score = self.objective(particles[i], features, embeddings, target_k)

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = particles[i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best = particles[i].copy()

            history.append(global_best_score)
            if verbose:
                print(
                    f"Iteration {it+1}/{self.max_iter} - Best Fitness: {-global_best_score:.4f}"
                )
            if it > 2 and abs(history[-1] - history[-2]) < early_stop_tol:
                break

        weights = np.abs(global_best)
        weights /= np.sum(weights)
        return weights, history
