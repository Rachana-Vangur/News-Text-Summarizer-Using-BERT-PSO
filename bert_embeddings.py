import numpy as np
from sentence_transformers import SentenceTransformer


class BertEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_sentences(self, sentences, batch_size=32):
        """
        Generate BERT embeddings for a list of sentences.
        """
        return np.array(
            self.model.encode(
                sentences,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
        )
