import os

import numpy as np

from bert_embeddings import BertEmbedder
from generate_summary import SummaryGenerator
from rouge_metrics import compute_rouge
from text_preprocess import preprocess_text

# --------------------------
# 🧠 Initialize Components
# --------------------------
embedder = BertEmbedder()
summarizer = SummaryGenerator(diversity_lambda=0.5)

# --------------------------
# 📥 Step 1: Get Input Text
# --------------------------
print("=== BERT + PSO Text Summarizer ===")
choice = input("Enter 1 to paste custom text OR 2 to load from file: ").strip()

if choice == "1":
    text = input("\nPaste your text here:\n")
elif choice == "2":
    file_path = input("Enter file path: ").strip()
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
else:
    raise ValueError("Invalid choice. Please select 1 or 2.")

# --------------------------
# 🧹 Step 2: Preprocess & Embed
# --------------------------
sentences = preprocess_text(text)
embeddings = embedder.embed_sentences(sentences)

# --------------------------
# ⚙️ Step 3: Generate Summary
# --------------------------
# summary, weights, history = summarizer.summarize(
#     sentences, "Custom Input", embeddings, max_sentences=3, model=embedder
# )

# import numpy as np

# --------------------------
# ⚙️ Step 3: Generate Summary
# --------------------------

# Automatically choose the most representative sentence as the title
mean_emb = embeddings.mean(axis=0)

# Compute cosine similarity of each sentence embedding with the mean embedding
sims = np.dot(embeddings, mean_emb) / (
    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_emb) + 1e-8
)

# Pick the most central sentence as the "auto title"
title_idx = np.argmax(sims)
auto_title = sentences[title_idx]

print(f"\n🧠 Auto-generated title: {auto_title}\n")

# Now generate the summary
summary, weights, history = summarizer.summarize(
    sentences, auto_title, embeddings, max_sentences=3, model=embedder
)

# --------------------------
# 🧪 Step 4: Evaluate with ROUGE
# --------------------------
reference = input("\n(Optional) Enter reference summary (or press Enter to skip):\n")
if reference.strip():
    rouge = compute_rouge(reference, summary)
else:
    rouge = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

# --------------------------
# 💾 Step 5: Save Output
# --------------------------
os.makedirs("results", exist_ok=True)
output_file = "results/output_summary.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== BERT + PSO Text Summarizer ===\n\n")
    f.write("Original Text:\n" + text + "\n\n")
    f.write("=== Generated Summary ===\n" + summary + "\n\n")
    f.write(f"Feature Weights: {weights}\n\n")
    f.write(f"ROUGE Scores: {rouge}\n")

print(f"\n✅ Summary saved to: {output_file}")

# from bert_embeddings import BertEmbedder
# from generate_summary import SummaryGenerator
# from plot_results import plot_convergence
# from rouge_metrics import evaluate_summary
# from text_preprocess import preprocess_text
#
# demo_text = """
# Artificial intelligence has revolutionized industries worldwide.
# Machine learning finds patterns in massive data that humans miss.
# AI assists doctors, helps detect fraud, and powers self-driving cars.
# However, it raises concerns about bias, privacy, and unemployment.
# Responsible AI development and regulation are necessary.
# """
#
# reference_summary = """
# AI transforms industries but raises concerns about bias, privacy, and regulation.
# """
#
# # Pipeline
# sentences = preprocess_text(demo_text)
# embedder = BertEmbedder()
# embeddings = embedder.embed_sentences(sentences)
# summarizer = SummaryGenerator(diversity_lambda=0.5)
#
# summary, weights, history = summarizer.summarize(
#     sentences, "AI and Society", embeddings, max_sentences=3, model=embedder
# )
#
# print("\n--- Generated Summary ---\n", summary)
# print("\nFeature Weights:", weights)
#
# # Evaluation
# scores = evaluate_summary(reference_summary, summary)
# print("\nROUGE Scores:", scores)
#
# # Visualization
# plot_convergence(history)
