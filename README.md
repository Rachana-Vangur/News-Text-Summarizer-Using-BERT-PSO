
# News Text Summarizer using BERT and PSO algorithms

This repository now includes a simple Streamlit app to search BBC articles by keyword using the `bbc-news` Python package. And the results have a summarized version of the text. For summarization of the article, BERT and PSO are used.

## Setup

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you already installed `bbc-news`, ensure `streamlit` is installed:

```bash
python -m pip install streamlit
```

## Run the app

From the repository root:

```bash
streamlit run app.py
```


