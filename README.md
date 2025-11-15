# BBC Article Search UI

This repository now includes a simple Streamlit app to search BBC articles by keyword using the `bbc-news` Python package.

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

This will open the app in your browser. Enter a keyword and click "Search". Use the sidebar to adjust categories and result limits. Currently the app searches English categories exposed by the `bbc-news` client.
