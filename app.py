import streamlit as st
import re
from typing import List, Dict, Any
from typing import Optional
import datetime as _dt
import hashlib
import requests
from bs4 import BeautifulSoup
import numpy as np

try:
    import bbc
except Exception as exc:
    bbc = None
    _import_error = exc
else:
    _import_error = None

try:
    from gnews import GNews  # type: ignore
except Exception:
    GNews = None

try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None

# BERT + PSO summarization components
try:
    from bert_embeddings import BertEmbedder
    from generate_summary import SummaryGenerator
    from text_preprocess import preprocess_text
except Exception:
    BertEmbedder = None  # type: ignore
    SummaryGenerator = None  # type: ignore
    preprocess_text = None  # type: ignore


@st.cache_data(show_spinner=False)
def get_rss_category_map() -> Dict[str, str]:
    """
    Public BBC RSS feeds: https://www.bbc.co.uk/news/10628494#more-stories
    """
    return {
        "top_stories": "https://feeds.bbci.co.uk/news/rss.xml",
        "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "uk": "https://feeds.bbci.co.uk/news/uk/rss.xml",
        "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "health": "https://feeds.bbci.co.uk/news/health/rss.xml",
        "education": "https://feeds.bbci.co.uk/news/education/rss.xml",
        "science_and_environment": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "entertainment_and_arts": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
        "sport": "https://feeds.bbci.co.uk/sport/rss.xml",
        "newsbeat": "https://feeds.bbci.co.uk/news/newsbeat/rss.xml",
    }


@st.cache_data(show_spinner=False)
def get_categories() -> List[str]:
    cats = list(get_rss_category_map().keys())
    return cats


@st.cache_data(show_spinner=False)
def get_category_items(category: str) -> List[Dict[str, Any]]:
    """
    Fetch items for a category using public BBC RSS feeds.
    """
    rss_map = get_rss_category_map()
    url = rss_map.get(category)
    if not url or not feedparser:
        return []
    parsed = feedparser.parse(url)
    entries = parsed.entries or []
    normalized: List[Dict[str, Any]] = []
    for e in entries:
        title = getattr(e, "title", "") or ""
        link = getattr(e, "link", "") or ""
        summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
        # published date (optional)
        published = getattr(e, "published", None)
        published_dt = None
        if hasattr(e, "published_parsed") and getattr(e, "published_parsed") is not None:
            try:
                published_dt = _dt.datetime(*e.published_parsed[:6])
            except Exception:
                published_dt = None
        normalized.append(
            {"title": title, "link": link, "description": summary, "published": published or published_dt, "raw": e}
        )
    return normalized


@st.cache_data(show_spinner=False)
def google_news_search_bbc(query: str, max_results: int = 100) -> List[Dict[str, Any]]:
    """
    Fallback: use Google News to search BBC articles.
    """
    if not GNews or not query:
        return []
    try:
        gn = GNews(max_results=max_results)
        # Restrict to BBC domain
        results = gn.get_news_by_site('bbc.com') or []
        # If site fetch returns too broad/noisy, also try direct search query
        q_results = gn.get_news(query) or []
        merged = results + q_results
        # Normalize and filter to BBC domain
        normalized: List[Dict[str, Any]] = []
        for it in merged:
            url: Optional[str] = it.get("url") or ""
            if not url or ("bbc." not in url):
                continue
            title = it.get("title") or ""
            desc = it.get("description") or ""
            if query.lower() not in (title + " " + desc).lower():
                continue
            normalized.append({"title": title, "link": url, "description": desc, "raw": it})
        # Deduplicate by link
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for it in normalized:
            if it["link"] in seen:
                continue
            seen.add(it["link"])
            uniq.append(it)
        return uniq
    except Exception:
        return []

@st.cache_resource(show_spinner=False)
def get_summarization_components():
    """
    Initialize and cache BERT embedder and PSO summarizer.
    """
    if not (BertEmbedder and SummaryGenerator and preprocess_text):
        return None, None
    embedder = BertEmbedder()
    summarizer = SummaryGenerator(diversity_lambda=0.5)
    return embedder, summarizer


@st.cache_data(show_spinner=False)
def fetch_article_text(url: str, timeout: int = 12) -> str:
    """
    Fetch and heuristically extract main article text using simple HTML parsing.
    """
    if not url:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return ""
        html = resp.text
        soup = BeautifulSoup(html, "lxml")
        # Common BBC article content containers; fallback to paragraphs
        candidates = []
        for selector in [
            "article",
            "main",
            "div.ssrcss-1ocoo3l-ArticleWrapper",
            "div.ssrcss-uf6wea-RichTextComponentWrapper",
            "div.story-body__inner",
            "div#main-content",
        ]:
            section = soup.select_one(selector)
            if section:
                candidates.append(section)
        if not candidates:
            candidates = [soup.body] if soup.body else []
        texts: List[str] = []
        for node in candidates:
            for p in node.find_all(["p", "h2", "li"]):
                txt = p.get_text(" ", strip=True)
                if txt and len(txt.split()) >= 5:
                    texts.append(txt)
            if len(" ".join(texts)) > 400:
                break
        content = " ".join(texts).strip()
        return content
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def summarize_with_bert_pso(url: str, fallback_text: str = "", max_sentences: int = 3) -> str:
    """
    Fetch article text and produce a summary using BERT + PSO pipeline.
    Falls back to provided text if fetching fails.
    """
    embedder, summarizer = get_summarization_components()
    if not (embedder and summarizer and preprocess_text):
        return fallback_text or ""
    text = fetch_article_text(url)
    if not text:
        text = fallback_text
    if not text:
        return ""
    sentences = preprocess_text(text)
    if not sentences:
        return ""
    embeddings = embedder.embed_sentences(sentences)
    mean_emb = embeddings.mean(axis=0)
    sims = (embeddings @ mean_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_emb) + 1e-8)  # type: ignore[name-defined]
    import numpy as _np  # local import to avoid polluting global namespace
    title_idx = _np.argmax(sims)  # type: ignore[attr-defined]
    auto_title = sentences[title_idx]
    summary, _, _ = summarizer.summarize(sentences, auto_title, embeddings, max_sentences=max_sentences, model=embedder)
    return summary


def text_matches(query: str, *fields: str) -> bool:
    if not query:
        return True
    q = query.strip().lower()
    for f in fields:
        if f and q in f.lower():
            return True
    return False


def regex_matches(pattern: str, *fields: str) -> bool:
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return False
    for f in fields:
        if f and rx.search(f):
            return True
    return False


def main():
    st.set_page_config(page_title="BBC Article Search", page_icon="📰", layout="wide")
    st.title("📰 BBC Article Search")
    st.caption("Search BBC articles across categories using the `bbc-news` package.")

    if bbc is None:
        st.error(
            "The 'bbc' module (installed via 'bbc-news') could not be imported.\n\n"
            f"Details: {_import_error}\n\n"
            "Please ensure you've run: pip install bbc-news"
        )
        return

    with st.sidebar:
        st.header("Settings")
        all_categories = get_categories()
        search_all = st.checkbox("Search across all categories", value=True)
        if search_all:
            selected_categories = all_categories
        else:
            selected_categories = st.multiselect(
                "Categories",
                options=all_categories,
                default=all_categories,
                help="Select categories to search within",
            )
        debug_mode = st.checkbox("Debug mode", value=False, help="Show diagnostics and sample data")
        enable_gnews = st.checkbox("Enable Google News fallback", value=False, help="Search BBC via Google News if no results")
        if enable_gnews and GNews is None:
            st.warning("Google News fallback is enabled but the 'gnews' package is not installed. Install with: pip install gnews")
        num_sentences = st.slider("Summary length (sentences)", min_value=3, max_value=10, value=5, step=1)

    col_query, col_regex = st.columns([2, 1])
    with col_query:
        query = st.text_input("Keyword search", placeholder="e.g., climate, AI, economy", value="")
    with col_regex:
        use_regex = st.checkbox("Use regex", value=False, help="Interpret query as a regular expression")

    search_clicked = st.button("Search", type="primary")

    if search_clicked:
        if not selected_categories:
            st.info("Please select at least one category in the sidebar.")
            return

        num_found = 0
        if debug_mode:
            st.info(f"Searching in {len(selected_categories)} categories: {', '.join(selected_categories)}")
        for cat in selected_categories:
            items = get_category_items(cat)
            if debug_mode:
                st.write(f"Category '{cat}' returned {len(items)} item(s).")
            if not items:
                continue
            if use_regex and query:
                filtered = [it for it in items if regex_matches(query, it["title"], it["description"])]
            else:
                filtered = [it for it in items if text_matches(query, it["title"], it["description"])]

            if not filtered:
                if debug_mode:
                    st.write(f"No matches in '{cat}' for query: {query!r}")
                continue

            st.subheader(cat)
            for it in filtered:
                title = it["title"] or "(No title)"
                link = it["link"] or ""
                desc = it["description"] or ""
                st.markdown(f"- [{title}]({link})")
                # Summarization UI
                sum_key = f"summarize_{hashlib.md5(link.encode('utf-8')).hexdigest()}"
                with st.expander("Summary (BERT + PSO)"):
                    if not (BertEmbedder and SummaryGenerator and preprocess_text):
                        st.info("BERT+PSO modules not available. Showing feed summary instead.")
                        if desc:
                            st.write(desc)
                    else:
                        with st.spinner("Generating summary..."):
                            try:
                                bert_pso_summary = summarize_with_bert_pso(link, fallback_text=desc, max_sentences=num_sentences)
                            except Exception as e:
                                bert_pso_summary = desc or ""
                        if bert_pso_summary:
                            st.write(bert_pso_summary)
                        else:
                            st.write(desc or "No summary available.")
                num_found += 1

        if num_found == 0:
            # Global fallback via Google News if enabled
            if enable_gnews and query:
                gnews_items = google_news_search_bbc(query, max_results=200)
                if debug_mode:
                    st.write(f"GNews fallback returned {len(gnews_items)} item(s).")
                if gnews_items:
                    st.subheader("BBC (via Google News)")
                    for it in gnews_items:
                        title = it["title"] or "(No title)"
                        link = it["link"] or ""
                        desc = it["description"] or ""
                        st.markdown(f"- [{title}]({link})")
                        if desc:
                            with st.expander("Summary"):
                                st.write(desc)
                    num_found = len(gnews_items)
            if num_found == 0:
                st.warning("No matching articles found.")
        else:
            st.success(f"Found {num_found} matching article(s).")
    else:
        st.info("Enter a keyword and click Search, or toggle regex for advanced matching.")
        with st.expander("Browse a sample (first category)"):
            try:
                sample_cat = get_categories()[0]
                sample_items = get_category_items(sample_cat)
                st.write(f"Category: {sample_cat} — {len(sample_items)} item(s)")
                for it in sample_items[:10]:
                    st.markdown(f"- [{it['title'] or '(No title)'}]({it['link'] or '#'})")
            except Exception as e:
                st.write(f"Unable to load sample: {e}")


if __name__ == "__main__":
    main()


