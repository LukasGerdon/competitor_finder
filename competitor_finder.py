import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from pathlib import Path
import pandas as pd
import faiss
import requests
from bs4 import BeautifulSoup
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langdetect import detect
from streamlit.errors import StreamlitSecretNotFoundError

# -----------------------------
# Data Loading Utilities
# -----------------------------
def load_topics(file_path: Path) -> dict:
    try:
        return json.loads(file_path.read_text(encoding='utf-8'))
    except Exception as e:
        st.error(f"Error loading topics: {e}")
        return {}


def load_data_base(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error loading data base: {e}")
        return pd.DataFrame()
    

def load_api_key() -> str | None:
    try:
        return st.secrets["openai_api_key"]
    except (KeyError, StreamlitSecretNotFoundError):
        return None

# -----------------------------
# Model Initialization
# -----------------------------
def init_models(api_key: str):
    llm_client = OpenAI(api_key=api_key)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return llm_client, embed_model

# -----------------------------
# LLM Utilities
# -----------------------------
def translate_if_needed(llm_client: OpenAI, name: str, text: str) -> str:
    text = text.strip()
    if not text:
        return text
    try:
        if detect(text) == 'en':
            return text
    except Exception:
        pass
    words = text.split()
    approx = min(max(len(words) * 2, 128), 2000)
    prompt = (
        f"Please translate the following company description into clear, concise English, "
        f"preserving meaning. Use '{name}' as the company name in the output.\n\n{text}"
    )
    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=approx
    )
    return resp.choices[0].message.content.strip()


def match_topic_label(description: str, keyword_map: dict) -> str | None:
    desc = description.lower()
    scores = {topic: sum(kw.lower() in desc for kw in kws)
              for topic, kws in keyword_map.items()}
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score > 0 else None

# -----------------------------
# Search Functions
# -----------------------------
def internal_search(df: pd.DataFrame, embed_model, text: str, topic: str, top_k: int = 5) -> pd.DataFrame:
    subset = df[df['AssignedTopics'].str.contains(topic, case=False, na=False, regex=False)]
    if subset.empty:
        return pd.DataFrame()
    texts = subset['Description'].tolist()
    embeddings = embed_model.encode(texts).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    query_emb = embed_model.encode([text]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    results = subset.iloc[indices[0]].copy()
    results['Similarity'] = distances[0]
    return results


def format_for_prompt(df: pd.DataFrame) -> str:
    return "\n".join(
        f"- {row['Company name Latin alphabet']} ({row['BvD sectors']}, Growth: {row['Growth 2023']}%, "
        f"Employees: {row['Number of employees 2023']})\n"
        f"  Topic: {row['AssignedTopics']}\n  Description: {row['Description']}"
        for _, row in df.iterrows()
    )


def gpt_filter_internal(llm_client: OpenAI, desc: str, df: pd.DataFrame) -> str:
    prompt = f"""
A company is described as:
"{desc}"

Here are some high-growth companies:
{format_for_prompt(df)}

Select the top 2-3 competitors with details on why they're relevant.
"""
    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()


def gpt_external_search(llm_client: OpenAI, desc: str) -> str:
    prompt = f"""
A company is described as:
"{desc}"

Suggest up to 3 real-world competitors in Portugal. Include their website URLs and a short description.
"""
    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=500
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# Web Scraping & Extraction
# -----------------------------
def fetch_website_info(url: str, llm_client: OpenAI) -> tuple[str, str]:
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        raw_title = soup.title.string.strip() if soup.title and soup.title.string else ''
        name_prompt = (
            f"Extract the concise company name from this title: '{raw_title}'. Return only the name."
        )
        name_resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": name_prompt}],
            temperature=0.0,
            max_tokens=50
        )
        final_name = name_resp.choices[0].message.content.strip() or raw_title
        meta = soup.find('meta', attrs={'name':'description'}) or soup.find('meta', attrs={'property':'og:description'})
        raw_desc = meta['content'].strip() if meta and meta.get('content') else (
            soup.find('p').get_text(strip=True) if soup.find('p') else ''
        )
        desc_en = translate_if_needed(llm_client, final_name, raw_desc)
        return final_name, desc_en
    except Exception:
        return '', ''

# -----------------------------
# Streamlit App UI
# -----------------------------
def apply_css(css_file_path: Path):
    try:
        css_text = css_file_path.read_text(encoding='utf-8')
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")


def input_from_database(df: pd.DataFrame) -> tuple[str, str]:
    companies = df['Company name Latin alphabet'].dropna().unique().tolist()
    sel = st.selectbox("Company Name", ["--"] + companies)
    if sel != "--":
        row = df[df['Company name Latin alphabet'] == sel].iloc[0]
        desc = st.text_area("Company Description", value=row['Description'], height=150, key="db_desc")
        return desc
    return ''


def complete_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url
    scheme, rest = raw_url.split("://", 1)
    if not rest.startswith("www."):
        rest = "www." + rest
    return f"{scheme}://{rest}"


def input_via_url(llm_client: OpenAI) -> tuple[str, str]:
    url = st.text_input("Website URL", placeholder="https://example.com")
    if url:
        url = complete_url(url)
        with st.spinner("Fetching site info…"):
            name, desc = fetch_website_info(url, llm_client)
        if desc:
            name = st.text_input("Company Name", value=name, key="url_name")
            desc = st.text_area("Company Description", value=desc, height=150, key="url_desc")
            return desc
        else:
            st.warning("Couldn't extract from URL; please fill manually.")
    return ''


def input_manual() -> tuple[str, str]:
    name = st.text_input("Company Name", placeholder="Enter the company name")
    desc = st.text_area("Company Description", placeholder="Enter a description")
    return desc


def display_results(internal_resp: str, external_resp: str):
    st.markdown("### Analysis Results")
    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.markdown("#### Competitors (database)")
        st.markdown(internal_resp)
    with c2:
        st.markdown("#### Competitors (web search)")
        st.markdown(external_resp)


def main():
    task_dir = Path(__file__).resolve().parent
    st.set_page_config(page_title="Competitor Finder", layout="wide")
    apply_css(task_dir / "styles.css")
    st.title("🔍 Competitor Finder")

    topics_keywords = load_topics(task_dir / "topics.json")
    df = load_data_base(task_dir / "indexed_firms.xlsx")

    api_key = load_api_key()
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.info("🔒 Please add your OpenAI API key to continue.")
            return

    llm_client, embed_model = init_models(api_key)

    method = st.selectbox(
        "How would you like to provide the reference company?",
        ["--", "Select from database", "Company website URL", "Company name and description"],
        index=0
    )

    if method == "--":
        return
    elif method == "Select from database":
        description = input_from_database(df)
    elif method == "Company website URL":
        description = input_via_url(llm_client)
    else:
        description = input_manual()

    topic = match_topic_label(description, topics_keywords)

    if st.button("Find Competitors", use_container_width=True):
        if description and topic:
            with st.spinner("Analyzing..."):
                df_int = internal_search(df, embed_model, description, topic)
                internal_resp = (gpt_filter_internal(llm_client, description, df_int)
                                 if not df_int.empty else "No internal matches.")
                external_resp = gpt_external_search(llm_client, description)
            display_results(internal_resp, external_resp)
        else:
            st.warning("Please provide the necessary information and ensure a topic is detected.")

if __name__ == '__main__':
    main()
