import os
import re
import random
from collections import Counter
from itertools import combinations
import numpy as np

import spacy
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import google.generativeai as genai

# =========================
#  GEMINI & NORMALIZATION
# =========================
# Variabile globale per il client
_GEMINI_API_KEY = None

def init_gemini_client(credentials_file):
    """
    Inizializza il client Gemini leggendo la chiave dal file di credenziali JSON.
    Il file deve contenere {"api_key": "..."}.
    """
    global _GEMINI_API_KEY
    with open(credentials_file, "r") as f:
        creds = json.load(f)
    _GEMINI_API_KEY = creds.get("api_key")
    if not _GEMINI_API_KEY:
        raise ValueError("API key not found in credentials file")
    
    # Configura il client con la chiave
    genai.configure(api_key=_GEMINI_API_KEY)
    print("Gemini client initialized.")

def query_gemini(prompt, model_name="gemini-1.5-flash"):
    """
    Fa una richiesta al modello Gemini usando la chiave inizializzata.
    """
    if _GEMINI_API_KEY is None:
        raise RuntimeError("Gemini client not initialized. Call init_gemini_client() first.")
    
    full_model_name = f"models/{model_name}"
    model = genai.GenerativeModel(model_name=full_model_name)
    response = model.generate_content([prompt], request_options={"timeout": 600})
    return response.text

    
def extract_normalized_text(response):
    match = re.search(r"\*\*\*(.*?)\*\*\*", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def normalize_with_gemini(text, api_key):
    prompt = f"""
    Here is a text: ***{text}***.
    Normalize it:
    - lowercase all words
    - verbs in infinitive
    - nouns/adjectives singular
    - return ONLY the normalized text in triple stars (***)
    """
    response = query_gemini(prompt, api_key)
    return extract_normalized_text(response)

# =========================
#  SPAcY MODEL LOADER (Large if available)
# =========================
SPACY_LANGUAGE_MODELS = {
    "en": "en_core_web_lg",
    "it": "it_core_news_lg",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
    "pt": "pt_core_news_lg",
    "nl": "nl_core_news_lg",
    "xx": "xx_ent_wiki_lg",  # multilingual large
}

def load_spacy_model_for_text(text):
    try:
        lang = detect(text)
    except:
        lang = "en"  # fallback

    model_name = SPACY_LANGUAGE_MODELS.get(lang, "xx_ent_wiki_lg")
    
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        os.system(f"python -m spacy download {model_name}")
        nlp = spacy.load(model_name, disable=["parser", "ner"])
    return nlp

# =========================
#  PREPROCESS
# =========================
def preprocess(text, nlp, remove_duplicates=False, normalize=False, api_key=None):
    if normalize:
        text = normalize_with_gemini(text, api_key)

    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ"}]

    if remove_duplicates:
        tokens = list(dict.fromkeys(tokens))
    return tokens

# =========================
#  SIMILARITY
# =========================
def compute_similarity_dict(text1, text2, nlp1, nlp2, api_key):
    variants = {
        "no_rep_raw": preprocess(text1, nlp1, remove_duplicates=False, normalize=False),
        "rep_raw": preprocess(text1, nlp1, remove_duplicates=True, normalize=False),
        "no_rep_norm": preprocess(text1, nlp1, remove_duplicates=False, normalize=True, api_key=api_key),
        "rep_norm": preprocess(text1, nlp1, remove_duplicates=True, normalize=True, api_key=api_key),
    }
    variants2 = {
        "no_rep_raw": preprocess(text2, nlp2, remove_duplicates=False, normalize=False),
        "rep_raw": preprocess(text2, nlp2, remove_duplicates=True, normalize=False),
        "no_rep_norm": preprocess(text2, nlp2, remove_duplicates=False, normalize=True, api_key=api_key),
        "rep_norm": preprocess(text2, nlp2, remove_duplicates=True, normalize=True, api_key=api_key),
    }

    results = {}
    for key in variants:
        v1, v2 = variants[key], variants2[key]
        if "no_rep" in key:
            sim = sum(min(v1.count(w), v2.count(w)) for w in set(v1) & set(v2))
        else:
            sim = len(set(v1) & set(v2))
        results[key] = sim
    return results

# =========================
#  MATRIX BY YEAR
# =========================
def precompute_all_similarities(texts_by_year, api_key):
    similarities_all = {}
    titles = []
    texts = []

    text_nlp = dict()

    for year, entries in texts_by_year.items():
        for e in entries:
            titles.append(e["title"]+" "+str(year))
            texts.append(e["text"])
            for t in texts:
              if t not in text_nlp:
                text_nlp[t] = load_spacy_model_for_text(t)


    from itertools import combinations
    for (i, t1), (j, t2) in tqdm(combinations(enumerate(texts), 2)):
        key = (titles[i], titles[j])
        key_rev = (titles[j], titles[i])
        sim_dict = compute_similarity_dict(t1, t2, text_nlp[t1], text_nlp[t2], api_key=api_key)
        similarities_all[key] = sim_dict
        similarities_all[key_rev] = sim_dict  # simmetria
    return similarities_all
# ============================
#  Compute matrices by year
# ============================
def compute_matrix_by_year(texts_by_year, similarities_all, similarity_type="no_dup_raw", func=np.mean):
    """
    Compute similarity matrix between years using precomputed similarities.
    func can be np.mean or np.median
    """
    years = sorted(texts_by_year.keys(), key=int)
    n = len(years)
    matrix = np.zeros((n, n))

    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            sims = []
            for e1 in texts_by_year[y1]:
                for e2 in texts_by_year[y2]:
                    key = (e1["title"]+" "+str(y1), e2["title"]+" "+str(y2))
                    if key in similarities_all:
                        sims.append(similarities_all[key][similarity_type])
            matrix[i, j] = func(sims) if sims else np.nan

    return matrix, years

def compute_mean_matrix(texts_by_year, similarities_all, similarity_type="no_dup_raw", normalize=True):
    mean_matrix = compute_matrix_by_year(texts_by_year, similarities_all, similarity_type, func=np.mean)
    if normalize:
        mean_matrix = (mean_matrix - np.min(mean_matrix)) / (np.max(mean_matrix) - np.min(mean_matrix))
    return mean_matrix

def compute_median_matrix(texts_by_year, similarities_all, similarity_type="no_dup_raw", normalize=True):
    median_matrix = compute_matrix_by_year(texts_by_year, similarities_all, similarity_type, func=np.median)
    if normalize:
        median_matrix = (median_matrix - np.min(median_matrix)) / (np.max(median_matrix) - np.min(median_matrix))
    return median_matrix
