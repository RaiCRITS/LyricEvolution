import os
import re
import random
from collections import Counter
from itertools import combinations
import numpy as np
from tqdm import tqdm

import spacy
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

# =========================
#  CACHING DEI MODELLI
# =========================

# Mappa delle lingue ai modelli spaCy
SPACY_LANGUAGE_MODELS = {
    "en": "en_core_web_lg",
    "it": "it_core_news_lg",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
    "pt": "pt_core_news_lg",
    "nl": "nl_core_news_lg",
    "xx": "xx_ent_wiki_lg",
}

# Dizionario per memorizzare i modelli già caricati in RAM
LOADED_MODELS = {}

def get_nlp_model(lang):
    """Carica il modello solo se non è già presente in cache."""
    model_name = SPACY_LANGUAGE_MODELS.get(lang, "xx_ent_wiki_lg")
    
    if model_name not in LOADED_MODELS:
        try:
            print(f"--- Caricamento modello spaCy per lingua: {lang} ({model_name}) ---")
            LOADED_MODELS[model_name] = spacy.load(model_name, disable=["parser", "ner"])
        except OSError:
            print(f"--- Download modello {model_name} in corso... ---")
            os.system(f"python -m spacy download {model_name}")
            LOADED_MODELS[model_name] = spacy.load(model_name, disable=["parser", "ner"])
            
    return LOADED_MODELS[model_name]

def load_spacy_model_for_text(text):
    """Rileva la lingua e restituisce il modello dalla cache."""
    try:
        lang = detect(text)
    except:
        lang = "en"
    return get_nlp_model(lang)

# =========================
#  PREPROCESS
# =========================
def preprocess(text, nlp, remove_duplicates=False):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ"}]

    if remove_duplicates:
        tokens = list(dict.fromkeys(tokens))
    return tokens

# =========================
#  SIMILARITY
# =========================
def compute_similarity_dict(text1, text2, text1n, text2n, nlp1, nlp2):
    variants = {
        "no_rep_raw": preprocess(text1, nlp1, remove_duplicates=False),
        "rep_raw": preprocess(text1, nlp1, remove_duplicates=True),
        "no_rep_norm": preprocess(text1n, nlp1, remove_duplicates=False),
        "rep_norm": preprocess(text1n, nlp1, remove_duplicates=True),
    }
    variants2 = {
        "no_rep_raw": preprocess(text2, nlp2, remove_duplicates=False),
        "rep_raw": preprocess(text2, nlp2, remove_duplicates=True),
        "no_rep_norm": preprocess(text2n, nlp2, remove_duplicates=False),
        "rep_norm": preprocess(text2n, nlp2, remove_duplicates=True),
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
def precompute_all_similarities(texts_by_year):
    similarities_all = {}
    titles = []
    texts = []
    textsn = []

    text_nlp = dict()

    for year in tqdm(texts_by_year):
        entries = texts_by_year[year]
        for e in entries:
            titles.append(e["title"]+" "+str(year))
            texts.append(e["text"])
            textsn.append(e["normalized_text"])
            for t in texts:
              if t not in text_nlp:
                text_nlp[t] = load_spacy_model_for_text(t)


    from itertools import combinations
    for (i, t1), (j, t2) in tqdm(combinations(enumerate(texts), 2)):
        key = (titles[i], titles[j])
        key_rev = (titles[j], titles[i])
        t1n, t2n = textsn[i], textsn[j]
        sim_dict = compute_similarity_dict(t1, t2, t1n, t2n, text_nlp[t1], text_nlp[t2])
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

    return matrix

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
