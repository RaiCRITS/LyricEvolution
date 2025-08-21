import json
import numpy as np
import os
import pickle
from utils import embeddings as emb_utils
from utils import words as wrd_utils
from tqdm import tqdm
import argparse
import copy

# ===========================
#  PARSE ARGUMENTS
# ===========================
parser = argparse.ArgumentParser(description="Compute semantic similarity matrices for songs over years")
parser.add_argument("--credentials", type=str, required=True, help="Path to credentials JSON file")
parser.add_argument("--songs_file", type=str, required=True, help="Path to songs JSON file")
parser.add_argument("--output_dir", type=str, default="output_matrices", help="Directory to save matrices")
args = parser.parse_args()

CREDENTIALS_FILE = args.credentials
SONGS_FILE = args.songs_file
OUTPUT_DIR = args.output_dir
years = range(1951, 2026)
methods = ["openai", "sentence_transformers", "colbert"]

# ===========================
#  INIT OPENAI/AZURE CLIENT
# ===========================
emb_utils.init_openai_client(CREDENTIALS_FILE)

# ===========================
#  LOAD SONGS DATA
# ===========================
with open(SONGS_FILE, "r", encoding="utf-8") as f:
    songs_by_year_list = json.load(f)

# Prepare dictionaries for portion-level computations
songs_by_year_dict = {str(year): [] for year in years}
song_portions_dict = {}

for year, entries in songs_by_year_list.items():
    year = str(year)
    for el in entries:
        song_title = el['song']                # updated key
        songs_by_year_dict[year].append(song_title)
        song_portions_dict[song_title] = el.get('text_segments', [])  # updated key



# Deep copy per non modificare l'originale
songs_by_year_topics = copy.deepcopy(songs_by_year_list)

# Sostituire 'text' con 'topics' in tutti gli elementi
for year, entries in songs_by_year_topics.items():
    for el in entries:
        el['text'] = "\n".join(el['topics'])

# ===========================
#  FUNCTIONS TO COMPUTE MATRICES
# ===========================
def compute_all_text_matrices(songs_by_year_list):
    matrices = {}
    for method in methods:
        print(f"\n🔹 Computing embeddings for full texts using {method}...")
        emb_cache = emb_utils.precompute_all_embeddings(songs_by_year_list, method=method)

        print("  → Mean matrix")
        mean_mat = emb_utils.semantic_similarity_over_time_mean(emb_cache, years=years, normalize=True)

        print("  → Median matrix")
        median_mat = emb_utils.semantic_similarity_over_time_median(emb_cache, years=years, normalize=True)

        print("  → Quartile matrix")
        quartile_mat = emb_utils.semantic_similarity_over_time_quartile(emb_cache, years=years)

        matrices[method] = {
            "mean": mean_mat,
            "median": median_mat,
            "quartile": quartile_mat
        }
    return matrices

def compute_all_portion_matrices(song_portions_dict, songs_by_year_dict):
    matrices = {}
    for method in methods:
        print(f"\n🔹 Computing embeddings for portions using {method}...")
        emb_cache = emb_utils.precompute_all_embeddings_portions(
            song_portions_dict, method=method, client_openai=emb_utils.client_openai
        )

        print("  → Computing song similarities...")
        song_similarities = emb_utils.compute_song_similarities_from_portions(emb_cache)

        print("  → Mean matrix")
        mean_mat = emb_utils.semantic_similarity_over_time_portions_mean(
            song_similarities, songs_by_year_dict, years=years, normalize=True
        )

        print("  → Median matrix")
        median_mat = emb_utils.semantic_similarity_over_time_portions_median(
            song_similarities, songs_by_year_dict, years=years, normalize=True
        )

        print("  → Quartile matrix")
        quartile_mat = emb_utils.semantic_similarity_over_time_portions_quartile(
            song_similarities, songs_by_year_dict, years=years
        )

        matrices[method] = {
            "mean": mean_mat,
            "median": median_mat,
            "quartile": quartile_mat
        }
    return matrices

# ===========================
#  EXECUTION
# ===========================
print("=== Computing text-level matrices ===")
text_matrices = compute_all_text_matrices(songs_by_year_list)

print("\n=== Computing portion-level matrices ===")
portion_matrices = compute_all_portion_matrices(song_portions_dict, songs_by_year_dict)

print("\n=== Computing topic matrices ===")
topic_matrices = compute_all_text_matrices(songs_by_year_topics)



wrd_utils.init_gemini_client(CREDENTIALS_FILE)

def compute_all_words_matrices(songs_by_year_list):
    all_wd_similarities = wrd_utils.precompute_all_similarities(songs_by_year_list)
    methods = [
      "no_rep_raw",
      "rep_raw",
      "no_rep_norm",
      "rep_norm"
    ]
    matrices = {}
    for method in methods:
        print("  → Mean matrix")
        mean_mat = wrd_utils.compute_mean_matrix(songs_by_year_list, all_wd_similarities, method)

        print("  → Median matrix")
        median_mat = wrd_utils.compute_median_matrix(songs_by_year_list, all_wd_similarities, method)

        matrices[method] = {
            "mean": mean_mat,
            "median": median_mat
        }

    return matrices

print("\n=== Computing word matrices ===")
word_matrices = compute_all_words_matrices(songs_by_year_list)

# ===========================
#  SAVE RESULTS
# ===========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "text_matrices.pkl"), "wb") as f:
    pickle.dump(text_matrices, f)
with open(os.path.join(OUTPUT_DIR, "portion_matrices.pkl"), "wb") as f:
    pickle.dump(portion_matrices, f)
with open(os.path.join(OUTPUT_DIR, "topic_matrices.pkl"), "wb") as f:
    pickle.dump(topic_matrices, f)
with open(os.path.join(OUTPUT_DIR, "word_matrices.pkl"), "wb") as f:
    pickle.dump(word_matrices, f)

print(f"\n✅ All matrices computed and saved in '{OUTPUT_DIR}/'")
