import json
import numpy as np
import os
import pickle
from utils import embeddings as emb_utils
from utils import words as wrd_utils
from tqdm import tqdm
import argparse
import copy
import json
import seaborn as sns
import matplotlib.pyplot as plt 

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
methods = ["sentence_transformers", "colbert"]

flag_openai = False

with open(CREDENTIALS_FILE) as f:
    o = json.load(f)
    os.environ["HF_TOKEN"] = o["hf_token"]
    flag_openai = "openai" in o
    
if flag_openai:
    emb_utils.init_openai_client(CREDENTIALS_FILE)
    methods = ["openai"] + methods

# ===========================
#  LOAD SONGS DATA
# ===========================
with open(SONGS_FILE, "r", encoding="utf-8") as f:
    songs_by_year_list = json.load(f)

for k in list(songs_by_year_list.keys()):
  songs_by_year_list[int(k)] = songs_by_year_list[k]
  if str(k) in songs_by_year_list:
    del songs_by_year_list[str(k)]


years = [int(k) for k in list(songs_by_year_list.keys())]

# Prepare dictionaries for portion-level computations
songs_by_year_dict = {int(year): [] for year in years}
song_portions_dict = {}

for year, entries in songs_by_year_list.items():
    year = int(year)
    for el in entries:
        song_title = el['song']                # updated key
        songs_by_year_dict[year].append(song_title)
        song_portions_dict[song_title] = el.get('segments', [])  # updated key

# Deep copy per non modificare l'originale
songs_by_year_topics = copy.deepcopy(songs_by_year_list)

# Sostituire 'text' con 'topics' in tutti gli elementi
for year, entries in songs_by_year_topics.items():
    for el in entries:
        el['text'] = "\n".join([t['topic_title'] for t in el['topics']])


# ===========================
#  FUNCTIONS TO COMPUTE MATRICES
# ===========================
def compute_all_text_matrices(songs_by_year_list):
    matrices = {}
    for method in methods:
        print(f"\n🔹 Computing embeddings for full texts using {method}...")
        emb_cache = emb_utils.precompute_all_embeddings(songs_by_year_list, method=method)

        sim_years_values = emb_utils.get_sim_matrix_years_values(emb_cache,years)

        print("  → Mean matrix")
        mean_mat = emb_utils.semantic_similarity_over_time_mean(sim_years_values, years=years, normalize=True)

        print("  → Median matrix")
        median_mat = emb_utils.semantic_similarity_over_time_median(sim_years_values, years=years, normalize=True)

        print("  → Quartile matrix")
        quartile_mat = emb_utils.semantic_similarity_over_time_quartile(sim_years_values, years=years)

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


correlation_matrices = {
    "Full Text":text_matrices,
    "Portion Based":portion_matrices,
    "Topic Based":topic_matrices,
    "Word Based":word_matrices
}


def show_matrix(matrix_npy, matrix_name, years, output_dir, show = False):
    tick_labels = [year if int(year) % 5 == 0 else "" for year in years]
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(matrix_npy, cmap='coolwarm', vmin=0, vmax=1, cbar = False, xticklabels=tick_labels, yticklabels=tick_labels) #, cbar = True)
    
    
    # Imposta i tick principali ogni 5 anni
    major_ticks = [i + 0.5 for i, year in enumerate(years) if int(year) % 5 == 0]
    minor_ticks = [i + 0.5 for i, year in enumerate(years) if int(year) % 5 != 0]
    
    ax.set_xticks(major_ticks)  
    ax.set_xticklabels([year for year in years if int(year) % 5 == 0], fontsize=10)
    
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([year for year in years if int(year) % 5 == 0], fontsize=10)
    
    
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    
    ax.tick_params(axis='both', which='major', length=5, width=2)  # Più grandi e spessi
    
    ax.tick_params(axis='both', which='minor', length=2, width=1)  # Più piccoli e sottili
    
    plt.title(matrix_name)
    plt.savefig(f"{output_dir}/{matrix_name}.png", format="png", dpi=150)
    if show == True:
        plt.show()
    plt.close()

def mean_matrices(matrices):
    return np.mean(np.array(matrices), axis=0).tolist()


for method in correlation_matrices.keys():
    folder_path = f"{OUTPUT_DIR}/{method}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    for model in correlation_matrices[method].keys():
      for type_agg in correlation_matrices[method][model].keys():
        show_matrix(correlation_matrices[method][model][type_agg], f"{method}_{model}_{type_agg}", years, f"{OUTPUT_DIR}/{method}", show = False)

global_matrices = list()

for method in correlation_matrices.keys():
  folder_path = f"{OUTPUT_DIR}/{method}"
  method_matrices = list()
  if not os.path.exists(folder_path):
      os.mkdir(folder_path)
  for model in correlation_matrices[method].keys():
    type_agg = "mean"
    method_matrices.append(correlation_matrices[method][model][type_agg])
  
  show_matrix(mean_matrices(method_matrices), f"{method}_mean", years, f"{OUTPUT_DIR}/{method}", show = False)
  global_matrices.append(method_matrices)

show_matrix(mean_matrices(method_matrices), f"global_mean", years, f"{OUTPUT_DIR}", show = False)

