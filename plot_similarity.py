import os
import json
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# ===========================
# CONFIGURATION
# ===========================
parser = argparse.ArgumentParser(description="Plot similarity matrices and combined visualizations")
parser.add_argument("--songs_file", type=str, required=True, help="Path to songs JSON file")
parser.add_argument("--input_dir", type=str, default="output_matrices", help="Directory of saved matrices")
parser.add_argument("--output_dir", type=str, default="images", help="Where to save plots")
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
SONGS_FILE = args.songs_file

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# LOAD YEARS FROM JSON
# ===========================
with open(SONGS_FILE, "r", encoding="utf-8") as f:
    songs_by_year = json.load(f)
YEARS = sorted([int(y) for y in songs_by_year.keys()], key=lambda x: x)
YEARS = [str(y) for y in YEARS]  # keep as strings for labels

# ===========================
# HELPER FUNCTIONS
# ===========================
def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize matrix to [0, 1], preserving NaNs."""
    mn, mx = np.nanmin(matrix), np.nanmax(matrix)
    return (matrix - mn) / (mx - mn) if mx > mn else matrix

def plot_heatmap(matrix: np.ndarray, years, title: str, filename: str):
    """Plot a heatmap with ticks every 5 years."""
    mat_norm = normalize_matrix(matrix)
    mask = np.triu(np.ones_like(mat_norm, dtype=bool), k=1)

    tick_labels = [year if int(year) % 5 == 0 else "" for year in years]
    major_ticks = [i + 0.5 for i, year in enumerate(years) if int(year) % 5 == 0]

    plt.figure(figsize=(18, 18))
    ax = sns.heatmap(mat_norm, cmap="coolwarm", vmin=0, vmax=1, cbar=False,
                     xticklabels=tick_labels, yticklabels=tick_labels, mask=mask)

    ax.set_xticks(major_ticks)
    ax.set_xticklabels([year for year in years if int(year) % 5 == 0], fontsize=16)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([year for year in years if int(year) % 5 == 0], fontsize=16)

    minor_ticks = [i + 0.5 for i, year in enumerate(years) if int(year) % 5 != 0]
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)

    ax.tick_params(axis="both", which="major", length=6, width=2)
    ax.tick_params(axis="both", which="minor", length=3, width=1)

    plt.title(title, fontsize=20)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), format="eps", dpi=300)
    plt.close()

# ===========================
# LOAD PRE-COMPUTED MATRICES
# ===========================
with open(os.path.join(INPUT_DIR, "text_matrices.pkl"), "rb") as f: text_matrices = pickle.load(f)
with open(os.path.join(INPUT_DIR, "portion_matrices.pkl"), "rb") as f: portion_matrices = pickle.load(f)
with open(os.path.join(INPUT_DIR, "topic_matrices.pkl"), "rb") as f: topic_matrices = pickle.load(f)
with open(os.path.join(INPUT_DIR, "word_matrices.pkl"), "rb") as f: word_matrices = pickle.load(f)

categories = {
    "text": text_matrices,
    "portion": portion_matrices,
    "topic": topic_matrices,
    "word": word_matrices,
}

# ===========================
# PLOT EACH MATRIX
# ===========================
for cat, mats in categories.items():
    for method, stats in mats.items():
        for stat_name, mat in stats.items():
            if mat is not None:
                title = f"{cat.upper()} — {method} — {stat_name}"
                fname = f"{cat}_{method}_{stat_name}.eps"
                print(f"Plotting {title}")
                plot_heatmap(mat, YEARS, title, fname)

# ===========================
# COMBINE MATRICES FUNCTION
# ===========================
def combine_category_matrices(cat_stats, stat_type):
    normed_matrices = []
    for method, stats in cat_stats.items():
        mat = stats.get(stat_type)
        if mat is not None:
            normed_matrices.append(normalize_matrix(mat))
    if normed_matrices:
        return normalize_matrix(np.mean(normed_matrices, axis=0))
    return None

# Generate combined matrices
combined = {}
for stat in ["mean", "median"]:
    per_cat = []
    for cat in categories:
        nm = combine_category_matrices(categories[cat], stat)
        if nm is not None:
            per_cat.append(nm)
            fname = f"{cat}_combined_{stat}.eps"
            plot_heatmap(nm, YEARS, f"{cat.upper()} Combined {stat.upper()}", fname)
    if per_cat:
        combined_mat = normalize_matrix(np.mean(per_cat, axis=0))
        fname = f"combined_all_categories_{stat}.eps"
        plot_heatmap(combined_mat, YEARS, f"ALL CATEGORIES COMBINED {stat.upper()}", fname)
        combined[stat] = combined_mat
        np.save(os.path.join(INPUT_DIR, f"combined_{stat}.npy"), combined_mat)

print("✅ All plots saved and combined matrices computed.")
