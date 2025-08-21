import json
import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score
from itertools import combinations

# ========================
#  CLIENT SETUP
# ========================
client_openai = None

def init_openai_client(credentials_file="credentials.json"):
    """
    Initialize OpenAI or AzureOpenAI client from a JSON credentials file.
    If azure_endpoint is present -> use AzureOpenAI
    Otherwise use standard OpenAI.
    """
    global client_openai

    with open(credentials_file, "r") as f:
        creds = json.load(f)

    api_key = creds.get("api_key")
    azure_endpoint = creds.get("azure_endpoint")
    api_version = creds.get("api_version")

    if azure_endpoint:
        print("🔹 Using AzureOpenAI")
        client_openai = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
    else:
        print("🔹 Using standard OpenAI")
        client_openai = OpenAI(api_key=api_key)

# ========================
#  EMBEDDING METHODS
# ========================
model_st = SentenceTransformer('intfloat/multilingual-e5-large')

# ========================
#  PRECOMPUTE EMBEDDINGS
# ========================
def precompute_all_embeddings(songs_by_year, method="openai", embedding_model="text-embedding-3-small", colbert_model="antoinelouis/colbert-xm"):
    """
    Compute embeddings according to the selected method.
    - method: "openai" | "sentence_transformers" | "colbert"
    """
    if method == "openai":
        if client_openai is None:
            raise RuntimeError("⚠️ You must first call init_openai_client() with a valid credentials file")

        embeddings_cache = {}
        for year in tqdm(range(1951, 2026)):
            year_str = str(year)
            for el in songs_by_year.get(year_str, []):
                response = client_openai.embeddings.create(
                    input=el['text'],
                    model=embedding_model
                )
                embeddings_cache.setdefault(year_str, {"songs": [], "embeddings": []})
                embeddings_cache[year_str]["songs"].append(el['song'])
                embeddings_cache[year_str]["embeddings"].append(response.data[0].embedding)

        # Convert lists to numpy arrays
        for year_str in embeddings_cache:
            embeddings_cache[year_str]["embeddings"] = np.array(embeddings_cache[year_str]["embeddings"])
        return embeddings_cache

    elif method == "sentence_transformers":
        embeddings_cache = {}
        for year in tqdm(range(1951, 2026)):
            year_str = str(year)
            for el in songs_by_year.get(year_str, []):
                embeddings_cache.setdefault(year_str, {"songs": [], "embeddings": []})
                embeddings_cache[year_str]["songs"].append(el['song'])
                embeddings_cache[year_str]["embeddings"].append(model_st.encode(el['text']))

        for year_str in embeddings_cache:
            embeddings_cache[year_str]["embeddings"] = np.array(embeddings_cache[year_str]["embeddings"])
        return embeddings_cache

    elif method == "colbert":
        config = ColBERTConfig(doc_maxlen=500, nbits=2)
        ckpt = Checkpoint(colbert_model, colbert_config=config)

        all_texts, years_list = [], []
        for year in range(1951, 2026):
            year_str = str(year)
            year_texts = [el['text'] for el in songs_by_year.get(year_str, [])]
            all_texts.extend(year_texts)
            years_list.extend([year_str] * len(year_texts))

        D = ckpt.docFromText(all_texts, bsize=32)[0]
        D_mask = torch.ones(D.shape[:2], dtype=torch.long)
        D = D.detach().cpu().numpy()
        D_mask = D_mask.detach().cpu().numpy()

        embeddings_cache = {}
        for year in range(1951, 2026):
            year_str = str(year)
            indices = [i for i, y in enumerate(years_list) if y == year_str]
            embeddings_cache[year_str] = {
                "songs": [el['song'] for el in songs_by_year.get(year_str, [])],
                "embeddings": {
                    "D": D[indices],
                    "D_mask": D_mask[indices]
                }
            }
        return embeddings_cache

    else:
        raise ValueError("Invalid method. Use 'openai', 'sentence_transformers' or 'colbert'.")


# ========================
#  SIMILARITY MATRIX
# ========================
def sim_matrix_years(year1, year2, dict_emb_songs, normalize=False, exclude_diagonal=True):
    year1, year2 = str(year1), str(year2)

    labels1 = [x.split(" - ")[1] if " - " in x else x for x in dict_emb_songs[year1]['songs']]
    labels2 = [x.split(" - ")[1] if " - " in x else x for x in dict_emb_songs[year2]['songs']]

    embeddings1 = dict_emb_songs[year1]['embeddings']
    embeddings2 = dict_emb_songs[year2]['embeddings']

    # ColBERT case
    if isinstance(embeddings1, dict) and "D" in embeddings1:
        D1, D2 = embeddings1['D'], embeddings2['D']
        mask1, mask2 = embeddings1['D_mask'], embeddings2['D_mask']

        D1_tensor = torch.from_numpy(D1)
        D2_tensor = torch.from_numpy(D2)
        mask1_tensor = torch.from_numpy(mask1)
        mask2_tensor = torch.from_numpy(mask2)

        sim_QD = np.zeros((D1.shape[0], D2.shape[0]))
        sim_DQ = np.zeros((D2.shape[0], D1.shape[0]))

        for i, q in enumerate(D1_tensor):
            sim_QD[i] = colbert_score(q.unsqueeze(0), D2_tensor, D_mask=mask2_tensor).cpu().numpy()
        for i, q in enumerate(D2_tensor):
            sim_DQ[i] = colbert_score(q.unsqueeze(0), D1_tensor, D_mask=mask1_tensor).cpu().numpy()

        similarity_matrix = sim_QD + sim_DQ.T

    # Standard case (OpenAI / SBERT)
    else:
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    if normalize:
        similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))

    if exclude_diagonal and similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        np.fill_diagonal(similarity_matrix, np.nan)

    return similarity_matrix, labels1, labels2



def semantic_similarity_over_time_quartile(embeddings, years=range(1951, 2026), percentile_threshold=0.75):
    """
    Compute the year-by-year semantic similarity matrix using a quartile threshold.
    Returns only the final matrix of percentages over the threshold.
    """
    all_values = []

    # Step 1: collect all similarity values
    for y in tqdm(years, desc="Collecting similarity values"):
        for y2 in range(y, max(years)+1):
            sim_matrix, _, _ = sim_matrix_years(y, y2, embeddings, normalize=False, exclude_diagonal=y==y2)
            all_values += sim_matrix.flatten().tolist()

    # Step 2: compute threshold
    threshold_value = round(pd.Series(all_values).quantile(percentile_threshold), 5)

    # Step 3: build year-by-year percentage matrix
    over_threshold_matrix = []
    for y1 in tqdm(years, desc="Computing percentages per year"):
        row = []
        for y2 in years:
            _, percentage = get_similarity_percentage(
                y1, y2, embeddings, threshold=threshold_value, exclude_diagonal=y1==y2, normalize=False
            )
            row.append(percentage)
        over_threshold_matrix.append(row)

    return over_threshold_matrix



def semantic_similarity_over_time_mean(embeddings, years=range(1951, 2026), normalize=True):
    """
    Compute the year-by-year mean semantic similarity matrix.
    - embeddings: dict con embeddings per anno
    - normalize: se True, normalizza la matrice tra 0 e 1
    Returns only the final mean similarity matrix.
    """
    mean_matrix = []

    for y1 in tqdm(years, desc="Computing mean similarity per year"):
        embeddings1 = embeddings[y1]['embeddings']
        row = []
        for y2 in years:
            embeddings2 = embeddings[y2]['embeddings']

            if y1 != y2:
                sim = np.mean(cosine_similarity(embeddings1, embeddings2))
            else:
                matrix = cosine_similarity(embeddings1, embeddings2)
                np.fill_diagonal(matrix, np.nan)
                sim = np.nanmean(matrix)
            row.append(sim)
        mean_matrix.append(row)

    mean_matrix = np.array(mean_matrix)

    if normalize:
        mean_matrix = (mean_matrix - np.nanmin(mean_matrix)) / (np.nanmax(mean_matrix) - np.nanmin(mean_matrix))

    return mean_matrix


def semantic_similarity_over_time_median(embeddings, years=range(1951, 2026), normalize=True):
    """
    Compute the year-by-year median semantic similarity matrix.
    - embeddings: dict con embeddings per anno
    - normalize: se True, normalizza la matrice tra 0 e 1
    Returns only the final median similarity matrix.
    """
    median_matrix = []

    for y1 in tqdm(years, desc="Computing median similarity per year"):
        embeddings1 = embeddings[y1]['embeddings']
        row = []
        for y2 in years:
            embeddings2 = embeddings[y2]['embeddings']

            if y1 != y2:
                sim = np.median(cosine_similarity(embeddings1, embeddings2))
            else:
                matrix = cosine_similarity(embeddings1, embeddings2)
                np.fill_diagonal(matrix, np.nan)
                sim = np.nanmedian(matrix)
            row.append(sim)
        median_matrix.append(row)

    median_matrix = np.array(median_matrix)

    if normalize:
        median_matrix = (median_matrix - np.nanmin(median_matrix)) / (np.nanmax(median_matrix) - np.nanmin(median_matrix))

    return median_matrix

# ========================
#  PORTIONS EMBEDDING METHODS
# ========================
def precompute_all_embeddings_portions(song_portions_dict, method="openai", embedding_model="text-embedding-3-small", colbert_model="antoinelouis/colbert-xm", client_openai=None):
    """
    Compute embeddings for song portions.
    method: "openai" | "sentence_transformers" | "colbert"
    """
    if method == "openai":
        if client_openai is None:
            raise RuntimeError("You must provide an initialized OpenAI/Azure client.")

        embeddings_cache = {}
        for song, segments in tqdm(song_portions_dict.items()):
            response = client_openai.embeddings.create(input=segments, model=embedding_model)
            embeddings_cache[song] = {
                "embeddings": np.array([d.embedding for d in response.data]),
                "segments": list(range(len(segments)))
            }
        return embeddings_cache

    elif method == "sentence_transformers":
        model_st = SentenceTransformer('intfloat/multilingual-e5-large')
        embeddings_cache = {}
        for song, segments in tqdm(song_portions_dict.items()):
            embeddings_cache[song] = {
                "embeddings": np.array([model_st.encode(s) for s in segments]),
                "segments": list(range(len(segments)))
            }
        return embeddings_cache

    elif method == "colbert":
        config = ColBERTConfig(doc_maxlen=500, nbits=2)
        ckpt = Checkpoint(colbert_model, colbert_config=config)

        all_texts, ids = [], []
        for song, segments in song_portions_dict.items():
            for i, s in enumerate(segments):
                ids.append(f"{song}#{i}")
                all_texts.append(s)

        D = ckpt.docFromText(all_texts, bsize=32)[0]
        D_mask = torch.ones(D.shape[:2], dtype=torch.long)

        D = D.detach().cpu().numpy()
        D_mask = D_mask.detach().cpu().numpy()

        embeddings_cache = {}
        for song in song_portions_dict:
            indices = [j for j, k in enumerate(ids) if k.split("#")[0] == song]
            embeddings_cache[song] = {
                "embeddings": {"D": D[indices], "D_mask": D_mask[indices]},
                "segments": list(range(len(song_portions_dict[song])))
            }
        return embeddings_cache

    else:
        raise ValueError("Invalid method. Use 'openai', 'sentence_transformers', or 'colbert'.")


def compute_song_similarities_from_portions(dict_emb_portions, top_k=4):
    """
    Compute song-to-song similarities from portion embeddings.
    Supports OpenAI / SBERT / ColBERT embeddings.
    Aggregates top-k portion similarities for each song pair.
    """
    songs = list(dict_emb_portions.keys())
    song_similarities = {}

    for song1, song2 in tqdm(combinations(songs, 2)):
        embeddings1 = dict_emb_portions[song1]["embeddings"]
        embeddings2 = dict_emb_portions[song2]["embeddings"]

        # ColBERT case
        if isinstance(embeddings1, dict) and "D" in embeddings1:
            D1, D2 = embeddings1['D'], embeddings2['D']
            mask1, mask2 = embeddings1['D_mask'], embeddings2['D_mask']

            D1_tensor = torch.from_numpy(D1)
            D2_tensor = torch.from_numpy(D2)
            mask1_tensor = torch.from_numpy(mask1)
            mask2_tensor = torch.from_numpy(mask2)

            sim_QD = np.zeros((D1.shape[0], D2.shape[0]))
            sim_DQ = np.zeros((D2.shape[0], D1.shape[0]))

            for j, q in enumerate(D1_tensor):
                sim_QD[j] = colbert_score(q.unsqueeze(0), D2_tensor, D_mask=mask2_tensor).cpu().numpy()
            for j, q in enumerate(D2_tensor):
                sim_DQ[j] = colbert_score(q.unsqueeze(0), D1_tensor, D_mask=mask1_tensor).cpu().numpy()

            similarity_matrix = sim_QD + sim_DQ.T

        # Standard case (OpenAI / SBERT)
        else:
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        # Flatten and sort similarities
        similarities = sorted(
            [(i, j, similarity_matrix[i, j]) for i in range(similarity_matrix.shape[0]) for j in range(similarity_matrix.shape[1])],
            key=lambda x: x[2],
            reverse=True
        )
        top_sim = [s[2] for s in similarities[:top_k]]

        song_similarities[(song1, song2)] = {
            "similarities": {(i, j): s for i, j, s in similarities},
            **{f"total_similarity_{n+1}": sum(top_sim[:n+1]) for n in range(top_k)}
        }
        song_similarities[(song2, song1)] = song_similarities[(song1, song2)]

    return song_similarities



def _get_max_total_key(song_similarities, s1, s2):
    """Return the highest 'total_similarity_{n}' key available for a song pair."""
    keys = [k for k in song_similarities[(s1, s2)].keys() if k.startswith("total_similarity_")]
    if not keys:
        return None
    # Get the key with the highest n
    return max(keys, key=lambda x: int(x.split("_")[-1]))

def semantic_similarity_over_time_portions_quartile(song_similarities, year_songs_dict, years=range(1951, 2026)):
    """
    Compute year-by-year similarity matrix using the 75th percentile of top portion similarities.
    """
    sim_matrix = []
    for y1 in tqdm(years, desc="Computing quartile similarities"):
        row = []
        songs1 = year_songs_dict[y1]
        for y2 in years:
            songs2 = year_songs_dict[y2]
            all_values = []
            for s1 in songs1:
                for s2 in songs2:
                    if (s1, s2) in song_similarities:
                        key = _get_max_total_key(song_similarities, s1, s2)
                        if key:
                            all_values.append(song_similarities[(s1, s2)][key])
            row.append(np.quantile(all_values, 0.75) if all_values else np.nan)
        sim_matrix.append(row)
    return np.array(sim_matrix)

def semantic_similarity_over_time_portions_mean(song_similarities, year_songs_dict, years=range(1951, 2026), normalize=True):
    """
    Compute year-by-year mean similarity matrix using top portion similarities per song pair.
    """
    sim_matrix = []
    for y1 in tqdm(years, desc="Computing mean similarities"):
        row = []
        songs1 = year_songs_dict[y1]
        for y2 in years:
            songs2 = year_songs_dict[y2]
            all_values = []
            for s1 in songs1:
                for s2 in songs2:
                    if (s1, s2) in song_similarities:
                        key = _get_max_total_key(song_similarities, s1, s2)
                        if key:
                            all_values.append(song_similarities[(s1, s2)][key])
            row.append(np.nanmean(all_values) if all_values else np.nan)
        sim_matrix.append(row)
    sim_matrix = np.array(sim_matrix)
    if normalize:
        sim_matrix = (sim_matrix - np.nanmin(sim_matrix)) / (np.nanmax(sim_matrix) - np.nanmin(sim_matrix))
    return sim_matrix

def semantic_similarity_over_time_portions_median(song_similarities, year_songs_dict, years=range(1951, 2026), normalize=True):
    """
    Compute year-by-year median similarity matrix using top portion similarities per song pair.
    """
    sim_matrix = []
    for y1 in tqdm(years, desc="Computing median similarities"):
        row = []
        songs1 = year_songs_dict[y1]
        for y2 in years:
            songs2 = year_songs_dict[y2]
            all_values = []
            for s1 in songs1:
                for s2 in songs2:
                    if (s1, s2) in song_similarities:
                        key = _get_max_total_key(song_similarities, s1, s2)
                        if key:
                            all_values.append(song_similarities[(s1, s2)][key])
            row.append(np.nanmedian(all_values) if all_values else np.nan)
        sim_matrix.append(row)
    sim_matrix = np.array(sim_matrix)
    if normalize:
        sim_matrix = (sim_matrix - np.nanmin(sim_matrix)) / (np.nanmax(sim_matrix) - np.nanmin(sim_matrix))
    return sim_matrix
