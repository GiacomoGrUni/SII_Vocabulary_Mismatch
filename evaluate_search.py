import pandas as pd

from search import create_tfidf_index
from search import search_baseline
from search import search_with_expansion


# ========================================
# CARICAMENTO CORPUS
# ========================================

corpus = pd.read_csv(
    "data/movie_documents.csv"
)

print("Numero di documenti:", len(corpus))

movie_ids = corpus["movieId"].tolist()
documents = corpus["document"].tolist()


# ========================================
# CREAZIONE INDICE TF-IDF
# ========================================

print("\nCreazione indice TF-IDF...")

vectorizer, document_vectors = create_tfidf_index(
    documents
)

print("Indice creato!")


# ========================================
# CARICAMENTO TAG ORIGINALI
# ========================================

tags = pd.read_csv(
    "data/ml-20m/tags.csv"
)

tags = tags.dropna(subset=["tag"])
tags["tag"] = tags["tag"].str.lower()


# ========================================
# COPPIE DA TESTARE
# ========================================

test_pairs = [
    ("humour", "humor"),
    ("television", "tv"),
    ("nonsensical", "ridiculous"),
    ("drab", "gloomy"),
    ("remade", "remake"),
    ("teenager", "teenagers"),
    ("united states", "usa"),
    ("history", "story"),
    ("funny", "humour"),
    ("absurd", "ridiculous"),
]


# ========================================
# FUNZIONE GOLD SET
# ========================================

def create_gold_set(query, target):

    query_movies = set(
        tags.loc[
            tags["tag"] == query,
            "movieId"
        ]
    )

    target_movies = set(
        tags.loc[
            tags["tag"] == target,
            "movieId"
        ]
    )

    # Film che hanno il termine target
    # ma NON hanno il termine query

    gold_set = target_movies - query_movies

    return query_movies, target_movies, gold_set


# ========================================
# FUNZIONE VALUTAZIONE
# ========================================

def evaluate_ranking(
    ranking,
    movie_ids,
    gold_set,
    k
):

    top_k = ranking[:k]

    retrieved_movies = {
        movie_ids[index]
        for index, score in top_k
    }

    hits = retrieved_movies & gold_set

    precision = len(hits) / k

    if len(gold_set) > 0:
        recall = len(hits) / len(gold_set)
    else:
        recall = 0.0

    return precision, recall, hits


# ========================================
# VALUTAZIONE DI UNA COPPIA
# ========================================

def evaluate_pair(query, target):

    print("\n")
    print("=" * 70)
    print(f"QUERY: {query}")
    print(f"TARGET: {target}")
    print("=" * 70)

    # ----------------------------
    # GOLD SET
    # ----------------------------

    query_movies, target_movies, gold_set = create_gold_set(
        query,
        target
    )

    print("\nFilm con query:", len(query_movies))
    print("Film con target:", len(target_movies))
    print("Mismatch target:", len(gold_set))

    # ----------------------------
    # BASELINE
    # ----------------------------

    baseline_ranking = search_baseline(
        query,
        vectorizer,
        document_vectors,
        documents
    )

    # ----------------------------
    # QUERY EXPANSION
    # ----------------------------

    expanded_query, expansion_ranking = search_with_expansion(
        query,
        vectorizer,
        document_vectors,
        documents
    )

    print("\nQuery espansa:")
    print(expanded_query)

     # ----------------------------
    # CONTROLLO TOP 10 BASELINE
    # ----------------------------
    '''
    print("\nTop 10 Baseline:")

    top_10 = baseline_ranking[:10]

    for idx, score in top_10:

        movie_id = movie_ids[idx]

        if movie_id in gold_set:
            print(
                f"MovieID: {movie_id} | "
                f"Score: {score:.6f} | "
                f"GOLD HIT"
            )
        else:
            print(
                f"MovieID: {movie_id} | "
                f"Score: {score:.6f}"
            )

    print("Score min Top 10:", top_10[-1][1])
    '''

    # ----------------------------
    # RISULTATI
    # ----------------------------

    k_values = [5, 10, 20, 50]

    metrics = {}

    for k in k_values:

        baseline_precision, baseline_recall, _ = evaluate_ranking(
            baseline_ranking,
            movie_ids,
            gold_set,
            k
        )

        expansion_precision, expansion_recall, _ = evaluate_ranking(
            expansion_ranking,
            movie_ids,
            gold_set,
            k
        )

        metrics[f"baseline_p{k}"] = baseline_precision
        metrics[f"expansion_p{k}"] = expansion_precision

        metrics[f"baseline_r{k}"] = baseline_recall
        metrics[f"expansion_r{k}"] = expansion_recall

    # ----------------------------
    # SALVATAGGIO RISULTATI
    # ----------------------------

    results.append({
        "query": query,
        "target": target,
        "gold_size": len(gold_set),

        "baseline_p10": metrics["baseline_p10"],
        "expansion_p10": metrics["expansion_p10"],

        "baseline_r10": metrics["baseline_r10"],
        "expansion_r10": metrics["expansion_r10"],

        "baseline_p50": metrics["baseline_p50"],
        "expansion_p50": metrics["expansion_p50"],

        "baseline_r50": metrics["baseline_r50"],
        "expansion_r50": metrics["expansion_r50"],
    })

# ========================================
# ESECUZIONE ESPERIMENTI
# ========================================

results = []

for query, target in test_pairs:

    evaluate_pair(
        query,
        target
    )


# ========================================
# TABELLA FINALE
# ========================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 100)
print("TABELLA FINALE")
print("=" * 100)

print(
    results_df[
        [
            "query",
            "target",
            "gold_size",
            "baseline_p10",
            "expansion_p10",
            "baseline_r10",
            "expansion_r10",
            "baseline_p50",
            "expansion_p50",
            "baseline_r50",
            "expansion_r50",
        ]
    ].to_string(index=False)
)

# ========================================
# MEDIE
# ========================================

print("\n")
print("=" * 70)
print("MEDIE COMPLESSIVE")
print("=" * 70)

print(
    "\nPrecision@10:"
)

print(
    "Baseline:",
    results_df["baseline_p10"].mean()
)

print(
    "Expansion:",
    results_df["expansion_p10"].mean()
)

print(
    "\nRecall@10:"
)

print(
    "Baseline:",
    results_df["baseline_r10"].mean()
)

print(
    "Expansion:",
    results_df["expansion_r10"].mean()
)

print(
    "\nPrecision@50:"
)

print(
    "Baseline:",
    results_df["baseline_p50"].mean()
)

print(
    "Expansion:",
    results_df["expansion_p50"].mean()
)

print(
    "\nRecall@50:"
)

print(
    "Baseline:",
    results_df["baseline_r50"].mean()
)

print(
    "Expansion:",
    results_df["expansion_r50"].mean()
)

# ========================================
# ANALISI DEGLI ERRORI
# ========================================

print("\n")
print("=" * 100)
print("ANALISI DEGLI ERRORI")
print("=" * 100)

for _, row in results_df.iterrows():

    delta_p10 = row["expansion_p10"] - row["baseline_p10"]
    delta_p50 = row["expansion_p50"] - row["baseline_p50"]

    delta_r10 = row["expansion_r10"] - row["baseline_r10"]
    delta_r50 = row["expansion_r50"] - row["baseline_r50"]

    print("\nQuery:", row["query"])
    print("Target:", row["target"])

    print(f"P@10: {row['baseline_p10']:.3f} -> {row['expansion_p10']:.3f} "
          f"(Δ {delta_p10:+.3f})")

    print(f"R@10: {row['baseline_r10']:.3f} -> {row['expansion_r10']:.3f} "
          f"(Δ {delta_r10:+.3f})")

    print(f"P@50: {row['baseline_p50']:.3f} -> {row['expansion_p50']:.3f} "
          f"(Δ {delta_p50:+.3f})")

    print(f"R@50: {row['baseline_r50']:.3f} -> {row['expansion_r50']:.3f} "
          f"(Δ {delta_r50:+.3f})")