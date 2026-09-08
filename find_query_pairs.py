import pandas as pd
from nltk.corpus import wordnet


# ----------------------------------------
# CARICAMENTO TAG
# ----------------------------------------

path = "data/ml-20m/tags.csv"

tags = pd.read_csv(path)

# Rimuoviamo i tag mancanti
tags = tags.dropna(subset=["tag"])

# Normalizziamo
tags["tag"] = tags["tag"].str.lower()


# Insieme dei tag presenti nel dataset
dataset_tags = set(tags["tag"].unique())


print("Numero di tag nel dataset:", len(dataset_tags))


# ----------------------------------------
# CERCHIAMO COPPIE CON WORDNET
# ----------------------------------------

candidate_pairs = set()


for tag in dataset_tags:

    # Ci interessano per ora solo tag composti da una parola
    if " " in tag:
        continue

    # Cerchiamo i synset WordNet
    synsets = wordnet.synsets(tag)

    for syn in synsets:

        for lemma in syn.lemmas():

            synonym = lemma.name().replace("_", " ").lower()

            # Evitiamo la coppia identica
            if synonym == tag:
                continue

            # Controlliamo se il sinonimo esiste nel dataset
            if synonym in dataset_tags:

                pair = tuple(sorted([tag, synonym]))
                candidate_pairs.add(pair)


# ----------------------------------------
# FREQUENZA DEI TAG
# ----------------------------------------

tag_movie_counts = (
    tags.groupby("tag")["movieId"]
    .nunique()
)


# ----------------------------------------
# ANALISI DELLE COPPIE
# ----------------------------------------

MIN_MOVIES = 20

interesting_pairs = []

for tag_a, tag_b in candidate_pairs:

    movies_a = set(
        tags.loc[tags["tag"] == tag_a, "movieId"]
    )

    movies_b = set(
        tags.loc[tags["tag"] == tag_b, "movieId"]
    )

    count_a = len(movies_a)
    count_b = len(movies_b)

    # Entrambi i termini devono avere una presenza minima
    if count_a < MIN_MOVIES or count_b < MIN_MOVIES:
        continue

    intersection = len(movies_a & movies_b)
    union = len(movies_a | movies_b)

    if union == 0:
        continue

    jaccard = intersection / union

    interesting_pairs.append(
        (
            tag_a,
            tag_b,
            count_a,
            count_b,
            intersection,
            jaccard
        )
    )


# ----------------------------------------
# ORDINIAMO PER JACCARD CRESCENTE
# ----------------------------------------

interesting_pairs.sort(
    key=lambda x: x[5]
)


# ----------------------------------------
# RISULTATI
# ----------------------------------------

print("\nTOP 50 COPPIE CON BASSO OVERLAP:")

for pair in interesting_pairs[:50]:

    print(
        f"{pair[0]} ↔ {pair[1]} | "
        f"{pair[0]}: {pair[2]} | "
        f"{pair[1]}: {pair[3]} | "
        f"insieme: {pair[4]} | "
        f"Jaccard: {pair[5]:.3f}"
    )