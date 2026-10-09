import pandas as pd


# ----------------------------------------
# CARICAMENTO DATASET
# ----------------------------------------

path = "data/ml-20m/tags.csv"

tags = pd.read_csv(path)


# ----------------------------------------
# PULIZIA
# ----------------------------------------

print("Righe iniziali:", len(tags))

# Eliminazione righe senza tag
tags = tags.dropna(subset=["tag"])

print("Righe dopo rimozione tag mancanti:", len(tags))


# Conversione tag in minuscolo
tags["tag"] = tags["tag"].str.lower()


# ----------------------------------------
# COSTRUZIONE DEI DOCUMENTI
# ----------------------------------------

documents = (
    tags
    .groupby("movieId")["tag"]
    .apply(" ".join)
    .reset_index()
)


# Rinomina della colonna dei tag
documents = documents.rename(
    columns={"tag": "document"}
)


# ----------------------------------------
# INFORMAZIONI
# ----------------------------------------

print("\nNumero di documenti:", len(documents))

print("\nPrime 10 righe:")
print(documents.head(10))

# ----------------------------------------
# ANALISI DEI TAG
# ----------------------------------------

print("\nTAG PIÙ FREQUENTI:")

tag_counts = tags["tag"].value_counts()

print(tag_counts.head(20))


# ----------------------------------------
# NUMERO DI FILM PER TAG
# ----------------------------------------

movies_per_tag = tags.groupby("tag")["movieId"].nunique()

print("\nTAG PRESENTI IN PIÙ FILM:")
print(movies_per_tag.sort_values(ascending=False).head(20))


# ----------------------------------------
# NUMERO DI TAG PER FILM
# ----------------------------------------

tags_per_movie = tags.groupby("movieId")["tag"].count()

print("\nSTATISTICHE TAG PER FILM:")
print(tags_per_movie.describe())

# ----------------------------------------
# SALVATAGGIO CORPUS
# ----------------------------------------

documents.to_csv(
    "data/movie_documents.csv",
    index=False
)

print("\nCorpus salvato in:")
print("data/movie_documents.csv")