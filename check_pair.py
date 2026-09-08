import pandas as pd


# ----------------------------------------
# CARICAMENTO
# ----------------------------------------

tags = pd.read_csv("data/ml-20m/tags.csv")

tags = tags.dropna(subset=["tag"])
tags["tag"] = tags["tag"].str.lower()


# ----------------------------------------
# COPPIA DA ANALIZZARE
# ----------------------------------------

tag_a = "car"
tag_b = "automobile"


# ----------------------------------------
# FILM ASSOCIATI
# ----------------------------------------

movies_a = set(
    tags.loc[tags["tag"] == tag_a, "movieId"]
)

movies_b = set(
    tags.loc[tags["tag"] == tag_b, "movieId"]
)


# ----------------------------------------
# RISULTATI
# ----------------------------------------

print(f"Film con '{tag_a}':", len(movies_a))
print(f"Film con '{tag_b}':", len(movies_b))

print(
    f"Film con entrambi:",
    len(movies_a & movies_b)
)

print(
    f"Film solo '{tag_a}':",
    len(movies_a - movies_b)
)

print(
    f"Film solo '{tag_b}':",
    len(movies_b - movies_a)
)