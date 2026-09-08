from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet

#Espande la query utilizzando i sinonimi di WordNet.
def expand_query(query):
   
    expanded = set(query.split())

    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))

    return list(expanded)


def create_tfidf_index(documents):
    """
    Crea il vettore TF-IDF dei documenti.

    Restituisce:
    - vectorizer: il TfidfVectorizer
    - document_vectors: i vettori TF-IDF dei documenti
    """
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(documents)

    return vectorizer, document_vectors


def search_baseline(query, vectorizer, document_vectors, documents):
    """
    Ricerca baseline:
    query → TF-IDF → Cosine Similarity → ranking
    """
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranking = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranking


def search_with_expansion(
    query,
    vectorizer,
    document_vectors,
    documents
):
    """
    Ricerca con Query Expansion:
    query → WordNet → TF-IDF → Cosine Similarity → ranking
    """

    expanded_query = expand_query(query)

    expanded_query_text = " ".join(expanded_query)

    expanded_query_vector = vectorizer.transform(
        [expanded_query_text]
    )

    scores = cosine_similarity(
        expanded_query_vector,
        document_vectors
    )[0]

    ranking = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    return expanded_query, ranking