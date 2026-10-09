# Superare il Vocabulary Mismatch nella Social Search tramite Query Expansion e Conoscenza Esterna

## Descrizione

Il progetto studia il problema del **Vocabulary Mismatch** nella ricerca di contenuti taggati, con particolare riferimento alla Social Search basata su folksonomie.

Il problema nasce quando la query dell'utente e i tag presenti nei documenti utilizzano termini differenti, anche se semanticamente correlati. In questo caso, una ricerca basata esclusivamente sulla corrispondenza dei termini può non recuperare documenti rilevanti.

L'obiettivo del progetto è verificare sperimentalmente se la **Query Expansion**, utilizzando conoscenza esterna fornita da **WordNet**, possa migliorare il recupero dei documenti rilevanti.

Vengono confrontati due approcci:

1. **Baseline**: query → TF-IDF → Cosine Similarity → ranking
2. **Query Expansion**: query → WordNet → TF-IDF → Cosine Similarity → ranking

---

## Obiettivo

L'obiettivo principale è verificare se l'espansione della query permetta di recuperare documenti che utilizzano termini differenti da quello presente nella query originale.

Ad esempio, una query contenente:

```text
remade
```

può essere espansa includendo:

```text
remake
```

permettendo di recuperare documenti che contengono il termine `remake` ma non `remade`.

L'ipotesi sperimentale è quindi che la Query Expansion possa ridurre gli effetti del Vocabulary Mismatch.

---

## Dataset

Per gli esperimenti è stato utilizzato il dataset **MovieLens 20M**, in particolare il file `tags.csv`.

Il dataset contiene associazioni tra utenti, film e tag.

Dopo la rimozione delle righe prive di tag:

* Righe iniziali: **465.564**
* Righe dopo la pulizia: **465.548**
* Numero di film/documenti: **19.545**
* Numero di tag distinti analizzati: **35.172**

Per costruire i documenti utilizzati dal motore di ricerca, tutti i tag associati allo stesso film vengono concatenati in un unico documento testuale.

Esempio:

```text
movieId: 1

document:
watched computer animation disney animated features
```

In questo modo ogni film viene rappresentato come un documento composto dai suoi tag.

---

## Architettura del sistema

Il sistema è composto da due modalità di ricerca.

### Baseline

```text
Query
  ↓
TF-IDF
  ↓
Cosine Similarity
  ↓
Ranking dei documenti
```

La query viene trasformata nello stesso spazio vettoriale dei documenti mediante TF-IDF.

Successivamente viene calcolata la **Cosine Similarity** tra il vettore della query e ogni documento.

---

### Query Expansion

```text
Query
  ↓
WordNet
  ↓
Query espansa
  ↓
TF-IDF
  ↓
Cosine Similarity
  ↓
Ranking dei documenti
```

In questo caso la query viene prima espansa utilizzando i sinonimi e i termini correlati forniti da WordNet.

La query originale e i termini ottenuti dall'espansione vengono quindi utilizzati per costruire il vettore TF-IDF della query espansa.

---

## Query Expansion con WordNet

La Query Expansion è implementata utilizzando **WordNet** tramite la libreria NLTK.

Per ogni termine della query vengono ricercati i synset disponibili e vengono estratti i lemma associati.

Ad esempio, per alcune query WordNet può produrre termini come:

```text
humour → humor, wit, wittiness, sense of humor, ...
```

oppure:

```text
nonsensical → ridiculous, absurd, idiotic, ludicrous, ...
```

L'espansione viene poi trasformata in un'unica stringa e processata dal vettorizzatore TF-IDF.

---

## Vocabulary Mismatch

Per studiare il problema in modo controllato sono state selezionate coppie di termini semanticamente correlati ma caratterizzate da un basso overlap nei film associati.

Alcuni esempi utilizzati negli esperimenti sono:

| Query         | Target     |
| ------------- | ---------- |
| humour        | humor      |
| television    | tv         |
| nonsensical   | ridiculous |
| drab          | gloomy     |
| remade        | remake     |
| teenager      | teenagers  |
| united states | usa        |
| history       | story      |
| funny         | humour     |
| absurd        | ridiculous |

Per ogni coppia, il **gold set** è costituito dai film associati al termine target e non anche al termine della query.

In questo modo vengono valutati specificamente i documenti che rappresentano il Vocabulary Mismatch.

---

## Metriche

Per confrontare Baseline e Query Expansion sono state utilizzate:

* **Precision@10**
* **Recall@10**
* **Precision@50**
* **Recall@50**

La precision misura la proporzione di documenti rilevanti tra quelli recuperati nei primi K risultati.

La recall misura invece la proporzione dei documenti rilevanti presenti nel gold set che vengono recuperati nei primi K risultati.

---

## Risultati

Le metriche riportate sono le medie ottenute sulle 10 coppie query-target.

| Metrica      | Baseline | Query Expansion |
| ------------ | -------: | --------------: |
| Precision@10 |    0.020 |       **0.130** |
| Recall@10    |   0.0027 |      **0.0094** |
| Precision@50 |    0.012 |       **0.178** |
| Recall@50    |   0.0067 |      **0.0915** |

I risultati mostrano un miglioramento complessivo della Query Expansion rispetto alla Baseline.

Il miglioramento è particolarmente evidente per **Precision@50**, che passa da:

```text
0.012 → 0.178
```

e per **Recall@50**, che passa da:

```text
0.0067 → 0.0915
```

Il comportamento, tuttavia, non è uniforme per tutte le query.

---

## Analisi degli errori

Un caso particolarmente significativo è:

```text
remade → remake
```

Per questa coppia:

```text
P@10: 0.000 → 0.800
P@50: 0.000 → 0.760
```

L'espansione permette quindi di introdurre il termine `remake`, consentendo al sistema di recuperare documenti che utilizzano un termine differente dalla query originale.

Sono presenti anche casi in cui l'espansione non produce miglioramenti o può peggiorare il ranking.

Un esempio è:

```text
united states → usa
```

In questo caso l'espansione di WordNet può introdurre termini più generici o semanticamente ambigui, aumentando il rumore nella rappresentazione della query.

Questo evidenzia una limitazione dell'approccio: **l'espansione automatica non garantisce che tutti i termini aggiunti siano utili per il problema di Information Retrieval considerato.**

---

## Struttura del progetto

```text
SII-Vocabulary-Mismatch/
│
├── data/
│   └── ml-20m/
│
├── search.py
├── explore_data.py
├── find_query_pairs.py
├── check_pair.py
├── evaluate_search.py
│
├── results/
│   └── comparison_metrics.png
│
├── requirements.txt
└── README.md
```

### Descrizione degli script

**`explore_data.py`**

Effettua l'analisi iniziale del dataset e costruisce i documenti a partire dai tag associati ai film.

**`find_query_pairs.py`**

Analizza le possibili coppie di tag semanticamente correlate e caratterizzate da un basso overlap.

**`check_pair.py`**

Script di supporto utilizzato durante l'analisi per verificare singole coppie di termini e il relativo overlap nei documenti.

**`search.py`**

Implementa il motore di ricerca, la costruzione dell'indice TF-IDF, la ricerca baseline e la Query Expansion.

**`evaluate_search.py`**

Esegue gli esperimenti sulle coppie query-target e calcola Precision e Recall a diversi valori di K.

---

## Installazione

È consigliato utilizzare un ambiente virtuale Python.

Creazione dell'ambiente:

```bash
python -m venv .venv
```

Attivazione su Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Installazione delle dipendenze:

```bash
pip install -r requirements.txt
```

---

## Esecuzione

Per analizzare il dataset:

```bash
python explore_data.py
```

Per eseguire gli esperimenti:

```bash
python evaluate_search.py
```

Il programma stampa i risultati della Baseline e della Query Expansion e calcola le metriche Precision@K e Recall@K.

Il grafico comparativo delle metriche viene salvato nella cartella:

```text
results/
```

---

## Librerie utilizzate

Il progetto è stato sviluppato in Python utilizzando:

* **Pandas** per la manipolazione e analisi del dataset
* **scikit-learn** per TF-IDF e Cosine Similarity
* **NLTK** per l'accesso a WordNet e la Query Expansion
* **Matplotlib** per la visualizzazione dei risultati

Le dipendenze sono riportate nel file `requirements.txt`.

---

## Riferimenti

* MovieLens 20M Dataset
* WordNet
* NLTK
* scikit-learn

Ulteriori riferimenti bibliografici e web utilizzati nello sviluppo saranno riportati nel report finale.