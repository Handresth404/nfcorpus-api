from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'Datos' / 'raw' / 'nfcorpus'
corpus = pd.read_json(DATA_DIR / 'corpus.jsonl', lines=True, dtype={'_id': str})
corpus['_id'] = corpus['_id'].astype(str)


import re
from collections import Counter, defaultdict
import nltk
import sklearn
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
TOKEN_PATTERN = re.compile('[a-z0-9]+(?:\\.[0-9]+)?')
NEGATIONS = {'no', 'nor', 'not'}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS
stemmer = PorterStemmer(mode=PorterStemmer.NLTK_EXTENSIONS)

def tokenizar(text):
    return TOKEN_PATTERN.findall(str(text).lower())

def preprocesar(text):
    return [stemmer.stem(token) for token in tokenizar(text) if token not in STOP_WORDS]

documents = corpus.copy()
title = documents['title'].fillna('').astype(str).str.strip()
text = documents['text'].fillna('').astype(str).str.strip()
documents['content'] = (title + ' ' + text).str.replace('\\s+', ' ', regex=True).str.strip()
documents['tokens'] = documents['content'].map(preprocesar)
documents['token_count'] = documents['tokens'].str.len()
assert documents['_id'].tolist() == corpus['_id'].tolist()
assert len(documents) == len(corpus)
assert documents['token_count'].sum() > 0

inverted_index = defaultdict(dict)
document_counts = {}
document_lengths = {}
for document_id, tokens in zip(documents['_id'], documents['tokens'], strict=True):
    counts = Counter(tokens)
    document_counts[document_id] = counts
    document_lengths[document_id] = len(tokens)
    for term, frequency in counts.items():
        inverted_index[term][document_id] = frequency
inverted_index = dict(inverted_index)
vocabulary = sorted(inverted_index)
term_to_column = {term: column for column, term in enumerate(vocabulary)}
document_frequency = {term: len(postings) for term, postings in inverted_index.items()}
N = len(documents)
avgdl = sum(document_lengths.values()) / N

import numpy as np
from scipy.sparse import csr_matrix, diags
ordered_document_ids = documents['_id'].tolist()
document_to_row = {doc_id: row for row, doc_id in enumerate(ordered_document_ids)}
idf_vector = np.array([1 + np.log((N + 1) / (document_frequency[term] + 1)) for term in vocabulary], dtype=np.float64)
rows, columns, values = ([], [], [])
for term, postings in inverted_index.items():
    column = term_to_column[term]
    for doc_id, frequency in postings.items():
        rows.append(document_to_row[doc_id])
        columns.append(column)
        values.append(frequency * idf_vector[column])
weighted_matrix = csr_matrix((values, (rows, columns)), shape=(N, len(vocabulary)), dtype=np.float64)
document_norms = np.sqrt(weighted_matrix.multiply(weighted_matrix).sum(axis=1)).A1
inverse_norms = np.divide(1.0, document_norms, out=np.zeros_like(document_norms), where=document_norms > 0)
tfidf_matrix = (diags(inverse_norms) @ weighted_matrix).tocsr()
assert tfidf_matrix.shape == (len(documents), len(vocabulary))
assert np.isfinite(tfidf_matrix.data).all()
assert np.allclose(np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1[document_norms > 0], 1)

def vectorizar_consulta(query):
    if not isinstance(query, str):
        raise TypeError('query debe ser texto')
    counts = Counter(preprocesar(query))
    cols, vals = ([], [])
    for term, frequency in counts.items():
        if term in term_to_column:
            column = term_to_column[term]
            cols.append(column)
            vals.append(frequency * idf_vector[column])
    vector = csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, len(vocabulary)), dtype=np.float64)
    norm = np.sqrt(vector.multiply(vector).sum())
    return vector / norm if norm > 0 else vector

def scores_tfidf(query):
    vector = vectorizar_consulta(query)
    return (tfidf_matrix @ vector.T).toarray().ravel()

def buscar_tfidf(query, top_k=10):
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError('top_k debe ser un entero positivo')
    scores = scores_tfidf(query)
    valid = np.flatnonzero(scores > 0)
    ids = np.asarray(ordered_document_ids)
    order = np.lexsort((ids[valid], -scores[valid]))
    selected = valid[order[:top_k]]
    results = []
    for rank, row in enumerate(selected, 1):
        document = documents.iloc[int(row)]
        results.append({'rank': rank, 'document_id': str(document['_id']), 'score': float(scores[row]), 'title': str(document['title']), 'snippet': str(document['content'])[:240]})
    return {'model': 'tfidf', 'results': results}

K1 = 1.2
B = 0.75
bm25_idf = {term: np.log(1 + (N - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()}

def buscar_bm25(query, top_k=10):
    if not isinstance(query, str):
        raise TypeError('query debe ser texto')
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError('top_k debe ser un entero positivo')
    scores = defaultdict(float)
    for term in set(preprocesar(query)):
        for doc_id, frequency in inverted_index.get(term, {}).items():
            norm = 1 - B + B * document_lengths[doc_id] / avgdl
            scores[doc_id] += bm25_idf[term] * frequency * (K1 + 1) / (frequency + K1 * norm)
    ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    results = []
    for rank, (doc_id, score) in enumerate(ranking, 1):
        document = documents.iloc[document_to_row[doc_id]]
        results.append({'rank': rank, 'document_id': doc_id, 'score': float(score), 'title': str(document['title']), 'snippet': str(document['content'])[:240]})
    return {'model': 'bm25', 'results': results}
