from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
import numpy as np

nltk.download('punkt')

documents = [
    "AI is transforming retail and supply chains",
    "Machine learning improves supply chain efficiency",
    "Retail sustainability uses AI and machine learning"
]

count_vectorizer = CountVectorizer()
bow_counts = count_vectorizer.fit_transform(documents)
bow_df = pd.DataFrame(bow_counts.toarray(), columns=count_vectorizer.get_feature_names_out())

bow_normalized = bow_counts.toarray() / bow_counts.toarray().sum(axis=1, keepdims=True)
bow_norm_df = pd.DataFrame(bow_normalized, columns=count_vectorizer.get_feature_names_out())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=5, min_count=1, workers=4)

w2v_vectors = pd.DataFrame(
    [w2v_model.wv[word] for word in w2v_model.wv.index_to_key],
    index=w2v_model.wv.index_to_key
)

bow_df.to_csv("bow_counts.csv", index=False)
bow_norm_df.to_csv("bow_normalized.csv", index=False)
tfidf_df.to_csv("tfidf_features.csv", index=False)
w2v_vectors.to_csv("word2vec_embeddings.csv")
w2v_model.save("word2vec_model.model")
