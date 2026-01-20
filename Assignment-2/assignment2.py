# # NLP Feature Extraction: BoW, TF-IDF, Word2Vec



documents = [
    "AI is transforming retail and supply chains",
    "Machine learning improves supply chain efficiency",
    "Retail sustainability uses AI and machine learning"
]


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd


count_vectorizer = CountVectorizer()
bow_counts = count_vectorizer.fit_transform(documents)

bow_df = pd.DataFrame(bow_counts.toarray(), columns=count_vectorizer.get_feature_names_out())
bow_df


bow_normalized = bow_counts.toarray() / bow_counts.toarray().sum(axis=1, keepdims=True)
bow_norm_df = pd.DataFrame(bow_normalized, columns=count_vectorizer.get_feature_names_out())
bow_norm_df



tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=5, min_count=1, workers=4)

w2v_model.wv['ai']



