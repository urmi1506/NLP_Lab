import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

data = {
    "text": [
        "AI is transforming Retail Industry!",
        "Machine learning improves supply chain efficiency.",
        "Retail sustainability uses AI and ML."
    ],
    "label": ["tech", "tech", "sustainability"]
}

df = pd.DataFrame(data)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["clean_text"])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

df.to_csv("cleaned_and_encoded_data.csv", index=False)
tfidf_df.to_csv("tfidf_features.csv", index=False)

label_mapping = pd.DataFrame({
    "label": label_encoder.classes_,
    "encoded_value": range(len(label_encoder.classes_))
})
label_mapping.to_csv("label_mapping.csv", index=False)
