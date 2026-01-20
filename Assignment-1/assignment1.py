

# %%
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_percepton_tagger')

# %%
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from textblob import TextBlob

# %%
text = "I went on a vacation to the mountains with my family. The weather was pleasant and the scenery was beautiful. We enjoyed trekking, local food, and peaceful evenings near the river. The trip made me feel relaxed, happy, and refreshed."

# %%
token = word_tokenize(text)
print("Tokens")
print(token)

# %%
stemmer =PorterStemmer()
stem = [stemmer.stem(word) for word in token]
print("Stemming")
print(stem)


# %%
lemmatizer= WordNetLemmatizer()
lemmas =[lemmatizer.lemmatize(word) for word in token]
print("Lemmatizer")
print(lemmas)

# %%
pos_tags = nltk.pos_tag(token)
print("POS Tagging")
print(pos_tags)

# %%
blob = TextBlob(text)
sentiment = blob.sentiment

print("Sentiment Analysis")
print("Polarity",sentiment.polarity)
print("Subjectivity",sentiment.subjectivity)

if sentiment.polarity > 0:
    print("Overall Positive Sentiment")

elif sentiment.polarity < 0:
    print("Overall Negative Sentiment")

else:
    print("Neutral Sentiment")





