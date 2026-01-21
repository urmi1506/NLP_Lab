
# Named Entity Recognition (NER) – spaCy


import spacy
from sklearn.metrics import classification_report, accuracy_score

texts = [
    "Narendra Modi visited New York to attend the UN General Assembly.",
    "Apple launched the iPhone 15 in California on Tuesday.",
    "Virat Kohli scored a century against Australia in Mumbai."
]

true_entities = [
    [("Narendra Modi", "PERSON"), ("New York", "GPE"), ("UN General Assembly", "ORG")],
    [("Apple", "ORG"), ("iPhone 15", "PRODUCT"), ("California", "GPE"), ("Tuesday", "DATE")],
    [("Virat Kohli", "PERSON"), ("Australia", "GPE"), ("Mumbai", "GPE")]
]

nlp = spacy.load("en_core_web_sm")


predicted_entities = []

for text in texts:
    doc = nlp(text)
    predicted_entities.append([(ent.text, ent.label_) for ent in doc.ents])

print("Predicted Entities:")
for p in predicted_entities:
    print(p)


def get_token_labels(text, entities):
    tokens = text.split()
    labels = ["O"] * len(tokens)

    for ent_text, ent_label in entities:
        ent_tokens = ent_text.split()

        for i in range(len(tokens)):
            if tokens[i:i + len(ent_tokens)] == ent_tokens:
                for j in range(len(ent_tokens)):
                    labels[i + j] = ent_label
    return labels


y_true = []
y_pred = []

for i, text in enumerate(texts):
    y_true.extend(get_token_labels(text, true_entities[i]))
    y_pred.extend(get_token_labels(text, predicted_entities[i]))


print("\n=== Evaluation Metrics ===")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))
