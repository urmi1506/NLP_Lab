from nltk.corpus import wordnet as wn


def get_semantic_relations(word):
    synonyms = set()
    antonyms = set()
    hypernyms = set()

    # Get synsets of word
    for synset in wn.synsets(word):
        
        # Synonyms
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())

            # Antonyms
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())

        # Hypernyms 
        for hyper in synset.hypernyms():
            hypernyms.add(hyper.name())

    return synonyms, antonyms, hypernyms


word = input("Enter a word: ")

synonyms, antonyms, hypernyms = get_semantic_relations(word)

print("\nWord:", word)

print("\nSynonyms:")
print(synonyms if synonyms else "No synonyms found")

print("\nAntonyms:")
print(antonyms if antonyms else "No antonyms found")

print("\nHypernyms:")
print(hypernyms if hypernyms else "No hypernyms found")