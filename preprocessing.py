from nltk.corpus import stopwords
import pandas as pd

def clean_sentence(sent):

    # Converting all the sentences to lowercase
    sent = [s.lower() for s in sent]

    # Removing the punctuations, numbers and any special characters
    sent = pd.Series(sent).str.replace("[^a-zA-Z]", " ")

    # using stopwords from nltk corpus
    stop_words = stopwords.words('english')

    cleaned_sentence = list()

    for i in sent:
        k = i.split()
        sentence = " ".join([j for j in k if j not in stop_words])
        cleaned_sentence.append(sentence)

    return cleaned_sentence