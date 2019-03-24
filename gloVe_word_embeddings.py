import numpy as np

def get_word_embeddings():

    words_embeddings = dict()
    with open('glove.6B.100d.txt', encoding='utf8') as f:
        for lines in f:
            vals = lines.split()
            word = vals[0]
            coefs = np.asarray(vals[1:], dtype='float32')
            words_embeddings[word] = coefs
    return words_embeddings