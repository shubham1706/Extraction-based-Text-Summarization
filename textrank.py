from gloVe_word_embeddings import get_word_embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

words_embeddings = get_word_embeddings()

def create_similarity_matrix(sentences, words_embeddings):

    # similarity matrix 
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    sent_vectors = list()

    # Creating vectors for our sentences. Fetching the vectors and then taking the average.
    for i in sentences:
        if len(i) == 0:
            vec = np.zeros((100,))
        else:
            vec = sum([words_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)

        sent_vectors.append(vec)

    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            else:
                similarity_matrix[i][j] = cosine_similarity(sent_vectors[i].reshape(1,100), sent_vectors[j].reshape(1,100))

    return similarity_matrix

