import nltk
from preprocessing import  clean_sentence
from gloVe_word_embeddings import get_word_embeddings
from textrank import create_similarity_matrix
import networkx as nx


# Loading the text from the txt file.
with open('test.txt', 'r') as f:
    text = f.read()

# tokenizing the sentences 
sents = nltk.sent_tokenize(text)

# cleaning or preprocessing the data
clean_sent = clean_sentence(sents)

# Making a word embedding
word_embedding = get_word_embeddings()

# creating vectors and similarity matrix

similarity_matrix = create_similarity_matrix(clean_sent, word_embedding)

# Making a graph of the similarity matrix where vertices are the sentences and the edges are the similarity values
graph = nx.from_numpy_array(similarity_matrix)

# Calculating scores of the sentence using Pagerank algorithm
score = nx.pagerank(graph)

# sorting the scores 
sentences = sorted(((score[i],s) for i,s in enumerate(sents)), reverse=True)

# Printing first 10 lines as the summary
for i in range(10):
  print(sentences[i][1])
