# Extraction-based-Text-Summarization

Text-Summarization is a process of summarizing a given article or text and get the important parts or the summary of the article. 

Extraction based Text Summarization means extracting the sentences from the article itself and not using any new words, it is like extracting the important sentences. 

For extracting the important sentences for the summary I have used **Textrank** technique which is based on **Pagerank** algorithm.

**PageRank** algorithm is used  by Google for ranking the webpages in their results. The Pagerank algorithm outputs a probability distribution used to represent the likelyhood that a person randomly clicking on a wepage. The algorithm uses iteration to approximate's the pagerank values.

**TextRank** algorithm basically uses the sentences instead of the webpages, computes the similarity matrix and computes the similarity in the vector space.

I have also used **GloVe** here for getting the pretrained model and the word embeddings so as to create the vectors of the sentences.

Here what I have done:

- Reading the article and breaking it into sentences.
- Tokenizing the sentences.
- Cleaning the sentences.
- Creating the vectors of the sentences with the GloVe's pretrained word embeddings on wikipedia data.
- Creating the graph of sentences, where the vertices are the sentences and the edges are the similarity scores.
- Computing Similarity matrix and returning the sentences with the top scores for the summary. 

## For Installation


```

'pip3 install requirement.txt'

```

## For getting the GloVe pretrained vectors:

```

wget nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

```
Downloading the GloVe vector is important before anything.

Also one can download the pretrained model by going [here](https://nlp.stanford.edu/projects/glove) and downloading the glove.6B.zip then unzipping it to the data to the same folder.

I have used 100 dimension data so as to get good accuracy and it takes a bit less time, one can use with 50 or 200 dimension data too.

## For usage 

Just copy your article into the test.txt or change the destination to your .txt document.


For running the model:

```
python main.py

```
