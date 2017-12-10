import pandas as pd

def readWordEmbedding(file):
    embeddings = pd.DataFrame(pd.read_csv(file, sep=" ", header=None, quotechar = "'"))
    return embeddings

# Example usage
# word_embeddings_50d = readWordEmbedding('/home/koen/Documents/NaturalLanguageProcessing/Project/glove.6B.50d.txt')
