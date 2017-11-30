import pandas as pd
import tqdm

word_embeddings_50d = pd.DataFrame(pd.read_csv('/home/koen/Documents/NaturalLanguageProcessing/Project/glove.6B.50d.txt', sep=" ", header=None, quotechar = "'"))
word_embeddings_100d = pd.DataFrame(pd.read_csv('/home/koen/Documents/NaturalLanguageProcessing/Project/glove.6B.100d.txt', sep=" ", header=None, quotechar = "'"))
word_embeddings_200d = pd.DataFrame(pd.read_csv('/home/koen/Documents/NaturalLanguageProcessing/Project/glove.6B.200d.txt', sep=" ", header=None, quotechar = "'"))
word_embeddings_300d = pd.DataFrame(pd.read_csv('/home/koen/Documents/NaturalLanguageProcessing/Project/glove.6B.300d.txt', sep=" ", header=None, quotechar = "'"))
