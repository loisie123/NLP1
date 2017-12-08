import pandas as pd
import tqdm

word_embeddings_300d_NL = pd.DataFrame(pd.read_csv('/home/koen/Documents/NaturalLanguageProcessing/Project/nl.txt', sep=" ", header=None, quotechar = "'"))
del word_embeddings_300d_NL[301]
