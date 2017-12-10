import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict

def data_extraction(file, p):

    tmp = []

    with open(file, 'r', newline='\n') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_ALL)
        for row in tqdm(reader):
            time.sleep(0.00000000001)
            if len(row) != 0:
                if row[0].split(" ")[0] != "#":
                    del row[2]
                    del row[4]
                    del row[-1]
                    del row[-1]
                    tmp.append(row)

    # Make it a pandas dataframe
    tmp_data = pd.DataFrame(tmp, columns = ['Number', 'Word', "POS", "POS2", "NUMBER-ARC", "LABEL-ARC"])
    tmp_data.reset_index(inplace=True)

    tmp_data["Index"] = tmp_data.index

    # replace values that occur once
    c = tmp_data.sort_index().groupby('Word').filter(lambda group: len(group) == 1)
    index_occur_once = np.array(c.iloc[:,0])
    for i in range(0,len(index_occur_once)):
        tmp_data.loc[index_occur_once[i],"Word"] = "<unk>"

    if p == True:
        print(len(c)/len(tmp_data.index)*100, "% of data is '<unk>'.")
        print(tmp_data)

    # Create dictionaries for indexing

    w2i = tmp_data.groupby("Word")[["Index"]].apply(lambda x: x["Index"].to_dict())
    i2w = tmp_data.set_index('Index')['Word'].to_dict()

    t2i = tmp_data.groupby("POS")[["Index"]].apply(lambda x: x["Index"].to_dict())
    i2t = tmp_data.set_index('Index')['POS'].to_dict()

    l2i = tmp_data.groupby("LABEL-ARC")[["Index"]].apply(lambda x: x["Index"].to_dict())
    i2l = tmp_data.set_index('Index')['LABEL-ARC'].to_dict()

    return tmp_data, w2i, i2w, t2i, i2t, l2i, i2l

# Example usage
# a, b, c, d, e, f, g = data_extraction("/home/koen/Documents/NaturalLanguageProcessing/Project/NLP1/en-ud-test.conllu", False)
