import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict

######################################################################################################################################

# Development data
print("DEV DATA")

dev_dat = []

with open('/home/koen/Documents/NaturalLanguageProcessing/Project/NLP1/en-ud-dev.conllu', 'r', newline='\n') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_ALL)
    for row in tqdm(reader):
        time.sleep(0.00000000001)
        if len(row) != 0:
            if row[0].split(" ")[0] != "#":
                del row[2]
                del row[4]
                del row[-1]
                del row[-1]
                dev_dat.append(row)

# Make it a pandas dataframe
dev_data = pd.DataFrame(dev_dat, columns = ['Number', 'Word', "POS", "POS2", "NUMBER-ARC", "LABEL-ARC"])
dev_data.reset_index(inplace=True)

dev_data["Index"] = dev_data.index

# replace values that occur once
c = dev_data.sort_index().groupby('Word').filter(lambda group: len(group) == 1)
index_occur_once = np.array(c.iloc[:,0])
for i in range(0,len(index_occur_once)):
    dev_data.loc[index_occur_once[i],"Word"] = "<unk>"

print(len(c)/len(dev_data.index)*100, "% of dev data is '<unk>'.")
# print(dev_data)

# Create dictionaries for indexing
# w2i_dev = dev_data.set_index('Word')['Index'].to_dict()
w2i_dev = dev_data.groupby("Word")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2w_dev = dev_data.set_index('Index')['Word'].to_dict()
# t2i_dev = dev_data.set_index('POS')['index'].to_dict()
t2i_dev = dev_data.groupby("POS")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2t_dev = dev_data.set_index('Index')['POS'].to_dict()
# l2i_dev = dev_data.set_index('LABEL-ARC')['index'].to_dict()
l2i_dev = dev_data.groupby("LABEL-ARC")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2l_dev = dev_data.set_index('Index')['LABEL-ARC'].to_dict()

######################################################################################################################################

# Test data
print("TEST DATA")

test_dat = []

with open('/home/koen/Documents/NaturalLanguageProcessing/Project/NLP1/en-ud-test.conllu', 'r', newline='\n') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_ALL)
    for row in tqdm(reader):
        time.sleep(1e-10)
        if len(row) != 0:
            if row[0].split(" ")[0] != "#":
                del row[2]
                del row[4]
                del row[-1]
                del row[-1]
                test_dat.append(row)

# Make it a pandas dataframe
test_data = pd.DataFrame(test_dat, columns = ['Number', 'Word', "POS", "POS2", "NUMBER-ARC", "LABEL-ARC"])
test_data.reset_index(inplace=True)

test_data["Index"] = test_data.index

# replace values that occur once
c = test_data.sort_index().groupby('Word').filter(lambda group: len(group) == 1)
index_occur_once = np.array(c.iloc[:,0])
for i in range(0,len(index_occur_once)):
    test_data.loc[index_occur_once[i],"Word"] = "<unk>"

print(len(c)/len(test_data.index)*100, "% of test data is '<unk>'.")
# print(test_data)

# Create dictionaries for indexing
# w2i_test = test_data.set_index('Word')['index'].to_dict()
w2i_test = test_data.groupby("Word")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2w_test = test_data.set_index('Index')['Word'].to_dict()
# t2i_test = test_data.set_index('POS')['index'].to_dict()
t2i_test = test_data.groupby("POS")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2t_test = test_data.set_index('Index')['POS'].to_dict()
# l2i_test = test_data.set_index('LABEL-ARC')['index'].to_dict()
l2i_test = test_data.groupby("LABEL-ARC")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2l_test = test_data.set_index('Index')['LABEL-ARC'].to_dict()

######################################################################################################################################

# Train data
print("TRAIN DATA")

train_dat = []

with open('/home/koen/Documents/NaturalLanguageProcessing/Project/NLP1/en-ud-train.conllu', 'r', newline='\n') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_ALL)
    for row in tqdm(reader):
        time.sleep(1e-10)
        if len(row) != 0:
            if row[0].split(" ")[0] != "#":
                if len(row) == 10:
                    del row[2]
                    del row[4]
                    del row[-1]
                    del row[-1]
                    train_dat.append(row)

# Make it a pandas dataframe
train_data = pd.DataFrame(train_dat, columns = ['Number', 'Word', "POS", "POS2", "NUMBER-ARC", "LABEL-ARC"])
train_data.reset_index(inplace=True)

train_data["Index"] = train_data.index

# replace values that occur once
c = train_data.sort_index().groupby('Word').filter(lambda group: len(group) == 1)
index_occur_once = np.array(c.iloc[:,0])
for i in range(0,len(index_occur_once)):
    train_data.loc[index_occur_once[i],"Word"] = "<unk>"

print(len(c)/len(train_data.index)*100, "% of train data is '<unk>'.")
# print(train_data)

# Create dictionaries for indexing
# w2i_train = train_data.set_index('Word')['index'].to_dict()
w2i_train = train_data.groupby("Word")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2w_train = train_data.set_index('Index')['Word'].to_dict()
# t2i_train = train_data.set_index('POS')['index'].to_dict()
t2i_train = train_data.groupby("POS")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2t_train = train_data.set_index('Index')['POS'].to_dict()
# l2i_train = train_data.set_index('LABEL-ARC')['index'].to_dict()
l2i_train = train_data.groupby("LABEL-ARC")[["Index"]].apply(lambda x: x["Index"].to_dict())
i2l_train = train_data.set_index('Index')['LABEL-ARC'].to_dict()

######################################################################################################################################
