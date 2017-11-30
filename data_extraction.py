from conllu.parser import parse, parse_tree
import re
import csv
import numpy as np
import pandas as pd

# Development data
print("READING IN DEV DATA - START")

dev_dat = []

with open('/home/koen/Documents/NaturalLanguageProcessing/Project/NLP1/en-ud-dev.conllu', 'r', newline='\n') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) != 0:
            if row[0].split(" ")[0] != "#":
                del row[2]
                del row[4]
                del row[4]
                del row[4]
                del row[-1]
                dev_dat.append(row)

# Make it a pandas dataframe
dev_data = pd.DataFrame(dev_dat, columns = ['Number', 'Word', "POS", "LABEL", "ARC"])
dev_data.reset_index(inplace=True)

# replace values that occur once
c = dev_data.sort_index().groupby('Word').filter(lambda group: len(group) == 1)
index_occur_once = np.array(c.iloc[:,0])
for i in range(0,len(index_occur_once)):
    dev_data.loc[index_occur_once[i],"Word"] = "<unk>"

print("READING IN DEV DATA - DONE")
# print(dev_data)

# Test data
print("READING IN TEST DATA - START")

test_dat = []

with open('/home/koen/Documents/NaturalLanguageProcessing/Project/NLP1/en-ud-test.conllu', 'r', newline='\n') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) != 0:
            if row[0].split(" ")[0] != "#":
                del row[2]
                del row[4]
                del row[4]
                del row[4]
                del row[-1]
                test_dat.append(row)

# Make it a pandas dataframe
test_data = pd.DataFrame(test_dat, columns = ['Number', 'Word', "POS", "LABEL", "ARC"])
test_data.reset_index(inplace=True)

# replace values that occur once
c = test_data.sort_index().groupby('Word').filter(lambda group: len(group) == 1)
index_occur_once = np.array(c.iloc[:,0])
for i in range(0,len(index_occur_once)):
    test_data.loc[index_occur_once[i],"Word"] = "<unk>"

print("READING IN TEST DATA - DONE")
# print(test_data)

# Train data
print("READING IN TRAIN DATA - START")

train_dat = []

with open('/home/koen/Documents/NaturalLanguageProcessing/Project/NLP1/en-ud-train.conllu', 'r', newline='\n') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) != 0:
            if row[0].split(" ")[0] != "#":
                if len(row) == 10:
                    del row[2]
                    del row[4]
                    del row[4]
                    del row[4]
                    del row[-1]
                    train_dat.append(row)

# Make it a pandas dataframe
train_data = pd.DataFrame(train_dat, columns = ['Number', 'Word', "POS", "LABEL", "ARC"])
train_data.reset_index(inplace=True)

# replace values that occur once
c = train_data.sort_index().groupby('Word').filter(lambda group: len(group) == 1)
index_occur_once = np.array(c.iloc[:,0])
for i in range(0,len(index_occur_once)):
    train_data.loc[index_occur_once[i],"Word"] = "<unk>"

print("READING IN TRAIN DATA - DONE")
# print(train_data)
