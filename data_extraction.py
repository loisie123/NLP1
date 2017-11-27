from conllu.parser import parse, parse_tree
import re

data = []

with open('en-ud-dev.txt', 'r') as myfile:
    data=myfile.readlines()

hoi = []

for row in data:
    if not (not row):
        print(row)
        if not (row.split()[0] == "#"):
            hoi.append(row)

print(hoi)

data = re.sub(r" +", r"\t", data)
# print(data)

results = parse(data)

print("results =")
# print(results)
