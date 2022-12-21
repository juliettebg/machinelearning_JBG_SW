
import numpy as np
import csv
import pandas as pd

training = {}
validation = {}
with open('/Users/juliettebg/Desktop/EPFL/MachineLearning/train.csv') as file:
    data = csv.DictReader(file)
    n = 1
    for cell in data:
        if n <= 4000:
       # print(cell["labels"])
            genes = {k: cell[k] for k in list(cell.keys())[:-1]}
            if len([float(x) for x in genes.values()]) == 32285:
                if cell["labels"] not in training.keys():
                    training[cell["labels"]] = [np.array([float(x) for x in genes.values()])]
                else:
                    training[cell["labels"]] += [np.array([float(x) for x in genes.values()])]
        else:
            genes = {k: cell[k] for k in list(cell.keys())[:-1]}
            if len([float(x) for x in genes.values()]) == 32285:
                if cell["labels"] not in validation.keys():
                    validation[cell["labels"]] = [np.array([float(x) for x in genes.values()])]
                else:
                    validation[cell["labels"]] += [np.array([float(x) for x in genes.values()])]
        n += 1

print("finished reading")

cellmatrix = []
celllabel = []
for key, val in training.items():
    for cell in val :
        cellmatrix += [cell]
        celllabel += [key]
print("made the matrix")

cellmatrix = np.array(cellmatrix)
from sklearn.decomposition import PCA
pca = PCA().fit(cellmatrix)
print("pca done")

print(pca.explained_variance_ratio_)
eigencells = pca.components_
print(eigencells)
weights = eigencells @ (cellmatrix - pca.mean_).T
print(weights)

good = 0
bad = 0
print(good, bad)
for key, val in validation.items():
    for cell in val :
        query = cell
        #print("query = ", query)
        query_weight = eigencells @ (query - pca.mean_).T
        #print("query weight = ", query_weight)
        euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
        print("euclidiean distance = ", euclidean_distance)
       # best_match = np.argmin(euclidean_distance)
        mini = np.sort(euclidean_distance)
        print("mini =", mini)
        kAT5 = 0
        cBP = 0
        eGFP = 0
        for ele in mini[0:500]:
            index = np.where(euclidean_distance == ele)[0][0]
           # print(index)
            neighbour = celllabel[index]
            if neighbour == "KAT5":
                kAT5 += 1
            elif neighbour == "CBP":
                cBP += 1
            elif neighbour == "eGFP":
                eGFP += 1
        if kAT5 > cBP:
            if kAT5 > eGFP:
                best_match = "KAT5"
            else:
                best_match = "eGFP"
        elif cBP > eGFP:
            best_match = "CBP"
        else:
            best_match = "eGFP"


      #  print(best_match)
        #predict = celllabel[best_match]
        predict = best_match
        true = key
        #print(true, predict)
        if true != predict:
            #print (true, predict)
            bad += 1
        else:
            good += 1
        print(good, bad)
misclassification_rate = bad/(good + bad)

print(misclassification_rate)