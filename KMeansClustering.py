from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics import normalized_mutual_info_score


# Euclidean Distance Caculator
def dist1(a, b, ax=1):
    err = []
    for i in range(len(a)):
        err.append(euclidean(a[i], b[i]))
    return np.array(err)


# Euclidean Distance Caculator
def dist(a, b):
    list1 = []
    for centroid in b:
        list1.append(euclidean(a,centroid))
    return np.array(list1)


data = pd.read_csv('yeastData.csv')

X = np.array(data.iloc[:,:-1].values)
y = np.array(data.iloc[:,-1].values)
# Number of clusters
ks = [i for i in range(1,15)]
sse_list = []
for k in ks:
    #generate random centroid list
    print("Executing for k:",k)
    centroid_list = []
    for i in range(k):

        OneDict = []
        randomIndex = random.randrange(0, len(X))

        for col in range(len(X[0])):
            OneDict.append(X[randomIndex][col])
        centroid_list.append(OneDict)
    centroid_list = np.array(centroid_list)

    # To store the value of centroids when it updates
    C_old = np.zeros(centroid_list.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist1(centroid_list, C_old)
    # Loop will run till the error becomes zero
    while np.mean(error) != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], centroid_list)
            cluster = np.where(distances == distances.min())
            clusters[i] = cluster[0][0]
        # Storing the old centroid values
        C_old = deepcopy(centroid_list)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            if len(points) != 0:
                centroid_list[i] = np.nanmean(points, axis=0)
            else:
                centroid_list[i] = np.zeros(len(X[0]))
        error = dist1(centroid_list, C_old)
    errors = []
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        errors.extend(dist(centroid_list[i], points))
    errors = np.array(errors)
    sse = np.sum(errors)
    sse_list.append(sse)
    clusters = [c + 1 for c in clusters]
    nmi = normalized_mutual_info_score(y,clusters)
    print("nmi: ",nmi)
    print("sse:",sse)
plt.plot(ks,sse_list,'b--')
plt.title('SSE FOR ALL K')
plt.xlabel('K')
plt.ylabel('SSE')
plt.show()