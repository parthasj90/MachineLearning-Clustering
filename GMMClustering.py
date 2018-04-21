from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import multivariate_normal


def dist(a, b):
    list1 = []
    for centroid in b:
        list1.append(euclidean(a,centroid))
    return np.array(list1)


def calculate_n(x,mean,cov):
    rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
    return rv.pdf(x)

def calculate_variance(gamma,X,mean):
    var = []
    for k in range(len(gamma[0])):
        sum = 0
        for n in range(len(X)):
            sub = np.subtract(X[n],mean[k])
            test = np.outer(sub,sub)
            sum = sum + gamma[n][k] * test
        sum = sum / np.sum(gamma,axis=0)[k]
        var.append(sum)
    return var


data = pd.read_csv('dermatologyData.csv')

X = np.array(data.iloc[:,:-1].values)
y = np.array(data.iloc[:,-1].values)
# Number of clusters
ks = [i for i in range(1,11)]
sse_list = []
nmi_list = []
for k in ks:
    print("Executing for k:",k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    label_list, frequency = np.unique(labels, return_counts=True)
    #loop starts
    change = 5
    dll = 0

    var = []
    for inx in range(k):
        sum = np.zeros((len(X[0]),len(X[0])))
        for n in range(len(X)):
            if labels[n] == inx:
                sub = np.subtract(X[n], centroids[inx])
                test = np.outer(sub, sub)
                sum = sum + test
        var.append(sum)

    iterations = 1
    while change >= 0.0001 and iterations <= 100:
        print("running iteration:", iterations)
        iterations += 1
        pie = np.array([f / len(X) for f in frequency])
        gamma = []
        dll_old = dll
        dll = 0
        for i in range(len(X)):
            gamma_num = []
            for j in range(k):
                gamma_num.append(pie[j] * calculate_n(X[i],centroids[j],var[j]))
            sum = np.sum(gamma_num)
            dll += np.log(sum)
            gamma_num = [num/sum for num in gamma_num]
            gamma.append(gamma_num)
        gamma = np.array(gamma)
        change = abs(dll - dll_old)
        frequency = [np.sum(gamma[:, j]) for j in range(k)]

        for j in range(k):
            temp = [X[i] * gamma[i][j] for i in range(len(X))]
            centroids[j] = np.sum(temp,axis=0)/frequency[j]
        var = calculate_variance(gamma,X,centroids)

    predictions = [np.where(gamma[i] == gamma[i].max())[0][0] for i in range(len(gamma))]
    errors = []
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if predictions[j] == i])
        errors.extend(dist(centroids[i], points))
    errors = np.array(errors)
    sse = np.sum(errors)
    sse_list.append(sse)
    print("sse: ",sse)
    predictions = [c + 1 for c in predictions]
    nmi = normalized_mutual_info_score(y, predictions)
    print("nmi: ", nmi)
    nmi_list.append(nmi)
plt.plot(ks, sse_list, 'b--')
plt.title('SSE FOR ALL K DERMATOLOGY DATA')
plt.xlabel('K')
plt.ylabel('SSE')
plt.show()
plt.plot(ks,nmi_list,'r--')
plt.title('NMI FOR ALL K DERMATOLOGY DATA')
plt.xlabel('K')
plt.ylabel('NMI')
plt.show()