import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Boston dataset
dataset = datasets.load_boston()  # load dataset
reg = LinearRegression()  # create regression object
reg.fit(dataset.data, dataset.target)  # fit dataset
result = pd.DataFrame(
    abs(reg.coef_), dataset.feature_names, columns=["coef"]
)  # take regression coefficients
result.sort_values(
    by=["coef"], ascending=False, inplace=True
)  # sort regression coefficients
print(
    "The most affecting fator is", result.index[0], "with a value of :", result.coef[0]
)  # print best coefficient


# Iris dataset
iris = datasets.load_iris()  # load iris dataset
iris_dataframe = pd.DataFrame(
    iris.data, columns=iris.feature_names
)  # convert data of iris as dataframe

colors = np.array(
    ["blue", "green", "red"]
)  # set colors for target (there are 3 targets so 3 colors)
plt.scatter(
    iris_dataframe["sepal length (cm)"],
    iris_dataframe["sepal width (cm)"],
    c=colors[iris.target],
)
plt.title("sepal sength \/ sepal width")
plt.show()
plt.scatter(
    iris_dataframe["petal length (cm)"],
    iris_dataframe["petal width (cm)"],
    c=colors[iris.target],
)
plt.title("petal length \/ petal width")
plt.show()

# find optimal number of clusters using Silhouette method
def optimal_k(maxk):
    sil = []
    for k in range(2, maxk + 1):
        kmeans = KMeans(n_clusters=k).fit(iris.data)
        sil.append(silhouette_score(iris.data, kmeans.labels_, metric="euclidean"))
    return sil


ksim = optimal_k(10)
plt.plot(ksim)
plt.show()

# fit with k = 3

k_best = KMeans(n_clusters=3).fit(iris.data)

plt.scatter(
    iris_dataframe["sepal length (cm)"],
    iris_dataframe["sepal width (cm)"],
    c=colors[k_best.labels_],
)
plt.title("Kmeans = 3")
plt.show()
