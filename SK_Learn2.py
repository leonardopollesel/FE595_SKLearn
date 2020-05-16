import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Iris dataset
def iris_dataset_function():
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
    return plt.show()


if __name__ == "__main__":
    iris_dataset_function()
