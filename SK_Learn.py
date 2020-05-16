import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression


# 1. Boston dataset
def boston_function():
    dataset = datasets.load_boston()  # load dataset
    reg = LinearRegression()  # create regression object
    reg.fit(dataset.data, dataset.target)  # fit dataset
    result = pd.DataFrame(
    abs(reg.coef_), dataset.feature_names, columns=["coef"]
    )  # take regression coefficients
    result.sort_values(
    by=["coef"], ascending=False, inplace=True
    )  # sort regression coefficients
    return print("The most affecting fator is", result.index[0], "with a value of :", result.coef[0])  # print best coefficient


if __name__ == '__main__':
    boston_function()
