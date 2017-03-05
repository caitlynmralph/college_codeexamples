'''
Caitlyn Ralph, 2017 ISP: Data Journalism
Analysis Paper Code on _ Dataset
Code help from "Introduction to Machine Learning in Python: A Guide for Data Scientists"
by A. C. Miller and S. Guido.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

#import CSV file
dataset = {}
dataset["feature_names"] = ["Race/Ethnicity","2005","2006","2007","2008","2009","2010","2011","2012","2013"]
dataset["data"] = np.zeros((5,10))
dataset["target"] = np.zeros((5,1))

with open('data.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    count_row = 0
    for row in spamreader:
        count_column = 0
        for value in row:
            if count_column == 10:
                dataset["target"][count_row:count_row+1,:] = value
            else:
                dataset["data"][count_row:count_row+1,count_column:count_column+1] = value
            count_column+=1
        count_row+=1

print(dataset)

#what does the data look like
print("Keys of dataset: \n{}".format(dataset.keys()))
print("Feature names: \n{}".format(dataset['feature_names']))
print("Shape of data: \n{}".format(dataset['data'].shape))

#split data into training set and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset['data'],dataset['target'],random_state=0)

#visualize data in pair plot using pandas
dataframe = pd.DataFrame(X_train, columns=dataset["feature_names"])
grr = pd.scatter_matrix(dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60,alpha=.8)

#linear regression to assess over- or underfitting
print("Linear Regression")
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}",format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test data score: {:.2f}".format(lr.score(X_test, y_test)))

#ridge regression
print("Ridge Regression")
from sklearn.linear_model import Ridge

#can change alpha value like so Ridge.(alpha=0.1)
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train,y_train)))
print("Test data score: {:.2f}".format(ridge.score(X_test, y_test)))

