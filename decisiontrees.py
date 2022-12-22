#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:56:07 2022

@author: pav2001
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report




#Load dataset
wine = pd.read_csv("winequality-red.csv")

#Train test slit
x = wine.drop('quality',axis=1)
x = StandardScaler().fit_transform(x)
y= wine[['quality']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

x_train.shape


# Create a Decision Tree
dt_basic = DecisionTreeClassifier(max_depth=10)
# Fit the training data
dt_basic.fit(x_train,y_train)
# Predict based on test data
y_preds = dt_basic.predict(x_test)


# Calculate Accuracy
accuracy_value = metrics.accuracy_score(y_test,y_preds)
accuracy_value
# Create and print confusion matrix
confusion_matrix(y_test,y_preds)
print(classification_report(y_test,y_preds))
# Calculate the number of nodes in the tree
dt_basic.tree_.node_count
print(accuracy_value)


