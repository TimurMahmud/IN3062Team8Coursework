# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:24:23 2022

@author: timur
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# read in the wine quality data from a CSV file
df = pd.read_csv("winequality-red.csv")

# create feature and target variables
X = df.drop("quality",axis=1) # features are all columns except "quality"
y = df["quality"] # target is the "quality" column

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=7)

# standardize the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# apply the scaling to the testing data
X_test = scaler.transform(X_test)

# create an SVM model with the "rbf" kernel and C=3
svm = SVC(kernel="rbf", C=3)

# fit the SVM model to the training data
svm.fit(X_train,y_train)

# use the model to make predictions on the testing data
preds = svm.predict(X_test)

# evaluate the model's performance using classification report and accuracy score
print(classification_report(y_test, preds))
print(accuracy_score(y_test,preds))
