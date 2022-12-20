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

df = pd.read_csv("winequality-red.csv")

X = df.drop("quality",axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=7)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel="rbf", C=3)
svm.fit(X_train,y_train)

preds = svm.predict(X_test)
print(classification_report(y_test, preds))
print(accuracy_score(y_test,preds))