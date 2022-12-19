# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:43:26 2022

@author: timur
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("winequality-red.csv")

le = LabelEncoder()
le.fit(df["quality"].unique())
df["quality"] = le.transform(df["quality"])

X = df.drop("quality",axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=7)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(max_iter=5000,solver="lbfgs", alpha=1e-4, hidden_layer_sizes=(3,3), random_state=0)

clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print(classification_report(y_test, preds))
print(accuracy_score(y_test,preds))