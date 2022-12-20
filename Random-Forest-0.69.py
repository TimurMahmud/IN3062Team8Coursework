# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:56:09 2022

@author: timur
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("winequality-red.csv")

X = df.drop("quality",axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=7)



pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=120, class_weight="balanced",random_state=0)
rf_classifier.fit(X_train_pca, y_train)

preds = rf_classifier.predict(X_test_pca)
print(classification_report(y_test, preds))
print(accuracy_score(y_test,preds))
