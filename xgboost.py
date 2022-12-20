# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:36:49 2022

@author: timur
"""

# in anaconda cmd line write 'pip install xgboost'

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

xgb_cl = xgb.XGBClassifier(max_depth=20,eta=0.1)
xgb_cl.fit(X_train,y_train)

preds = xgb_cl.predict(X_test)

print(classification_report(y_test, preds))
print(accuracy_score(y_test,preds))