# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:58:35 2022

@author: djiko
"""

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df=pd.read_csv("C:/Users/djiko/Documents/ESME/Ingé2/Projet Clustering/ISCXURL2016/FinalDataset/All_InfoGain.csv")

df.replace([np.inf, -np.inf], inplace=True)
# Dropping all the rows with nan values
df.dropna(inplace=True)

X=df.drop("class",axis=1)
y=df["class"]

from sklearn.tree import DecisionTreeClassifier 
# Import train_test_split
from sklearn.model_selection import train_test_split 
# Import accuracy_score
from sklearn.metrics import accuracy_score

y = y.replace('Defacement', 1)
y = y.replace('begign', 0)
y = y.replace('malware', 2)
y = y.replace('pishing', 3)
y = y.replace('spam', 4)

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=1)
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

#Fit dt to the training set 
dt.fit(X_train,y_train)
# Predict test set labels 
y_pred = dt.predict(X_test)
# Evaluate test-set accuracy 
accuracy=accuracy_score(y_test, y_pred)

print(accuracy)