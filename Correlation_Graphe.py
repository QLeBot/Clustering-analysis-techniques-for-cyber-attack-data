# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:52:53 2022

@author: djiko
"""

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np



df=pd.read_csv("C:/Users/djiko/Documents/ESME/Ingé2/Projet Clustering/ISCXURL2016/FinalDataset/All.csv")

df = df.replace('Defacement', 1)
df = df.replace('begign', 0)
df = df.replace('malware', 2)
df = df.replace('pishing', 3)
df = df.replace('spam', 4)

print(df.head())

# Replacing infinite with nan
df.replace([np.inf, -np.inf], inplace=True)
# Dropping all the rows with nan values
df.dropna(inplace=True)

X=df.drop("URL_Type_obf_Type",axis=1)
y=df["URL_Type_obf_Type"]

from sklearn import decomposition, preprocessing
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

pca = decomposition.PCA(n_components=7)
pca.fit(X_scaled)
print(type(pca.components_))

#fig = plt.figure(figsize=(6, 5))
fig = plt.figure(figsize=(40, 40))
ax = fig.add_subplot(1, 1, 1)
#♣ax.set_xlim([-0.2, 0.3])
#ax.set_ylim([-0.2, 0.3])
ax.set_xlim([-0.3, 0.3])
ax.set_ylim([-0.3, 0.3])
for i, (x, y) in enumerate(zip(pca.components_[0, :], pca.components_[1, :])):
 # plot line between origin and point (x, y)
 ax.plot([0, x], [0, y], color='k')
 #plot point
 #ax.plot(x,y, color='k')
 # display the label of the point

 ax.text(x, y, X.columns[i], fontsize='14')
# ax.text(x, y, i, fontsize='14')