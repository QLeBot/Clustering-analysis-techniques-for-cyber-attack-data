# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:43:24 2022

@author: djiko
"""

#Importing required modules
 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

data=pd.read_csv("C:/Users/djiko/Documents/ESME/Ingé2/Projet Clustering/ISCXURL2016/FinalDataset/All_BestFirst.csv")

# Replacing infinite with nan
data.replace([np.inf, -np.inf], inplace=True)
# Dropping all the rows with nan values
data.dropna(inplace=True)

#X=data.drop("URL_Type_obf_Type",axis=1)
#y=data["URL_Type_obf_Type"]
X=data.drop("class",axis=1)
y=data["class"]


#Load Data
pca = PCA(2)
 
#Transform the data
df = pca.fit_transform(X)
 
df.shape

#Import required module
from sklearn.cluster import KMeans
 
#Initialize the class object
kmeans = KMeans(n_clusters= 5)
 
#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
print(label)

import matplotlib.pyplot as plt
 
#filter rows of original data
filtered_label0 = df[label == 0]
 
#plotting the results
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.show()

#filter rows of original data
filtered_label2 = df[label == 2]
 
filtered_label8 = df[label == 8]
 
#Plotting the results
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
plt.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'black')
plt.show()