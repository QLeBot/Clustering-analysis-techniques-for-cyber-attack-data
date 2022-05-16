# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:02:18 2022

@author: djiko
"""

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df=pd.read_csv("C:/Users/djiko/Documents/ESME/Ingé2/Projet Clustering/ISCXURL2016/FinalDataset/All.csv")

"""df.replace([np.inf, -np.inf], inplace=True)
# Dropping all the rows with nan values
df.dropna(inplace=True)"""

X=df.drop("URL_Type_obf_Type",axis=1)
y=df["URL_Type_obf_Type"]


#Using Pearson Correlation
plt.figure(figsize=(100,100))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()





df=pd.read_csv("C:/Users/djiko/Documents/ESME/Ingé2/Projet Clustering/ISCXURL2016/FinalDataset/All.csv")

"""df.replace([np.inf, -np.inf], inplace=True)
# Dropping all the rows with nan values
df.dropna(inplace=True)

X=df.drop("URL_Type_obf_Type",axis=1)
y=df["URL_Type_obf_Type"]"""

#df_test = df[["dld_url","dld_path","dld_getArg","charcompvowels","charcompace","Extension_DigitCount","URL_DigitCount"]]
#df_test = df_test.append(df[["Query_DigitCount","ArgLen","urlLen","argDomanRatio","pathLength","subDirLen"]])

df_test = df[["dld_url","dld_path","dld_getArg","charcompvowels","charcompace","Extension_DigitCount","URL_DigitCount","Query_DigitCount","ArgLen","urlLen","argDomanRatio","pathLength","subDirLen","URL_Letter_Count","LongestPathTokenLength","pathDomainRatio","Querylength","ldl_url","ldl_getArg"]]
#Using Pearson Correlation

plt.figure(figsize=(10,10))
cor = df_test.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


df_test2=df[["SymbolCount_URL","SymbolCount_Afterpath","SymbolCount_FileName","SymbolCount_Extension","delimeter_Count","URLQueries_variable"]]
plt.figure(figsize=(10,10))
cor = df_test2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


df_group3=df[["NumberRate_FileName","SymbolCount_Directoryname","NumberofDotsinURL","Entropy_DirectoryName","SymbolCount_Domain","domain_token_count","sub-Directory_LongestWordLength","NumberRate_DirectoryName","Entropy_Filename","Path_LongestWordLength","Entropy_Extension","Directory_LetterCount","domainlength"]]
plt.figure(figsize=(10,10))
cor = df_group3.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


