from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np



df=pd.read_csv("C:/Users/djiko/Documents/ESME/Ingé2/Projet Clustering/ISCXURL2016/FinalDataset/All.csv")

# =============================================================================
# Nombre de type URL différente
# =============================================================================
nb_type_url = len(df["URL_Type_obf_Type"].unique())
print(nb_type_url)

# =============================================================================
# PCA components
# =============================================================================

# Replacing infinite with nan
df.replace([np.inf, -np.inf], inplace=True)
# Dropping all the rows with nan values
df.dropna(inplace=True)

X=df.drop("URL_Type_obf_Type",axis=1)
y=df["URL_Type_obf_Type"]

#print(df[df.isnull().any(axis=1)])

scaler = StandardScaler()
pca=PCA()

pipeline=make_pipeline(scaler,pca)
pipeline.fit(X)

components=range(pca.n_components_)

plt.bar(components, pca.explained_variance_)
plt.xlabel("PCA components")
plt.ylabel("Variance")

# =============================================================================
# Cluster
# =============================================================================


from sklearn.cluster import KMeans
import numpy as np

"""articles=df.drop('words', axis=1)
articles=np.transpose(articles)
titles=df.columns.drop('words')"""


pca = PCA(n_components=8)
kmeans=KMeans(n_clusters=nb_type_url)
pipeline=make_pipeline(pca, kmeans)

pipeline.fit(X)

cluster_labels = pipeline.predict(X)
df_cluster=pd.DataFrame({'cluster labels':cluster_labels,'article titles': y})


print(df_cluster.sort_values('cluster labels'))

# =============================================================================
# Performance
# =============================================================================

model = KMeans(n_clusters=nb_type_url)

#cluster_labels=model.fit_predict(X)

pca = PCA(n_components=6)
kmeans=KMeans(n_clusters=nb_type_url)
pipeline=make_pipeline(pca, kmeans)

pipeline.fit(X)

cluster_labels = pipeline.predict(X)

df_ct =pd.DataFrame({'cluster labels':cluster_labels,'type_url':y})

ct=pd.crosstab(df_ct['cluster labels'],df_ct['type_url'])

print(ct)

# =============================================================================
# Corrélation
# =============================================================================
from sklearn import decomposition, preprocessing
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

pca = decomposition.PCA(n_components=7)
pca.fit(X_scaled)
print(type(pca.components_))

fig = plt.figure(figsize=(6, 5))
#fig = plt.figure(figsize=(40, 40))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim([-0.2, 0.3])
ax.set_ylim([-0.2, 0.3])
#♣ax.set_xlim([0.15, 0.25])
#ax.set_ylim([-0.1, 0.05])
for i, (x, y) in enumerate(zip(pca.components_[0, :], pca.components_[1, :])):
 # plot line between origin and point (x, y)
 ax.plot([0, x], [0, y], color='k')
 #plot point
 ax.plot(x,y, color='k')
 # display the label of the point

#ax.text(x, y, X.columns[i], fontsize='14')
# ax.text(x, y, i, fontsize='14')
 
# =============================================================================
# Classification Tree 
# =============================================================================
 
from sklearn.tree import DecisionTreeClassifier 
# Import train_test_split
from sklearn.model_selection import train_test_split 
# Import accuracy_score
from sklearn.metrics import accuracy_score

"""y = y.replace('Defacement', 1)
y = y.replace('begign', 0)
y = y.replace('malware', 2)
y = y.replace('pishing', 3)
y = y.replace('spam', 4)"""

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=1)
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

#Fit dt to the training set 
dt.fit(X_train,y_train)
# Predict test set labels 
y_pred = dt.predict(X_test)
# Evaluate test-set accuracy 
accuracy_score(y_test, y_pred)
 