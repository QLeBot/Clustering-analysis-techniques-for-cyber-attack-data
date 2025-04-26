from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier 
# Import train_test_split
from sklearn.model_selection import train_test_split 
# Import accuracy_score
from sklearn.metrics import accuracy_score


df=pd.read_csv("data/raw/All_BestFirst.csv")

df.replace([np.inf, -np.inf], inplace=True)
# Dropping all the rows with nan values
df.dropna(inplace=True)

X=df.drop("class",axis=1)
y=df["class"]

# Debug information
print("Unique values in class column:", y.unique())
print("Types of values:", [type(x) for x in y.unique()])

# Convert all labels to integers
label_mapping = {
    'Defacement': 1,
    'benign': 0,
    'malware': 2,
    'phishing': 3,
    'spam': 4
}

# Convert y to integers using the mapping
y = y.map(label_mapping)

# Verify all values are integers
if not all(isinstance(x, (int, np.integer)) for x in y):
    raise ValueError("Some labels could not be converted to integers")

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
