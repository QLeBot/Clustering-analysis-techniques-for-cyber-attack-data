import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('data/raw/All.csv')

# Drop the categorical column
numeric_df = df.select_dtypes(include=[np.number])

# Calculate correlations
correlations = numeric_df.corr()

# Create heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(correlations, annot=True)
plt.savefig('correlation_heatmap.png')
plt.close() 