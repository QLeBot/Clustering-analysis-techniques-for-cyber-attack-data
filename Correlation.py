import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
import logging

from settings import DATASET_PATH, HEATMAP_FIGSIZE
from utils import load_data, save_plot

logger = logging.getLogger(__name__)

def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str], title: str) -> None:
    """
    Plot correlation heatmap for specified columns.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to include in the correlation
        title: Title for the plot
    """
    try:
        df_subset = df[columns]
        plt.figure(figsize=HEATMAP_FIGSIZE)
        cor = df_subset.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.title(title)
        save_plot(plt.gcf(), f"correlation_{title.lower().replace(' ', '_')}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {str(e)}")
        raise

def main():
    """Main function to run correlation analysis."""
    try:
        # Load and preprocess data
        df = load_data(DATASET_PATH)
        
        # Define feature groups for correlation analysis
        feature_groups = {
            "Download_Features": [
                "dld_url", "dld_path", "dld_getArg", "charcompvowels", 
                "charcompace", "Extension_DigitCount", "URL_DigitCount",
                "Query_DigitCount", "ArgLen", "urlLen", "argDomanRatio",
                "pathLength", "subDirLen", "URL_Letter_Count",
                "LongestPathTokenLength", "pathDomainRatio", "Querylength",
                "ldl_url", "ldl_getArg"
            ],
            "Symbol_Features": [
                "SymbolCount_URL", "SymbolCount_Afterpath",
                "SymbolCount_FileName", "SymbolCount_Extension",
                "delimeter_Count", "URLQueries_variable"
            ],
            "Directory_Features": [
                "NumberRate_FileName", "SymbolCount_Directoryname",
                "NumberofDotsinURL", "Entropy_DirectoryName",
                "SymbolCount_Domain", "domain_token_count",
                "sub-Directory_LongestWordLength", "NumberRate_DirectoryName",
                "Entropy_Filename", "Path_LongestWordLength",
                "Entropy_Extension", "Directory_LetterCount",
                "domainlength"
            ]
        }
        
        # Plot correlation heatmaps for each feature group
        for group_name, features in feature_groups.items():
            plot_correlation_heatmap(df, features, f"{group_name} Correlation")
            
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()





df=pd.read_csv("data/raw/All.csv")

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





df=pd.read_csv("data/raw/All.csv")

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


