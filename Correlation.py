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
        
        # Drop non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
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
            # Filter features to only include those that exist in numeric_df
            valid_features = [f for f in features if f in numeric_df.columns]
            if valid_features:
                plot_correlation_heatmap(numeric_df, valid_features, f"{group_name} Correlation")
            else:
                logger.warning(f"No valid numeric features found for group: {group_name}")
            
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()