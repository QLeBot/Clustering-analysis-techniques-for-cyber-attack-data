"""
Database discovery and analysis module.
"""
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from settings import (
    DATASET_PATH, PCA_COMPONENTS, N_CLUSTERS, 
    TEST_SIZE, RANDOM_STATE, PLOT_FIGSIZE
)
from utils import load_data, prepare_features_target, save_plot

logger = logging.getLogger(__name__)

def analyze_url_types(df: pd.DataFrame) -> None:
    """Analyze and log the number of unique URL types."""
    nb_type_url = len(df["URL_Type_obf_Type"].unique())
    logger.info(f"Number of unique URL types: {nb_type_url}")
    return nb_type_url

def perform_pca_analysis(X: pd.DataFrame) -> None:
    """Perform PCA analysis and plot explained variance."""
    try:
        scaler = StandardScaler()
        pca = PCA()
        pipeline = make_pipeline(scaler, pca)
        pipeline.fit(X)

        # Plot explained variance
        plt.figure(figsize=PLOT_FIGSIZE)
        plt.bar(range(pca.n_components_), pca.explained_variance_)
        plt.xlabel("PCA Components")
        plt.ylabel("Explained Variance")
        plt.title("PCA Explained Variance")
        save_plot(plt.gcf(), "pca_explained_variance")
        plt.close()
        
        logger.info(f"Total explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")
    except Exception as e:
        logger.error(f"Error in PCA analysis: {str(e)}")
        raise

def perform_clustering(X: pd.DataFrame, y: pd.Series, n_clusters: int) -> None:
    """Perform clustering analysis."""
    try:
        pca = PCA(n_components=PCA_COMPONENTS)
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        pipeline = make_pipeline(pca, kmeans)
        
        # Fit and predict
        pipeline.fit(X)
        cluster_labels = pipeline.predict(X)
        
        # Create and display cluster analysis
        df_cluster = pd.DataFrame({
            'cluster_labels': cluster_labels,
            'url_type': y
        })
        
        # Print cluster distribution
        logger.info("\nCluster Distribution:")
        print(df_cluster.sort_values('cluster_labels'))
        
        # Create and display confusion matrix
        ct = pd.crosstab(df_cluster['cluster_labels'], df_cluster['url_type'])
        logger.info("\nCluster vs URL Type Crosstab:")
        print(ct)
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}")
        raise

def train_decision_tree(X: pd.DataFrame, y: pd.Series) -> float:
    """Train and evaluate a decision tree classifier."""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Train model
        dt = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
        dt.fit(X_train, y_train)
        
        # Evaluate
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Decision Tree Accuracy: {accuracy:.2f}")
        
        return accuracy
    except Exception as e:
        logger.error(f"Error in decision tree training: {str(e)}")
        raise

def main():
    """Main function to run database discovery and analysis."""
    try:
        # Load and preprocess data
        df = load_data(DATASET_PATH)
        X, y = prepare_features_target(df)
        
        # Analyze URL types
        n_clusters = analyze_url_types(df)
        
        # Perform PCA analysis
        perform_pca_analysis(X)
        
        # Perform clustering
        perform_clustering(X, y, n_clusters)
        
        # Train decision tree
        train_decision_tree(X, y)
        
    except Exception as e:
        logger.error(f"Error in database discovery: {str(e)}")
        raise

if __name__ == "__main__":
    main()
 