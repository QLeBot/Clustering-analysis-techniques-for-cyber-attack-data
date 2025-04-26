"""
Utility functions for the clustering analysis project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        
        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with NaN values
        initial_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with NaN values")
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features_target(df: pd.DataFrame, target_column: str = "URL_Type_obf_Type") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def save_plot(fig, filename: str, directory: Optional[Path] = None) -> None:
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: matplotlib figure object
        filename: Name of the output file
        directory: Directory to save the plot (optional)
    """
    from settings import RESULTS_DIR, PLOT_FORMAT, DPI
    
    if directory is None:
        directory = RESULTS_DIR
        
    output_path = directory / f"{filename}.{PLOT_FORMAT}"
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}") 