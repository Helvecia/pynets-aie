"""Core functions for EDA CLI."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary information
    """
    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    return summary


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate table with missing value statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with columns: column, missing_count, missing_percent
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    table = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percent": missing_pct.values
    })
    
    # Sort by missing percent descending
    table = table[table["missing_count"] > 0].sort_values(
        "missing_percent", ascending=False
    ).reset_index(drop=True)
    
    return table


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Correlation matrix as DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    return df[numeric_cols].corr()


def compute_quality_flags(
    df: pd.DataFrame,
    missing_threshold: float = 0.5,
    cardinality_threshold: float = 0.5,
    zero_threshold: float = 0.3
) -> Dict[str, bool]:
    """Compute data quality flags based on heuristics.
    
    Args:
        df: Input DataFrame
        missing_threshold: Threshold for high missing values (default: 0.5)
        cardinality_threshold: Threshold for high cardinality categoricals (default: 0.5)
        zero_threshold: Threshold for many zero values (default: 0.3)
        
    Returns:
        Dictionary with boolean flags for quality issues
    """
    flags = {}
    
    # Original flags
    flags["has_missing_values"] = df.isnull().any().any()
    flags["has_duplicates"] = df.duplicated().any()
    
    missing_pct = df.isnull().sum() / len(df)
    flags["has_high_missing_columns"] = (missing_pct > missing_threshold).any()
    
    # NEW FLAG 1: Constant columns
    flags["has_constant_columns"] = (df.nunique() == 1).any()
    
    # NEW FLAG 2: High cardinality categoricals
    # Check object/category columns with more than cardinality_threshold unique values
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality = False
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > cardinality_threshold:
            high_cardinality = True
            break
    flags["has_high_cardinality_categoricals"] = high_cardinality
    
    # NEW FLAG 3: Suspicious ID duplicates
    # Look for columns that look like IDs (name contains 'id', '_id', 'ID') with duplicates
    suspicious_id_duplicates = False
    for col in df.columns:
        col_lower = col.lower()
        if 'id' in col_lower or col.endswith('_id') or col.endswith('ID'):
            if df[col].duplicated().any():
                suspicious_id_duplicates = True
                break
    flags["has_suspicious_id_duplicates"] = suspicious_id_duplicates
    
    # NEW FLAG 4: Many zero values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    many_zeros = False
    for col in numeric_cols:
        zero_ratio = (df[col] == 0).sum() / len(df)
        if zero_ratio > zero_threshold:
            many_zeros = True
            break
    flags["has_many_zero_values"] = many_zeros
    
    return flags


def get_top_categories(
    df: pd.DataFrame, 
    column: str, 
    top_k: int = 10
) -> List[Tuple[Any, int]]:
    """Get top K most frequent values in a categorical column.
    
    Args:
        df: Input DataFrame
        column: Column name
        top_k: Number of top values to return
        
    Returns:
        List of tuples (value, count)
    """
    if column not in df.columns:
        return []
    
    value_counts = df[column].value_counts().head(top_k)
    return list(zip(value_counts.index, value_counts.values))


def get_problematic_columns(
    df: pd.DataFrame,
    min_missing_share: float = 0.1
) -> List[str]:
    """Get list of columns with missing values above threshold.
    
    Args:
        df: Input DataFrame
        min_missing_share: Minimum share of missing values (default: 0.1)
        
    Returns:
        List of column names
    """
    missing_pct = df.isnull().sum() / len(df)
    problematic = missing_pct[missing_pct >= min_missing_share]
    return list(problematic.index)
