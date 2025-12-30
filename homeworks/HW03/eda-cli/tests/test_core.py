"""Unit tests for core EDA functions."""

import pytest
import pandas as pd
import numpy as np
from eda_cli.core import (
    summarize_dataset,
    missing_table,
    correlation_matrix,
    compute_quality_flags,
    get_top_categories,
    get_problematic_columns,
)


def test_summarize_dataset():
    """Test dataset summarization."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    summary = summarize_dataset(df)
    
    assert summary["n_rows"] == 3
    assert summary["n_cols"] == 2
    assert "a" in summary["columns"]
    assert "b" in summary["columns"]
    assert summary["memory_usage_mb"] > 0


def test_missing_table():
    """Test missing value table generation."""
    df = pd.DataFrame({
        "a": [1, np.nan, 3],
        "b": ["x", "y", None],
        "c": [1, 2, 3],
    })
    missing = missing_table(df)
    
    assert len(missing) == 2  # Only columns with missing values
    assert "a" in missing["column"].values
    assert "b" in missing["column"].values
    assert (missing["missing_percent"] > 0).all()


def test_correlation_matrix():
    """Test correlation matrix computation."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 6, 8],
        "c": ["x", "y", "z", "w"],
    })
    corr = correlation_matrix(df)
    
    assert len(corr) == 2  # Only numeric columns
    assert "a" in corr.columns
    assert "b" in corr.columns
    assert abs(corr.loc["a", "b"] - 1.0) < 0.01  # Perfect correlation


def test_compute_quality_flags_basic():
    """Test basic quality flags."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    flags = compute_quality_flags(df)
    
    assert "has_missing_values" in flags
    assert "has_duplicates" in flags
    assert "has_high_missing_columns" in flags
    assert flags["has_missing_values"] is False
    assert flags["has_duplicates"] is False


def test_quality_flag_constant_columns():
    """Test detection of constant columns."""
    df_constant = pd.DataFrame({
        "a": [1, 1, 1, 1],
        "b": [2, 3, 4, 5],
    })
    flags = compute_quality_flags(df_constant)
    assert flags["has_constant_columns"] is True
    
    df_no_constant = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 3, 4, 5],
    })
    flags = compute_quality_flags(df_no_constant)
    assert flags["has_constant_columns"] is False


def test_quality_flag_high_cardinality():
    """Test detection of high cardinality categorical columns."""
    # High cardinality: 8 unique values in 10 rows (80%)
    df_high = pd.DataFrame({
        "category": ["a", "b", "c", "d", "e", "f", "g", "h", "a", "b"],
    })
    flags = compute_quality_flags(df_high, cardinality_threshold=0.5)
    assert flags["has_high_cardinality_categoricals"] is True
    
    # Low cardinality: 2 unique values in 10 rows (20%)
    df_low = pd.DataFrame({
        "category": ["a", "b"] * 5,
    })
    flags = compute_quality_flags(df_low, cardinality_threshold=0.5)
    assert flags["has_high_cardinality_categoricals"] is False


def test_quality_flag_suspicious_id_duplicates():
    """Test detection of suspicious ID duplicates."""
    df_with_dup_id = pd.DataFrame({
        "user_id": [1, 2, 2, 3],
        "value": [10, 20, 30, 40],
    })
    flags = compute_quality_flags(df_with_dup_id)
    assert flags["has_suspicious_id_duplicates"] is True
    
    df_no_dup_id = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "value": [10, 20, 30, 40],
    })
    flags = compute_quality_flags(df_no_dup_id)
    assert flags["has_suspicious_id_duplicates"] is False


def test_quality_flag_many_zeros():
    """Test detection of many zero values in numeric columns."""
    # 60% zeros (6 out of 10)
    df_many_zeros = pd.DataFrame({
        "value": [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
    })
    flags = compute_quality_flags(df_many_zeros, zero_threshold=0.3)
    assert flags["has_many_zero_values"] is True
    
    # 20% zeros (2 out of 10)
    df_few_zeros = pd.DataFrame({
        "value": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    })
    flags = compute_quality_flags(df_few_zeros, zero_threshold=0.3)
    assert flags["has_many_zero_values"] is False


def test_get_top_categories():
    """Test getting top category values."""
    df = pd.DataFrame({
        "category": ["a", "b", "a", "c", "a", "b"],
    })
    top = get_top_categories(df, "category", top_k=2)
    
    assert len(top) == 2
    assert top[0][0] == "a"  # Most frequent
    assert top[0][1] == 3    # Count
    assert top[1][0] == "b"  # Second most frequent
    assert top[1][1] == 2    # Count


def test_get_problematic_columns():
    """Test identification of problematic columns with high missing rates."""
    df = pd.DataFrame({
        "a": [1, np.nan, np.nan, np.nan, 5],  # 60% missing
        "b": [1, 2, np.nan, 4, 5],            # 20% missing
        "c": [1, 2, 3, 4, 5],                 # 0% missing
    })
    
    problematic = get_problematic_columns(df, min_missing_share=0.3)
    assert "a" in problematic
    assert "b" not in problematic
    assert "c" not in problematic
    
    problematic_low = get_problematic_columns(df, min_missing_share=0.1)
    assert "a" in problematic_low
    assert "b" in problematic_low
    assert "c" not in problematic_low
