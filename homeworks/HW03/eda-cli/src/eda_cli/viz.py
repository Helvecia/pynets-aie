"""Visualization functions for EDA."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional


def plot_histograms(
    df: pd.DataFrame,
    output_file: str,
    max_columns: int = 20,
    figsize_per_plot: tuple = (4, 3)
):
    """Plot histograms for numeric columns.
    
    Args:
        df: Input DataFrame
        output_file: Path to save the plot
        max_columns: Maximum number of columns to plot
        figsize_per_plot: Size of each subplot
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = numeric_cols[:max_columns]
    
    if len(numeric_cols) == 0:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No numeric columns to plot',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(col, fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_file: str,
    figsize: tuple = (10, 8)
):
    """Plot correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        output_file: Path to save the plot
        figsize: Figure size
    """
    if corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No correlation matrix available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()


def plot_missing_matrix(
    df: pd.DataFrame,
    output_file: str,
    figsize: tuple = (12, 6),
    max_columns: int = 50
):
    """Plot missing values matrix.
    
    Args:
        df: Input DataFrame
        output_file: Path to save the plot
        figsize: Figure size
        max_columns: Maximum number of columns to show
    """
    # Select columns with missing values
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    if len(cols_with_missing) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No missing values found',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    # Limit number of columns
    cols_with_missing = cols_with_missing[:max_columns]
    missing_data = df[cols_with_missing].isnull()
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        missing_data.T,
        cmap='YlOrRd',
        cbar_kws={'label': 'Missing'},
        yticklabels=cols_with_missing,
        xticklabels=False
    )
    
    plt.title('Missing Values Pattern', fontsize=14, pad=20)
    plt.xlabel('Row Index', fontsize=10)
    plt.ylabel('Columns', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
