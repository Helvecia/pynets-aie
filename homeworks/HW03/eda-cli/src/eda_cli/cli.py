"""Command-line interface for EDA CLI."""

import sys
import click
import pandas as pd
from pathlib import Path

from .core import (
    summarize_dataset,
    missing_table,
    correlation_matrix,
    compute_quality_flags,
    get_top_categories,
    get_problematic_columns,
)
from .viz import (
    plot_histograms,
    plot_correlation_heatmap,
    plot_missing_matrix,
)


@click.group()
def cli():
    """EDA CLI - Exploratory Data Analysis Command Line Interface."""
    pass


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
def overview(filepath):
    """Display dataset overview and quality flags.
    
    Args:
        filepath: Path to CSV file
    """
    try:
        df = pd.read_csv(filepath)
        summary = summarize_dataset(df)
        flags = compute_quality_flags(df)
        
        click.echo("\n=== Dataset Overview ===")
        click.echo(f"Rows: {summary['n_rows']}")
        click.echo(f"Columns: {summary['n_cols']}")
        click.echo(f"Memory: {summary['memory_usage_mb']:.2f} MB")
        
        click.echo("\n=== Data Types ===")
        for col, dtype in summary['dtypes'].items():
            click.echo(f"  {col}: {dtype}")
        
        click.echo("\n=== Quality Flags ===")
        for flag, value in flags.items():
            status = "✓" if value else "✗"
            click.echo(f"  {status} {flag}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./eda_report",
    help="Output directory for report files"
)
@click.option(
    "--max-hist-columns",
    type=int,
    default=20,
    help="Maximum number of columns for histograms"
)
@click.option(
    "--top-k-categories",
    type=int,
    default=10,
    help="Number of top category values to show"
)
@click.option(
    "--title",
    type=str,
    default="EDA Report",
    help="Report title"
)
@click.option(
    "--min-missing-share",
    type=float,
    default=0.1,
    help="Minimum share of missing values to highlight column"
)
def report(filepath, output_dir, max_hist_columns, top_k_categories, title, min_missing_share):
    """Generate comprehensive EDA report with visualizations.
    
    Args:
        filepath: Path to CSV file
        output_dir: Directory to save report files
        max_hist_columns: Maximum columns for histograms
        top_k_categories: Number of top categories to show
        title: Custom report title
        min_missing_share: Threshold for problematic columns
    """
    try:
        df = pd.read_csv(filepath)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"\n=== {title} ===")
        click.echo(f"Input: {filepath}")
        click.echo(f"Output: {output_dir}")
        
        # Summary
        summary = summarize_dataset(df)
        click.echo(f"\nDataset: {summary['n_rows']} rows × {summary['n_cols']} columns")
        
        # Quality flags
        flags = compute_quality_flags(df)
        click.echo("\n=== Quality Flags ===")
        for flag, value in flags.items():
            status = "⚠️  WARNING" if value else "✓ OK"
            click.echo(f"  {status}: {flag}")
        
        # Missing values
        missing = missing_table(df)
        if not missing.empty:
            click.echo("\n=== Missing Values ===")
            missing_file = output_path / "missing_values.csv"
            missing.to_csv(missing_file, index=False)
            click.echo(missing.to_string(index=False))
            click.echo(f"Saved to: {missing_file}")
            
            # Highlight problematic columns
            problematic = get_problematic_columns(df, min_missing_share)
            if problematic:
                click.echo(f"\n⚠️  Problematic columns (>{min_missing_share*100}% missing): {', '.join(problematic)}")
        
        # Correlation matrix
        corr = correlation_matrix(df)
        if not corr.empty:
            click.echo("\n=== Correlation Matrix ===")
            corr_file = output_path / "correlation.csv"
            corr.to_csv(corr_file)
            click.echo(f"Saved to: {corr_file}")
        
        # Categorical top values
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            click.echo(f"\n=== Top {top_k_categories} Category Values ===")
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                top_vals = get_top_categories(df, col, top_k_categories)
                click.echo(f"\n{col}:")
                for val, count in top_vals:
                    pct = 100 * count / len(df)
                    click.echo(f"  {val}: {count} ({pct:.1f}%)")
        
        # Visualizations
        click.echo("\n=== Generating Visualizations ===")
        
        hist_file = output_path / "histograms.png"
        plot_histograms(df, str(hist_file), max_columns=max_hist_columns)
        click.echo(f"  ✓ Histograms: {hist_file}")
        
        if not corr.empty:
            heatmap_file = output_path / "correlation_heatmap.png"
            plot_correlation_heatmap(corr, str(heatmap_file))
            click.echo(f"  ✓ Correlation heatmap: {heatmap_file}")
        
        if not missing.empty:
            missing_plot = output_path / "missing_matrix.png"
            plot_missing_matrix(df, str(missing_plot))
            click.echo(f"  ✓ Missing values matrix: {missing_plot}")
        
        click.echo(f"\n✅ Report generated successfully in: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option(
    "--n",
    type=int,
    default=5,
    help="Number of rows to display"
)
def head(filepath, n):
    """Display first N rows of the dataset.
    
    Args:
        filepath: Path to CSV file
        n: Number of rows to display (default: 5)
    """
    try:
        df = pd.read_csv(filepath)
        click.echo(f"\n=== First {n} rows ===")
        click.echo(df.head(n).to_string())
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option(
    "--n",
    type=int,
    default=5,
    help="Number of rows to sample"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
def sample(filepath, n, seed):
    """Display random N rows from the dataset.
    
    Args:
        filepath: Path to CSV file
        n: Number of rows to sample (default: 5)
        seed: Random seed for reproducibility
    """
    try:
        df = pd.read_csv(filepath)
        sampled = df.sample(n=min(n, len(df)), random_state=seed)
        
        click.echo(f"\n=== Random sample of {n} rows ===")
        if seed is not None:
            click.echo(f"(seed={seed})")
        click.echo(sampled.to_string())
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
