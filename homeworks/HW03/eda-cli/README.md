# EDA CLI - Exploratory Data Analysis Command Line Interface

**HW03 Solution** for pynets-aie course

A Python command-line tool for automated exploratory data analysis of CSV datasets with quality flag detection, statistical summaries, and visualizations.

---

## ✨ Features

### 📊 Data Quality Flags
Automatically detects 7 common data quality issues:
- ✅ Missing values
- ✅ Duplicate rows
- ✅ High missing rate columns (>50%)
- ✅ **Constant columns** (single unique value)
- ✅ **High cardinality categoricals** (>50% unique values)
- ✅ **Suspicious ID duplicates** (ID-like columns with duplicates)
- ✅ **Many zero values** (>30% zeros in numeric columns)

### 🔧 CLI Commands
- `overview` - Quick dataset summary and quality flags
- `report` - Comprehensive EDA report with visualizations
- `head` - Display first N rows
- `sample` - Display random N rows

### 📈 Visualizations
- Histograms for numeric columns
- Correlation heatmap
- Missing values matrix

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended) or `pip`

### Setup

```bash
# Clone repository
git clone https://github.com/Helvecia/pynets-aie.git
cd pynets-aie/homeworks/HW03/eda-cli

# Install with uv (recommended)
uv pip install -e .

# OR install with pip
pip install -e .
```

---

## 📖 Usage

### Command: `overview`
Display quick dataset summary and quality flags.

```bash
eda-cli overview data/example.csv
```

**Output:**
```
=== Dataset Overview ===
Rows: 100
Columns: 5
Memory: 0.45 MB

=== Data Types ===
  age: int64
  name: object
  salary: float64

=== Quality Flags ===
  ✓ has_missing_values
  ✗ has_duplicates
  ✓ has_constant_columns
  ✗ has_high_cardinality_categoricals
  ...
```

---

### Command: `report`
Generate comprehensive EDA report with visualizations.

```bash
# Basic usage
eda-cli report data/example.csv

# With custom options
eda-cli report data/example.csv \
  --output-dir ./my_report \
  --title "Customer Dataset Analysis" \
  --top-k-categories 15 \
  --min-missing-share 0.2 \
  --max-hist-columns 30
```

**Options:**
- `--output-dir`, `-o`: Output directory (default: `./eda_report`)
- `--max-hist-columns`: Max columns for histograms (default: 20)
- `--top-k-categories`: Number of top category values to show (default: 10)
- `--title`: Custom report title (default: "EDA Report")
- `--min-missing-share`: Threshold for problematic columns (default: 0.1)

**Generated files:**
```
eda_report/
├── missing_values.csv       # Missing value statistics
├── correlation.csv          # Correlation matrix
├── histograms.png           # Numeric column distributions
├── correlation_heatmap.png  # Correlation visualization
└── missing_matrix.png       # Missing value patterns
```

---

### Command: `head`
Display first N rows of the dataset.

```bash
# Show first 5 rows (default)
eda-cli head data/example.csv

# Show first 20 rows
eda-cli head data/example.csv --n 20
```

**Output:**
```
=== First 5 rows ===
   age    name  salary
0   25   Alice   50000
1   30     Bob   60000
2   35  Charlie  70000
...
```

---

### Command: `sample`
Display random N rows from the dataset.

```bash
# Random 5 rows (default)
eda-cli sample data/example.csv

# Random 10 rows with seed for reproducibility
eda-cli sample data/example.csv --n 10 --seed 42
```

**Output:**
```
=== Random sample of 5 rows ===
(seed=42)
    age    name  salary
42   28   David   55000
17   33     Eve   65000
...
```

---

## 🧪 Testing

Run unit tests with pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=eda_cli --cov-report=html

# Run specific test
pytest tests/test_core.py::test_quality_flag_constant_columns -v
```

**Test coverage:**
- ✅ `test_summarize_dataset` - Dataset summary statistics
- ✅ `test_missing_table` - Missing value detection
- ✅ `test_correlation_matrix` - Correlation computation
- ✅ `test_compute_quality_flags_basic` - Basic quality flags
- ✅ `test_quality_flag_constant_columns` - Constant column detection
- ✅ `test_quality_flag_high_cardinality` - High cardinality detection
- ✅ `test_quality_flag_suspicious_id_duplicates` - ID duplicate detection
- ✅ `test_quality_flag_many_zeros` - Zero value detection
- ✅ `test_get_top_categories` - Top category extraction
- ✅ `test_get_problematic_columns` - Problematic column identification

---

## 📁 Project Structure

```
eda-cli/
├── src/
│   └── eda_cli/
│       ├── __init__.py      # Package initialization
│       ├── core.py          # Core EDA functions (quality flags, statistics)
│       ├── cli.py           # Click-based CLI commands
│       └── viz.py           # Visualization functions
├── tests/
│   └── test_core.py         # Unit tests
├── data/
│   └── example.csv          # Sample dataset
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

---

## 🛠️ Implementation Details

### Quality Flag Heuristics

1. **`has_constant_columns`**  
   Detects columns with only one unique value using `df.nunique() == 1`

2. **`has_high_cardinality_categoricals`**  
   Flags categorical columns where `unique_values / total_rows > 0.5`

3. **`has_suspicious_id_duplicates`**  
   Checks if columns containing "id" (case-insensitive) have duplicate values

4. **`has_many_zero_values`**  
   Flags numeric columns where `zero_count / total_rows > 0.3`

### Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Visualizations
- `seaborn` - Statistical plots
- `click` - CLI framework
- `pytest` - Testing

---

## 📝 Example Workflow

```bash
# 1. Quick overview
eda-cli overview sales_data.csv

# 2. Generate full report
eda-cli report sales_data.csv --title "Sales Analysis Q4 2025"

# 3. Inspect sample data
eda-cli sample sales_data.csv --n 20 --seed 123

# 4. Check first rows
eda-cli head sales_data.csv --n 10
```

---

## 👨‍💻 Author
**Helvecia** - pynets-aie course, HW03  
GitHub: [@Helvecia](https://github.com/Helvecia)

---

## 📄 License
MIT License - see repository for details
