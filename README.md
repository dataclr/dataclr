# dataclr: The feature selection library

[![PyPI version](https://img.shields.io/pypi/v/dataclr?label=PyPI&color=blue)](https://pypi.org/project/dataclr/)
[![Python Versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/dataclr/dataclr?color=blue)](https://github.com/dataclr/dataclr/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/dataclr/dataclr?label=Stars&color=yellow)](https://github.com/dataclr/dataclr/stargazers)

<div align="center">
  <a href="https://www.dataclr.com/">Docs</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://www.dataclr.com/">Website</a>
  <hr />
</div>

_dataclr_ is a Python library for feature selection, enabling data scientists and ML engineers to identify optimal features from tabular datasets. By combining filter and wrapper methods, it achieves _state-of-the-art_ results, enhancing model performance and simplifying feature engineering.

## Features

- **Comprehensive Methods**:

  - **Filter Methods**: Statistical and data-driven approaches like `ANOVA`, `MutualInformation`, and `VarianceThreshold`.

    | Method                           | Regression | Classification |
    | -------------------------------- | ---------- | -------------- |
    | `ANOVA`                          | Yes        | Yes            |
    | `Chi2`                           | No         | Yes            |
    | `CumulativeDistributionFunction` | Yes        | Yes            |
    | `CohensD`                        | No         | Yes            |
    | `CramersV`                       | No         | Yes            |
    | `DistanceCorrelation`            | Yes        | Yes            |
    | `Entropy`                        | Yes        | Yes            |
    | `KendallCorrelation`             | Yes        | Yes            |
    | `Kurtosis`                       | Yes        | Yes            |
    | `LinearCorrelation`              | Yes        | Yes            |
    | `MaximalInformationCoefficient`  | Yes        | Yes            |
    | `MeanAbsoluteDeviation`          | Yes        | Yes            |
    | `mRMR`                           | Yes        | Yes            |
    | `MutualInformation`              | Yes        | Yes            |
    | `Skewness`                       | Yes        | Yes            |
    | `SpearmanCorrelation`            | Yes        | Yes            |
    | `VarianceThreshold`              | Yes        | Yes            |
    | `VarianceInflationFactor`        | Yes        | Yes            |
    | `ZScore`                         | Yes        | Yes            |

  - **Wrapper Methods**: Model-based iterative methods like `BorutaMethod`, `ShapMethod`, and `OptunaMethod`.

    | Method           | Regression | Classification |
    | ---------------- | ---------- | -------------- |
    | `BorutaMethod`   | Yes        | Yes            |
    | `HyperoptMethod` | Yes        | Yes            |
    | `OptunaMethod`   | Yes        | Yes            |
    | `ShapMethod`     | Yes        | Yes            |

- **Flexible and Scalable**:

  - Supports both regression and classification tasks.
  - Handles high-dimensional datasets efficiently.

- **Interpretable Results**:

  - Provides ranked feature lists with detailed importance scores.
  - Shows used methods along with their parameters.

- **Seamless Integration**:
  - Works with popular Python libraries like `pandas` and `scikit-learn`.

## Installation

Install `dataclr` using pip:

```bash
pip install dataclr
```

## Getting Started

### 1. Load Your Dataset

Prepare your dataset as pandas DataFrames or Series and preprocess it (e.g., encode categorical features and normalize numerical values):

```bash
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example dataset
X = pd.DataFrame({...})  # Replace with your feature matrix
y = pd.Series([...])     # Replace with your target variable

# Preprocessing
X_encoded = pd.get_dummies(X)  # Encode categorical features
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
```

### 2. Use `FeatureSelector`

The `FeatureSelector` is a high-level API that combines multiple methods to select the best feature subsets:

```bash
from sklearn.ensemble import RandomForestClassifier
from dataclr.feature_selection import FeatureSelector

# Define a scikit-learn model
my_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize the FeatureSelector
selector = FeatureSelector(
    model=my_model,
    metric="accuracy",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

# Perform feature selection
selected_features = selector.select_features(n_results=5)
print(selected_features)
```

### 3. Use Singular Methods

For granular control, you can use individual feature selection methods:

```bash
from sklearn.linear_model import LogisticRegression
from dataclr.methods import MutualInformation

# Define a scikit-learn model
my_model = LogisticRegression(solver="liblinear", max_iter=1000)

# Initialize a method
method = MutualInformation(model=my_model, metric="accuracy")

# Fit and transform
results = method.fit_transform(X_train, X_test, y_train, y_test)
print(results)
```

## Benchmarks

As our algorithm produces multiple results, we selected benchmark results that balance feature count with performance, while being capable of achieving the best performance if needed.

![benchmark_bank](https://i.imgur.com/qiG1L9j.png)
![benchmark_students](https://i.imgur.com/FpY3N9h.png)
![benchmark_fifa](https://i.imgur.com/BDTkYgL.png)
![benchmark_uber](https://i.imgur.com/X3uYyCX.png)

## Documentation

Explore the <a href="https://www.dataclr.com">full documentation</a> for detailed usage
instructions, API references, and examples.
