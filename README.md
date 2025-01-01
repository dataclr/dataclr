# dataclr

**dataclr** is a Python library for feature selection, designed to help machine learning engineers and data scientists quickly identify the best features from tabular datasets. By combining a wide range of filter, wrapper, and embedded methods, `dataclr` provides a robust and versatile approach to improve model performance and streamline feature engineering.

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
  - Supports visualization and reporting.

- **Seamless Integration**:
  - Works with popular Python libraries like `pandas`, `scikit-learn`, and `statsmodels`.

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
from dataclr.feature_selection import FeatureSelector

# Initialize the FeatureSelector
selector = FeatureSelector(
    model=my_model,  # Replace with your model
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
from dataclr.methods import MutualInformation

# Initialize a method
method = MutualInformation(model=my_model, metric="accuracy")

# Fit and transform
results = method.fit_transform(X_train, X_test, y_train, y_test)
print(results)
```

## Documentation

Explore the full documentation for detailed usage instructions, API references, and examples.
