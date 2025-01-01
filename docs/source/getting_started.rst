Getting Started
===============

This guide will help you quickly start using **dataclr** for feature selection in your machine learning pipeline.

Installation
------------

Install the library using pip:

.. code-block:: bash

    pip install dataclr

Data Requirements
------------------

To ensure accurate feature selection, your dataset must meet the following requirements:

- **Encoded Data**: All categorical features must be encoded (e.g., using one-hot encoding or label encoding).
- **Normalized Data**: Numerical features should be scaled to a similar range (e.g., using standardization or min-max scaling).

You can use libraries like ``pandas`` or ``scikit-learn`` to preprocess your data:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler

    # Example preprocessing
    X_encoded = pd.get_dummies(X)  # Encode categorical features
    scaler = StandardScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

Using FeatureSelector
---------------------

The ``FeatureSelector`` class provides a high-level API for selecting the best features from a dataset by combining filter and wrapper methods.

1. **Initialize the FeatureSelector**:
   Pass your model, metric, and training/testing datasets to the ``FeatureSelector`` class.

   .. code-block:: python

       from dataclr.feature_selection import FeatureSelector

       selector = FeatureSelector(
           model=my_model,
           metric="accuracy",
           X_train=X_train,
           X_test=X_test,
           y_train=y_train,
           y_test=y_test,
       )

2. **Select Features**:
   Use the ``select_features`` method to automatically determine the best feature subsets.

   .. code-block:: python

       selected_features = selector.select_features(n_results=5)
       print(selected_features)

Example Workflow:

.. code-block:: python

    from dataclr.feature_selection import FeatureSelector

    # Initialize the FeatureSelector
    selector = FeatureSelector(
        model=my_model,
        metric="f1",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # Perform feature selection
    selected_features = selector.select_features(n_results=10)
    print(selected_features)

Using Singular Methods
----------------------

If you want more granular control over the feature selection process, you can use singular methods directly. Here’s how:

1. **Initialize a Method**:
   Choose a specific filter or wrapper method (e.g., ``MutualInformation``, ``ShapMethod``).

   .. code-block:: python

       from dataclr.methods import MutualInformation

       method = MutualInformation(model=my_model, metric="accuracy")

2. **Fit and Retrieve Ranked Features**:
   Fit the method to your dataset and retrieve the ranked features.

   .. code-block:: python

       method.fit(X_train, y_train)
       print(method.ranked_features_)

Example Workflow:

.. code-block:: python

    from dataclr.methods import VarianceThreshold

    # Initialize the method
    method = VarianceThreshold(model=my_model, metric="rmse")

    # Fit and transform in one step
    results = method.fit_transform(X_train, X_test, y_train, y_test)

    # Print the results
    for result in results:
        print(f"Feature Set: {result.feature_set}, Score: {result.score}")
