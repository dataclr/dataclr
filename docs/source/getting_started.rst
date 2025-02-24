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

.. note::

    **Working with Large Datasets**:
    For very large datasets, it may be advisable to sample the data to speed up feature selection. Ensure the sample is representative of the original dataset to maintain meaningful results. You can use random sampling with libraries like ``pandas``:

    .. code-block:: python

        # Example of random sampling
        sample_fraction = 0.1  # Use 10% of the data
        sampled_data = data.sample(frac=sample_fraction, random_state=42)
        X_sampled = sampled_data.drop(columns=["target"])
        y_sampled = sampled_data["target"]

Model Requirements
-------------------

To work with **dataclr**, your machine learning model must implement the following:

- **fit**: A method to train the model on a dataset. It should have the signature ``fit(X, y)``.
- **predict**: A method to generate predictions on new data. It should have the signature ``predict(X)``.

Additionally, for most wrapper methods, the model must provide feature importance or coefficients via:

- **feature_importances_**: An attribute containing feature importances (e.g., for tree-based models like ``RandomForestClassifier``).
- **coef_**: An attribute containing feature coefficients (e.g., for linear models like ``LogisticRegression``).

These attributes are necessary for evaluating the relative importance of features during the feature selection process.

.. important::
    **Performance Tip:**
    To maximize performance, since **dataclr** algorithms are parallelized and distributed, it is recommended to run your model with a single thread. This avoids interference between parallel processes from the feature selection algorithm and the model.

    - Use ``n_jobs=1`` for models that support multithreading, such as ``RandomForestClassifier`` or ``RandomForestRegressor``.
    - Choose non-parallelized solvers for models where applicable, such as ``solver='liblinear'`` for ``LogisticRegression`` in scikit-learn.

.. note::

    For further details on model implementation, see :class:`dataclr.models.BaseModel`.

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
-----------------

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from dataclr.feature_selection import FeatureSelector

    # Define a Logistic Regression model
    my_model = LogisticRegression(solver="liblinear")

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
-----------------

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    from dataclr.methods import VarianceThreshold

    # Define a Random Forest Regressor model
    my_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Initialize the method
    method = VarianceThreshold(model=my_model, metric="rmse")

    # Fit and transform in one step
    results = method.fit_transform(X_train, X_test, y_train, y_test)

    # Print the results
    for result in results:
        print(f"Feature Set: {result.feature_set}, Score: {result.score}")

