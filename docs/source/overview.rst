Overview
========

``dataclr`` is a Python library for streamlined feature selection in tabular datasets. 
It offers a variety of filter and wrapper methods, delivering robust and interpretable 
feature rankings to enhance model performance and simplify feature engineering.

Key Features
------------

1. **Comprehensive Feature Selection Methods**:

   - **Filter Methods**:
     - Evaluate features independently of the predictive model.
     - Include techniques such as ``MutualInformation``, ``VarianceThreshold``, ``ANOVA``, ``KendallCorrelation``, and more.
     
   - **Wrapper Methods**:
     - Evaluate subsets of features using a predictive model.
     - Include methods such as ``BorutaMethod``, ``ShapMethod``, ``HyperoptMethod``, and ``OptunaMethod``.

2. **Customizable Evaluation Metrics**:

   - Supports both regression and classification tasks with a wide range of metrics.
   - Automatically adapts feature selection strategies based on the nature of the target variable.

3. **Highly Configurable and Scalable**:

   - Allows fine-grained control over the number of selected features, optimization trials, and thresholds.
   - Scales efficiently to handle large datasets and high-dimensional feature spaces.

4. **Interpretable Results**:

   - Provides ranked lists of features with detailed importance scores.
   - Supports visualization and reporting for better interpretability.

5. **Seamless Integration**:

   - Compatible with popular Python libraries such as ``pandas``, ``scikit-learn``.
   - Designed to integrate seamlessly into existing machine learning pipelines.

Use Cases
---------

- **Dimensionality Reduction**: Select the most relevant features for high-dimensional datasets, reducing computational overhead and improving model performance.
- **Feature Engineering**: Identify redundant or irrelevant features to focus on meaningful transformations.
- **Explainable AI (XAI)**: Use interpretable methods like ``ShapMethod`` to understand feature importance and model behavior.
- **Optimization**: Improve the generalization of machine learning models by using well-curated feature subsets.

How It Works
------------

``dataclr`` operates by:

1. **Accepting Tabular Data**: Input datasets in the form of ``pandas`` DataFrames.
2. **Applying Feature Selection Methods**:

   - Filter methods evaluate features based on statistical metrics or relationships with the target.
   - Wrapper methods iteratively select subsets by evaluating feature combinations against a predictive model.

3. **Returning Ranked Features Sets**: Output a ranked list of features sets along with used methods and additional metrics.

``dataclr`` enables machine learning practitioners to perform feature selection efficiently and with ease.
