.. module:: dataclr.feature_selector

dataclr.feature_selector
========================

The :mod:`~dataclr.feature_selector` module is the main component for running the 
feature selection algorithm in the ``dataclr`` package. It provides the 
:class:`~dataclr.feature_selector.FeatureSelector` class, which orchestrates the evaluation of machine learning 
models, applies various filter and wrapper feature selection methods, and identifies 
the best subset of features based on a specified performance metric.

.. important::
   In order to use most **wrapper methods**, the model must have either the ``coef_`` 
   or ``feature_importances_`` attribute. These attributes are used to evaluate the 
   importance of features during the selection process. Ensure that your model 
   implements at least one of these attributes for compatibility.

.. autoclass:: dataclr.feature_selector.FeatureSelector
   :members:
   :undoc-members:
   :inherited-members: