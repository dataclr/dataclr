.. module:: dataclr.models

dataclr.models
===============

To maximize the performance of **dataclr** models during feature selection:

- **Single-threaded Execution**: Ensure that models are configured to use a single thread (e.g., ``n_jobs=1``) if they support parallel execution. This avoids contention between the parallelized feature selection process and the model's internal parallelism.

- **Non-parallelized Solvers**: For models like ``LogisticRegression`` in scikit-learn, use solvers that are not parallelized, such as ``solver='liblinear'``.

These adjustments ensure the distributed feature selection algorithms in **dataclr** operate efficiently without interference.


.. automodule:: dataclr.models
   :members:
   :undoc-members:
   :inherited-members: