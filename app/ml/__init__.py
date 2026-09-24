"""NEPS ML pipeline.

Flow: loader -> schema.validate -> features -> train -> evaluate + fairness
      -> explain / alerts.
Every stage takes and returns pandas DataFrames so the data source can be
swapped (mock CSV today, REDCap export later) without touching models.
See docs/ml/README.md.
"""
