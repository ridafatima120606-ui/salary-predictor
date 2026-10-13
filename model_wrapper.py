"""
model_wrapper.py
Shared wrapper — imported by both train_model.py and app.py.
"""

import numpy as np


class RandomForestWrapper:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X):
        Xt = np.ascontiguousarray(self.preprocessor.transform(X), dtype=np.float64)
        return self.model.predict(Xt)
