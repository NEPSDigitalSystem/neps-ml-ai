"""Shared, session-scoped fixtures so the ML test modules train each model once."""
import warnings

import pytest

from app.ml import features, train
from app.ml.loader import load_datasets

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def data():
    return load_datasets()


@pytest.fixture(scope="session")
def frame(data):
    return features.build_training_frame(data)


@pytest.fixture(scope="session")
def classifier(frame):
    """(model, meta) for the risk classifier."""
    return train.train_risk_classifier(frame)
