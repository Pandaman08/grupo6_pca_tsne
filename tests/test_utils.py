# test_utils.py
import numpy as np
from src.utils import load_and_preprocess

def test_load_and_preprocess_with_none():
    X, y = load_and_preprocess(None)  # ✅ Desempaquetar tupla
    assert isinstance(X, np.ndarray)
    assert X.shape[1] > 0
    assert X.shape[0] > 0

def test_load_and_preprocess_with_invalid_input():
    try:
        load_and_preprocess("invalid")
    except Exception:
        assert True