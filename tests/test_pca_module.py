# test_pca_module.py
import numpy as np
from src.pca_module import run_pca, plot_pca_variance, plot_pca_2d, plot_pca_3d

X_dummy = np.random.rand(100, 5)

def test_run_pca():
    result = run_pca(X_dummy, 3)
    assert 'transformed' in result
    assert 'explained_variance_ratio' in result
    assert result['transformed'].shape == (100, 3)
    assert len(result['explained_variance_ratio']) == 3

def test_plot_pca_variance():
    result = run_pca(X_dummy, 3)
    fig = plot_pca_variance(result['explained_variance_ratio'])
    assert fig is not None

def test_plot_pca_2d():
    result = run_pca(X_dummy, 2)
    fig = plot_pca_2d(result['transformed'])
    assert fig is not None

def test_plot_pca_3d():
    result = run_pca(X_dummy, 3)
    fig = plot_pca_3d(result['transformed'])
    assert fig is not None