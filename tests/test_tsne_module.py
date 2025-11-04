# test_tsne_module.py
import numpy as np
from src.tsne_module import run_tsne, plot_tsne_2d

X_dummy = np.random.rand(100, 5)

def test_run_tsne():
    result = run_tsne(X_dummy, perplexity=30, n_iter=250)
    assert 'transformed' in result
    assert result['transformed'].shape == (100, 2)

def test_plot_tsne_2d():
    result = run_tsne(X_dummy, perplexity=30, n_iter=250)
    fig = plot_tsne_2d(result['transformed'])
    assert fig is not None