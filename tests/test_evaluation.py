# test_evaluation.py
import numpy as np
from src.evaluation import (
    run_clustering_comparison, 
    plot_silhouette_scores,
    validate_clustering_pipeline
)

X_orig = np.random.rand(100, 5)
X_pca = np.random.rand(100, 3)
X_tsne = np.random.rand(100, 2)

def test_run_clustering_comparison_original_only():
    results = run_clustering_comparison(X_orig)
    assert 'Original' in results
    assert isinstance(results['Original'], float)

def test_run_clustering_comparison_all():
    results = run_clustering_comparison(X_orig, X_pca, X_tsne)
    assert 'Original' in results
    assert 'PCA' in results
    assert 't-SNE' in results

def test_plot_silhouette_scores():
    scores = {'Original': 0.4, 'PCA': 0.5}
    fig = plot_silhouette_scores(scores)
    assert fig is not None

def test_validate_clustering_pipeline():
    results = validate_clustering_pipeline(X_orig, n_components=2, n_clusters=3)
    assert 'scores' in results
    assert 'mean' in results
    assert 'std' in results