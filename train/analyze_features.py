import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
from typing import Dict, Tuple
import os


def load_features(feature_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load features and labels."""
    data = torch.load(feature_path, weights_only=False)
    features = data['features'].numpy()
    labels = data['labels'].numpy()
    return features, labels


def analyze_feature_distribution(features: np.ndarray, labels: np.ndarray, save_dir: str):
    """Analyze feature distribution."""
    plt.figure(figsize=(15, 5))
    
    # Feature statistics
    plt.subplot(1, 3, 1)
    plt.hist(features.flatten(), bins=50, alpha=0.7)
    plt.title('Feature Value Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    
    # Feature differences between classes
    plt.subplot(1, 3, 2)
    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]
    
    plt.hist(class_0_features.mean(axis=1), bins=30, alpha=0.7, label='Class 0 (NonFight)', density=True)
    plt.hist(class_1_features.mean(axis=1), bins=30, alpha=0.7, label='Class 1 (Fight)', density=True)
    plt.title('Mean Feature Values by Class')
    plt.xlabel('Mean Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Feature norm distribution
    plt.subplot(1, 3, 3)
    norms = np.linalg.norm(features, axis=1)
    plt.hist(norms, bins=30, alpha=0.7)
    plt.title('Feature Norm Distribution')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_feature_space(features: np.ndarray, labels: np.ndarray, save_dir: str):
    """Visualize feature space."""
    # PCA dimensionality reduction
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    features_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 5))
    
    # PCA visualization
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    
    # t-SNE visualization
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()


def calculate_separability_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate class separability metrics."""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        print(f"Warning: Only {n_classes} class(es) found. Cannot calculate separability metrics.")
        return {
            'separability': 0.0,
            'silhouette_score': 0.0,
            'inter_class_distance': 0.0,
            'intra_class_0_distance': 0.0,
            'intra_class_1_distance': 0.0,
            'n_classes': n_classes
        }
    
    # Calculate inter-class distance
    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]
    
    # Intra-class distance
    intra_class_0 = np.mean([np.linalg.norm(f - class_0_features.mean(axis=0)) for f in class_0_features])
    intra_class_1 = np.mean([np.linalg.norm(f - class_1_features.mean(axis=0)) for f in class_1_features])
    
    # Inter-class distance
    inter_class = np.linalg.norm(class_0_features.mean(axis=0) - class_1_features.mean(axis=0))
    
    # Separability metric
    separability = inter_class / (intra_class_0 + intra_class_1)
    
    # Silhouette Score (only calculate when there are multiple classes)
    try:
        silhouette = silhouette_score(features, labels)
    except ValueError as e:
        print(f"Warning: Cannot calculate silhouette score: {e}")
        silhouette = 0.0
    
    return {
        'separability': separability,
        'silhouette_score': silhouette,
        'inter_class_distance': inter_class,
        'intra_class_0_distance': intra_class_0,
        'intra_class_1_distance': intra_class_1,
        'n_classes': n_classes
    }


def analyze_class_characteristics(features: np.ndarray, labels: np.ndarray, save_dir: str):
    """Analyze characteristics of each class."""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        print(f"Warning: Only {n_classes} class(es) found. Skipping class characteristics analysis.")
        return
    
    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]
    
    plt.figure(figsize=(15, 10))
    
    # Feature mean comparison
    plt.subplot(2, 3, 1)
    mean_0 = class_0_features.mean(axis=0)
    mean_1 = class_1_features.mean(axis=0)
    
    feature_indices = np.arange(min(100, len(mean_0)))  # Show only first 100 features
    plt.plot(feature_indices, mean_0[feature_indices], label='Class 0 (NonFight)', alpha=0.7)
    plt.plot(feature_indices, mean_1[feature_indices], label='Class 1 (Fight)', alpha=0.7)
    plt.title('Mean Feature Values Comparison')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.legend()
    
    # Feature variance comparison
    plt.subplot(2, 3, 2)
    var_0 = class_0_features.var(axis=0)
    var_1 = class_1_features.var(axis=0)
    
    plt.plot(feature_indices, var_0[feature_indices], label='Class 0 (NonFight)', alpha=0.7)
    plt.plot(feature_indices, var_1[feature_indices], label='Class 1 (Fight)', alpha=0.7)
    plt.title('Feature Variance Comparison')
    plt.xlabel('Feature Index')
    plt.ylabel('Variance')
    plt.legend()
    
    # Feature norm distribution
    plt.subplot(2, 3, 3)
    norms_0 = np.linalg.norm(class_0_features, axis=1)
    norms_1 = np.linalg.norm(class_1_features, axis=1)
    
    plt.hist(norms_0, bins=30, alpha=0.7, label='Class 0 (NonFight)', density=True)
    plt.hist(norms_1, bins=30, alpha=0.7, label='Class 1 (Fight)', density=True)
    plt.title('Feature Norm Distribution by Class')
    plt.xlabel('L2 Norm')
    plt.ylabel('Density')
    plt.legend()
    
    # Feature correlation heatmap (first 50 features)
    plt.subplot(2, 3, 4)
    corr_matrix = np.corrcoef(features[:, :50].T)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix (First 50 Features)')
    
    # Features with largest inter-class differences
    plt.subplot(2, 3, 5)
    feature_diff = np.abs(mean_0 - mean_1)
    top_diff_indices = np.argsort(feature_diff)[-20:]  # Top 20 most different features
    
    plt.barh(range(len(top_diff_indices)), feature_diff[top_diff_indices])
    plt.yticks(range(len(top_diff_indices)), [f'F{i}' for i in top_diff_indices])
    plt.title('Top 20 Most Discriminative Features')
    plt.xlabel('Absolute Difference in Mean Values')
    
    # Feature distribution box plot
    plt.subplot(2, 3, 6)
    # Select several representative features
    representative_features = np.argsort(feature_diff)[-5:]  # Select top 5 most different features
    
    data_for_boxplot = []
    labels_for_boxplot = []
    for feat_idx in representative_features:
        data_for_boxplot.extend([features[labels == 0, feat_idx], features[labels == 1, feat_idx]])
        labels_for_boxplot.extend([f'F{feat_idx}_0', f'F{feat_idx}_1'])
    
    plt.boxplot(data_for_boxplot, labels=labels_for_boxplot)
    plt.xticks(rotation=45)
    plt.title('Distribution of Top 5 Discriminative Features')
    plt.ylabel('Feature Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_characteristics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.feature_path}")
    features, labels = load_features(args.feature_path)
    
    print(f"Loaded features: {features.shape}, labels: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Analyze feature distribution
    print("Analyzing feature distribution...")
    analyze_feature_distribution(features, labels, args.output_dir)
    
    # Visualize feature space
    print("Visualizing feature space...")
    visualize_feature_space(features, labels, args.output_dir)
    
    # Analyze class characteristics
    print("Analyzing class characteristics...")
    analyze_class_characteristics(features, labels, args.output_dir)
    
    # Calculate separability metrics
    print("Calculating separability metrics...")
    metrics = calculate_separability_metrics(features, labels)
    
    print("\n=== Feature Analysis Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics
    import json
    with open(f'{args.output_dir}/feature_analysis_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze extracted features')
    parser.add_argument('--feature_path', type=str, required=True, help='Path to features.pth file')
    parser.add_argument('--output_dir', type=str, default='feature_analysis', help='Output directory')
    args = parser.parse_args()
    
    main(args)