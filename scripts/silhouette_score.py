from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not installed. Run: pip install umap-learn")

EMBEDDINGS_FILE = Path("cosine_similarity_results/all_prompts_with_embeddings.parquet")
OUTPUT_DIR = Path("embedding_silhouette_scores")

CATEGORIES = [
    "certain",
    "adversarial_nonsense",
    "corrupted",
    "gibberish",
    "nonword",
    "word_salad",
]


def ensure_dirs() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and normalize embeddings."""
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    embeddings = normalize(embeddings, norm="l2")
    return df, embeddings


def sample_balanced(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_per_category: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Balance categories for fair comparison."""
    if n_per_category is None:
        return df.reset_index(drop=True), embeddings
    
    rng = np.random.default_rng(seed)
    keep_idx = []
    for cat in CATEGORIES:
        idx = np.where(df["category"].to_numpy() == cat)[0]
        chosen = rng.choice(idx, size=min(n_per_category, len(idx)), replace=False)
        keep_idx.append(chosen)
    keep_idx = np.concatenate(keep_idx)
    return df.iloc[keep_idx].reset_index(drop=True), embeddings[keep_idx]


def compute_pairwise_silhouette(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    method_name: str,
) -> pd.DataFrame:
    """
    Compute silhouette scores for each pair of categories.
    """
    print(f"\nComputing pairwise silhouette scores for {method_name}...")
    
    unique_labels = np.unique(labels)
    n_categories = len(unique_labels)
    
    pairwise_scores = np.zeros((n_categories, n_categories))
    
    for i, cat1 in enumerate(unique_labels):
        for j, cat2 in enumerate(unique_labels):
            if i <= j:
                continue
            
            # Get indices for these two categories
            mask = (labels == cat1) | (labels == cat2)
            subset_embeddings = embeddings[mask]
            subset_labels = labels[mask]
            
            # Compute silhouette score for this pair
            score = silhouette_score(subset_embeddings, subset_labels, metric='euclidean')
            pairwise_scores[i, j] = score
            pairwise_scores[j, i] = score
    
    # Create DataFrame
    df_pairwise = pd.DataFrame(pairwise_scores, index=unique_labels, columns=unique_labels)
    
    return df_pairwise


def compute_pairwise_for_pca(embeddings: np.ndarray, labels: np.ndarray, n_components: int = 20) -> pd.DataFrame:
    """Reduce with PCA first, then compute pairwise silhouette."""
    print(f"\n{'='*60}")
    print(f"PCA WITH TOP {n_components} COMPONENTS (2D projection)")
    print(f"{'='*60}")
    
    pca = PCA(n_components=2, random_state=42)  # 2D for visualization
    reduced_2d = pca.fit_transform(embeddings)
    ev_ratio = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={ev_ratio[0]*100:.1f}%, PC2={ev_ratio[1]*100:.1f}%")
    print(f"  Total: {sum(ev_ratio)*100:.1f}%")
    
    # Also get top 20 features (components) for reference
    pca_20 = PCA(n_components=min(n_components, embeddings.shape[1]), random_state=42)
    pca_20.fit(embeddings)
    print(f"  Top {n_components} components explain: {pca_20.explained_variance_ratio_.sum()*100:.1f}% of variance")
    
    df_pairwise = compute_pairwise_silhouette(reduced_2d, labels, "PCA-2D")
    return df_pairwise


def compute_pairwise_for_tsne(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    perplexity: int = 30,
) -> pd.DataFrame:
    """Project with t-SNE first, then compute pairwise silhouette."""
    print(f"\n{'='*60}")
    print(f"T-SNE (2D, perplexity={perplexity})")
    print(f"{'='*60}")
    
    n_pca = min(50, embeddings.shape[1])
    pca_pre = PCA(n_components=n_pca, random_state=42)
    reduced_pca = pca_pre.fit_transform(embeddings)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        max_iter=1000,
        init='pca',
        random_state=42,
        n_jobs=1,  
    )
    projected = tsne.fit_transform(reduced_pca)
    
    method_name = f"tSNE-2D_perp{perplexity}"
    df_pairwise = compute_pairwise_silhouette(projected, labels, method_name)
    return df_pairwise


def compute_pairwise_for_umap(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame | None:
    """Project with UMAP first, then compute pairwise silhouette."""
    if not UMAP_AVAILABLE:
        print("\nUMAP not available. Skipping...")
        return None
    
    print(f"\n{'='*60}")
    print(f"UMAP (2D, n_neighbors={n_neighbors}, min_dist={min_dist})")
    print(f"{'='*60}")
    
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        n_jobs=1,  
    )
    projected = umap_reducer.fit_transform(embeddings)
    
    method_name = f"UMAP-2D_nn{n_neighbors}_md{str(min_dist).replace('.','')}"
    df_pairwise = compute_pairwise_silhouette(projected, labels, method_name)
    return df_pairwise


def plot_pairwise_heatmap(df_pairwise: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    """Create heatmap of pairwise silhouette scores."""
    plt.figure(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(df_pairwise), k=1)
    
    sns.heatmap(
        df_pairwise,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-0.5,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Silhouette Score', 'shrink': 0.8}
    )
    
    plt.title(f'Pairwise Silhouette Scores - {method_name}\n(higher = better separation)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'pairwise_silhouette_{method_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap → {output_dir / f'pairwise_silhouette_{method_name}.png'}")


def main():
    out_dir = ensure_dirs()
    
    print("=" * 60)
    print("PAIRWISE SILHOUETTE SCORE ANALYSIS (2D METHODS ONLY)")
    print("=" * 60)
    
    print("\n1. Loading embeddings...")
    df, embeddings = load_embeddings(EMBEDDINGS_FILE)
    print(f"   Loaded {len(df)} rows, embedding dim = {embeddings.shape[1]}")
    
    df_s, emb_s = sample_balanced(df, embeddings, n_per_category=500)
    labels = df_s["category"].to_numpy()
    print(f"   Using {len(df_s)} samples ({len(df_s) // len(CATEGORIES)} per category)")
    
    all_pairwise_dfs = {}
    
    df_pca = compute_pairwise_for_pca(emb_s, labels, n_components=20)
    all_pairwise_dfs['PCA-2D'] = df_pca
    plot_pairwise_heatmap(df_pca, 'PCA-2D', out_dir)
    
    tsne_configs = [30, 50]  # perplexities
    
    for perp in tsne_configs:
        df_tsne = compute_pairwise_for_tsne(emb_s, labels, perplexity=perp)
        if df_tsne is not None:
            method_key = f'tSNE-2D_perp{perp}'
            all_pairwise_dfs[method_key] = df_tsne
            plot_pairwise_heatmap(df_tsne, method_key, out_dir)
    
    if UMAP_AVAILABLE:
        umap_configs = [
            (15, 0.1),   
            (30, 0.1),   
        ]
        
        for nn, md in umap_configs:
            df_umap = compute_pairwise_for_umap(emb_s, labels, n_neighbors=nn, min_dist=md)
            if df_umap is not None:
                method_key = f'UMAP-2D_nn{nn}_md{str(md).replace(".","")}'
                all_pairwise_dfs[method_key] = df_umap
                plot_pairwise_heatmap(df_umap, method_key, out_dir)
    
    print("\n" + "=" * 60)
    print("SAVING ALL TABLES TO CSV")
    print("=" * 60)
    
    all_methods_data = []
    
    for method_name, df_pairwise in all_pairwise_dfs.items():
        csv_path = out_dir / f'pairwise_{method_name}.csv'
        df_pairwise.to_csv(csv_path)
        print(f"  Saved {method_name} → {csv_path}")
        
        for col in df_pairwise.columns:
            for idx in df_pairwise.index:
                if idx != col:  # Skip diagonal
                    all_methods_data.append({
                        'Method': method_name,
                        'Category_1': idx,
                        'Category_2': col,
                        'Silhouette_Score': df_pairwise.loc[idx, col]
                    })
    
    df_combined = pd.DataFrame(all_methods_data)
    df_combined.to_csv(out_dir / 'all_pairwise_scores_combined.csv', index=False)
    print(f"  Saved combined summary → {out_dir / 'all_pairwise_scores_combined.csv'}")
    
    print("\n" + "=" * 60)
    print("SUMMARY: AVERAGE SILHOUETTE SCORES BY METHOD (2D)")
    print("=" * 60)
    
    summary_data = []
    for method_name, df_pairwise in all_pairwise_dfs.items():
        upper_tri = df_pairwise.where(np.triu(np.ones(df_pairwise.shape), k=1).astype(bool))
        scores = upper_tri.stack()
        
        if len(scores) > 0:
            avg_score = scores.mean()
            min_score = scores.min()
            max_score = scores.max()
            
            certain_vs_others = []
            if 'certain' in df_pairwise.index:
                for col in df_pairwise.columns:
                    if col != 'certain':
                        certain_vs_others.append(df_pairwise.loc['certain', col])
            
            avg_certain_sep = np.mean(certain_vs_others) if certain_vs_others else 0
            
            summary_data.append({
                'Method': method_name,
                'Avg_Silhouette': f'{avg_score:.4f}',
                'Min_Worst_Pair': f'{min_score:.4f}',
                'Max_Best_Pair': f'{max_score:.4f}',
                'Certain_vs_Others': f'{avg_certain_sep:.4f}',
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(out_dir / 'method_comparison_summary_2d.csv', index=False)
    print(df_summary.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    certain_separation = []
    for method_name, df_pairwise in all_pairwise_dfs.items():
        if 'certain' in df_pairwise.index:
            certain_scores = [df_pairwise.loc['certain', col] for col in df_pairwise.columns if col != 'certain']
            avg_separation = np.mean(certain_scores)
            certain_separation.append((method_name, avg_separation))
    
    if certain_separation:
        certain_separation.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBest methods for separating 'certain' from other categories:")
        for i, (method, score) in enumerate(certain_separation[:3]):
            print(f"   {i+1}. {method}: {score:.4f}")
    
    print("\nMost overlapping category pairs (lowest silhouette scores):")
    all_pairs = []
    for method_name, df_pairwise in all_pairwise_dfs.items():
        upper_tri = df_pairwise.where(np.triu(np.ones(df_pairwise.shape), k=1).astype(bool))
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                score = upper_tri.loc[idx, col]
                if not np.isnan(score):
                    all_pairs.append((method_name, f"{idx} vs {col}", score))
    
    if all_pairs:
        all_pairs.sort(key=lambda x: x[2])  
        print("   Worst overlaps across all methods:")
        for method, pair, score in all_pairs[:5]:
            print(f"   • {pair}: {score:.4f} ({method})")
    

if __name__ == "__main__":
    main()