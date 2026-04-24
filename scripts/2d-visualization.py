from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not installed. Run: pip install umap-learn")

EMBEDDINGS_FILE = Path("cosine_similarity_results/all_prompts_with_embeddings.parquet")
OUTPUT_DIR = Path("embedding_2d_visualizations")

CATEGORIES = [
    "certain",
    "adversarial_nonsense",
    "corrupted",
    "gibberish",
    "nonword",
    "word_salad",
]

# Colorblind-friendly palette
PALETTE = {
    "certain":               "#2196F3",  # blue
    "adversarial_nonsense":  "#F44336",  # red
    "corrupted":             "#FF9800",  # orange
    "gibberish":             "#9C27B0",  # purple
    "nonword":               "#4CAF50",  # green
    "word_salad":            "#795548",  # brown
}

ALPHA = 0.35
POINT_SIZE = 6


def ensure_dirs() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    # L2-normalize so cosine similarity == dot product (consistent with your pipeline)
    embeddings = normalize(embeddings, norm="l2")
    return df, embeddings


def sample_balanced(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_per_category: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Downsample to n_per_category per class so t-SNE/UMAP stay fast
    and no single class dominates. Set n_per_category=None to use all data.
    """
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


def scatter_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    explained_var: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for cat in CATEGORIES:
        mask = labels == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=PALETTE[cat],
            s=POINT_SIZE,
            alpha=ALPHA,
            linewidths=0,
            rasterized=True,
        )

    legend_handles = [
        mpatches.Patch(color=PALETTE[cat], label=cat) for cat in CATEGORIES
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=9,
        markerscale=2,
        framealpha=0.85,
        loc="best",
    )

    xlabel = "PC 1" if explained_var else "Dim 1"
    ylabel = "PC 2" if explained_var else "Dim 2"
    if explained_var:
        xlabel += f"  ({explained_var[0]*100:.1f}% var)"
        ylabel += f"  ({explained_var[1]*100:.1f}% var)"

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# PCA
def run_pca(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
) -> None:
    print("Running PCA …")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    ev = pca.explained_variance_ratio_

    scatter_2d(
        coords=coords,
        labels=df["category"].to_numpy(),
        title="PCA – 2D projection of prompt embeddings",
        output_path=out_dir / "pca_2d.png",
        explained_var=(ev[0], ev[1]),
    )

    # Also save a variance scree plot for the top 20 components
    pca_full = PCA(n_components=min(20, embeddings.shape[1]), random_state=42)
    pca_full.fit(embeddings)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_ * 100,
        color="#2196F3",
        alpha=0.8,
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot (top 20 components)")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree.png", dpi=200)
    plt.close(fig)
    print(f"  Saved → {out_dir / 'pca_scree.png'}")


# t-sne

def run_tsne(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
    perplexities: list[int] | None = None,
) -> None:
    if perplexities is None:
        perplexities = [30, 50]

    # Reduce to 50 dims with PCA first — standard practice for speed & stability
    n_pca = min(50, embeddings.shape[1])
    print(f"  Pre-reducing to {n_pca} dims with PCA before t-SNE …")
    pca_pre = PCA(n_components=n_pca, random_state=42)
    reduced = pca_pre.fit_transform(embeddings)

    labels = df["category"].to_numpy()

    for perp in perplexities:
        print(f"Running t-SNE (perplexity={perp}) …")
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=1000,
            learning_rate="auto",
            init="pca",
            random_state=42,
            n_jobs=-1,
        )
        coords = tsne.fit_transform(reduced)
        scatter_2d(
            coords=coords,
            labels=labels,
            title=f"t-SNE – 2D projection of prompt embeddings  (perplexity={perp})",
            output_path=out_dir / f"tsne_perp{perp}_2d.png",
        )


# UMAP

def run_umap(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
    configs: list[dict] | None = None,
) -> None:
    if not UMAP_AVAILABLE:
        print("Skipping UMAP (not installed).")
        return

    if configs is None:
        configs = [
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 30, "min_dist": 0.1},
        ]

    labels = df["category"].to_numpy()

    for cfg in configs:
        nn = cfg["n_neighbors"]
        md = cfg["min_dist"]
        print(f"Running UMAP (n_neighbors={nn}, min_dist={md}) …")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=md,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        scatter_2d(
            coords=coords,
            labels=labels,
            title=f"UMAP – 2D projection of prompt embeddings  (n_neighbors={nn}, min_dist={md})",
            output_path=out_dir / f"umap_nn{nn}_md{str(md).replace('.','')}_2d.png",
        )



def scatter_certain_vs_uncertain(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    certain_mask = labels == "certain"
    uncertain_mask = ~certain_mask

    ax.scatter(
        coords[uncertain_mask, 0],
        coords[uncertain_mask, 1],
        c="#F44336",
        s=POINT_SIZE,
        alpha=ALPHA,
        linewidths=0,
        rasterized=True,
        label="uncertain (all)",
    )
    ax.scatter(
        coords[certain_mask, 0],
        coords[certain_mask, 1],
        c="#2196F3",
        s=POINT_SIZE,
        alpha=min(ALPHA * 1.5, 0.7),
        linewidths=0,
        rasterized=True,
        label="certain",
    )

    ax.legend(fontsize=10, markerscale=3, framealpha=0.85)
    ax.set_xlabel("Dim 1", fontsize=11)
    ax.set_ylabel("Dim 2", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def run_certain_vs_uncertain_overlays(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
) -> None:
    labels = df["category"].to_numpy()
    n_pca = min(50, embeddings.shape[1])
    pca_pre = PCA(n_components=n_pca, random_state=42)
    reduced = pca_pre.fit_transform(embeddings)

    # PCA
    pca2 = PCA(n_components=2, random_state=42)
    coords_pca = pca2.fit_transform(embeddings)
    scatter_certain_vs_uncertain(
        coords=coords_pca,
        labels=labels,
        title="PCA – Certain vs All Uncertain",
        output_path=out_dir / "certain_vs_uncertain_pca.png",
    )

    # t-SNE
    print("Running t-SNE for certain vs uncertain overlay …")
    coords_tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
        random_state=42,
        n_jobs=-1,
    ).fit_transform(reduced)
    scatter_certain_vs_uncertain(
        coords=coords_tsne,
        labels=labels,
        title="t-SNE – Certain vs All Uncertain",
        output_path=out_dir / "certain_vs_uncertain_tsne.png",
    )

    # UMAP
    if UMAP_AVAILABLE:
        print("Running UMAP for certain vs uncertain overlay …")
        coords_umap = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings)
        scatter_certain_vs_uncertain(
            coords=coords_umap,
            labels=labels,
            title="UMAP – Certain vs All Uncertain",
            output_path=out_dir / "certain_vs_uncertain_umap.png",
        )



def main() -> None:
    out_dir = ensure_dirs()

    print("Loading embeddings …")
    df, embeddings = load_embeddings(EMBEDDINGS_FILE)
    print(f"  Loaded {len(df)} rows, embedding dim = {embeddings.shape[1]}")

    # Downsample for t-SNE / UMAP speed (set None to use all)
    df_s, emb_s = sample_balanced(df, embeddings, n_per_category=500)
    print(f"  Using {len(df_s)} samples ({len(df_s) // len(CATEGORIES)} per category)")

    run_pca(df_s, emb_s, out_dir)
    run_tsne(df_s, emb_s, out_dir, perplexities=[30, 50])
    run_umap(df_s, emb_s, out_dir)
    run_certain_vs_uncertain_overlays(df_s, emb_s, out_dir)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()