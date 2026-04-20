from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
OUTPUT_DIR = Path("embedding_3d_visualizations")

CATEGORIES = [
    "certain",
    "adversarial_nonsense",
    "corrupted",
    "gibberish",
    "nonword",
    "word_salad",
]

PALETTE = {
    "certain":              "#2196F3",
    "adversarial_nonsense": "#F44336",
    "corrupted":            "#FF9800",
    "gibberish":            "#9C27B0",
    "nonword":              "#4CAF50",
    "word_salad":           "#795548",
}

MARKER_SIZE = 3
MARKER_OPACITY = 0.55


def ensure_dirs() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
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


def build_3d_figure(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    axis_labels: tuple[str, str, str],
) -> go.Figure:
    """Build a Plotly 3D scatter with one trace per category."""
    fig = go.Figure()

    for cat in CATEGORIES:
        mask = labels == cat
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=cat,
                marker=dict(
                    size=MARKER_SIZE,
                    color=PALETTE[cat],
                    opacity=MARKER_OPACITY,
                    line=dict(width=0),
                ),
                hovertemplate=f"<b>{cat}</b><br>"
                              f"{axis_labels[0]}: %{{x:.3f}}<br>"
                              f"{axis_labels[1]}: %{{y:.3f}}<br>"
                              f"{axis_labels[2]}: %{{z:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            xaxis=dict(showbackground=False, gridcolor="#e0e0e0"),
            yaxis=dict(showbackground=False, gridcolor="#e0e0e0"),
            zaxis=dict(showbackground=False, gridcolor="#e0e0e0"),
        ),
        legend=dict(
            title="Category",
            itemsizing="constant",
            font=dict(size=12),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        width=1000,
        height=750,
    )
    return fig


# PCA
def run_pca_3d(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
) -> None:
    print("Running PCA 3D …")
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(embeddings)
    ev = pca.explained_variance_ratio_

    axis_labels = (
        f"PC1 ({ev[0]*100:.1f}% var)",
        f"PC2 ({ev[1]*100:.1f}% var)",
        f"PC3 ({ev[2]*100:.1f}% var)",
    )

    fig = build_3d_figure(
        coords=coords,
        labels=df["category"].to_numpy(),
        title=f"PCA 3D — prompt embeddings  "
              f"(total variance explained: {sum(ev)*100:.1f}%)",
        axis_labels=axis_labels,
    )

    out_path = out_dir / "pca_3d.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Saved → {out_path}")


# t-SNE

def run_tsne_3d(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
    perplexities: list[int] | None = None,
    learning_rate: float | str = "auto",
    max_iter: int = 1000,
) -> None:
    """
    Run t-SNE in 3D with optional perplexity values.
    Uses PCA pre-reduction to 50 dims for speed and stability.
    """
    if perplexities is None:
        perplexities = [30, 50]

    # Pre-reduce to 50 dims with PCA (standard practice for t-SNE)
    n_pca = min(50, embeddings.shape[1])
    print(f"  Pre-reducing to {n_pca} dims with PCA before t-SNE …")
    pca_pre = PCA(n_components=n_pca, random_state=42)
    reduced = pca_pre.fit_transform(embeddings)

    labels = df["category"].to_numpy()

    for perp in perplexities:
        print(f"Running t-SNE 3D (perplexity={perp}) …")
        
        # t-SNE in 3D
        tsne = TSNE(
            n_components=3,  # 3D!
            perplexity=perp,
            learning_rate=learning_rate,
            max_iter=max_iter,
            init="pca",
            random_state=42,
            n_jobs=-1,
        )
        coords = tsne.fit_transform(reduced)

        fig = build_3d_figure(
            coords=coords,
            labels=labels,
            title=f"t-SNE 3D — prompt embeddings  (perplexity={perp})",
            axis_labels=("t-SNE 1", "t-SNE 2", "t-SNE 3"),
        )

        out_path = out_dir / f"tsne_3d_perp{perp}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  Saved → {out_path}")


# UMAP

def run_umap_3d(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    if not UMAP_AVAILABLE:
        print("Skipping UMAP 3D (umap-learn not installed).")
        return

    print(f"Running UMAP 3D (n_neighbors={n_neighbors}, min_dist={min_dist}) …")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    fig = build_3d_figure(
        coords=coords,
        labels=df["category"].to_numpy(),
        title=f"UMAP 3D — prompt embeddings  "
              f"(n_neighbors={n_neighbors}, min_dist={min_dist})",
        axis_labels=("UMAP 1", "UMAP 2", "UMAP 3"),
    )

    fname = f"umap_3d_nn{n_neighbors}_md{str(min_dist).replace('.','')}.html"
    out_path = out_dir / fname
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Saved → {out_path}")



def build_certain_vs_uncertain_3d(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    axis_labels: tuple[str, str, str],
) -> go.Figure:
    fig = go.Figure()
    groups = {
        "uncertain (all)": (labels != "certain", "#F44336"),
        "certain":         (labels == "certain",  "#2196F3"),
    }
    for name, (mask, colour) in groups.items():
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=name,
                marker=dict(size=MARKER_SIZE, color=colour, opacity=MARKER_OPACITY, line=dict(width=0)),
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        ),
        legend=dict(itemsizing="constant", font=dict(size=13)),
        margin=dict(l=0, r=0, t=50, b=0),
        width=1000,
        height=750,
    )
    return fig


def main() -> None:
    out_dir = ensure_dirs()

    print("Loading embeddings …")
    df, embeddings = load_embeddings(EMBEDDINGS_FILE)
    print(f"  Loaded {len(df)} rows, embedding dim = {embeddings.shape[1]}")

    df_s, emb_s = sample_balanced(df, embeddings, n_per_category=500)
    print(f"  Using {len(df_s)} samples ({len(df_s) // len(CATEGORIES)} per category)")
    
    # pca w all 6 categories
    run_pca_3d(df_s, emb_s, out_dir)

    # pca certain vs uncertain
    print("Running PCA 3D certain vs uncertain …")
    pca3 = PCA(n_components=3, random_state=42)
    coords_pca = pca3.fit_transform(emb_s)
    ev = pca3.explained_variance_ratio_
    fig_pca_cvu = build_certain_vs_uncertain_3d(
        coords=coords_pca,
        labels=df_s["category"].to_numpy(),
        title="PCA 3D — Certain vs All Uncertain",
        axis_labels=(
            f"PC1 ({ev[0]*100:.1f}%)",
            f"PC2 ({ev[1]*100:.1f}%)",
            f"PC3 ({ev[2]*100:.1f}%)",
        ),
    )
    p = out_dir / "pca_3d_certain_vs_uncertain.html"
    fig_pca_cvu.write_html(str(p), include_plotlyjs="cdn")
    print(f"  Saved → {p}")

    # t-sne w all 6 categories
    run_tsne_3d(df_s, emb_s, out_dir, perplexities=[30, 50])

    # t-sne certain vs uncertain
    print("Running t-SNE 3D certain vs uncertain …")
    # Pre-reduce to 50 dims with PCA
    n_pca = min(50, emb_s.shape[1])
    pca_pre = PCA(n_components=n_pca, random_state=42)
    reduced = pca_pre.fit_transform(emb_s)
    
    tsne = TSNE(
        n_components=3,
        perplexity=30,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=42,
        n_jobs=-1,
    )
    coords_tsne = tsne.fit_transform(reduced)
    
    fig_tsne_cvu = build_certain_vs_uncertain_3d(
        coords=coords_tsne,
        labels=df_s["category"].to_numpy(),
        title="t-SNE 3D — Certain vs All Uncertain (perp=30)",
        axis_labels=("t-SNE 1", "t-SNE 2", "t-SNE 3"),
    )
    p = out_dir / "tsne_3d_certain_vs_uncertain.html"
    fig_tsne_cvu.write_html(str(p), include_plotlyjs="cdn")
    print(f"  Saved → {p}")

    # UMAP w all 6 categories
    run_umap_3d(df_s, emb_s, out_dir, n_neighbors=15, min_dist=0.1)

    # UMAP certain vs uncertain
    if UMAP_AVAILABLE:
        print("Running UMAP 3D certain vs uncertain …")
        reducer = umap.UMAP(
            n_components=3, n_neighbors=15, min_dist=0.1,
            metric="cosine", random_state=42,
        )
        coords_umap = reducer.fit_transform(emb_s)
        fig_umap_cvu = build_certain_vs_uncertain_3d(
            coords=coords_umap,
            labels=df_s["category"].to_numpy(),
            title="UMAP 3D — Certain vs All Uncertain",
            axis_labels=("UMAP 1", "UMAP 2", "UMAP 3"),
        )
        p = out_dir / "umap_3d_certain_vs_uncertain.html"
        fig_umap_cvu.write_html(str(p), include_plotlyjs="cdn")
        print(f"  Saved → {p}")

    print(f"\nAll interactive plots saved to: {OUTPUT_DIR}")
    print("Open any .html file in your browser to rotate and explore.")
    print("\nGenerated files:")
    print("  - pca_3d.html + pca_3d_certain_vs_uncertain.html")
    print("  - tsne_3d_perp30.html + tsne_3d_perp50.html + tsne_3d_certain_vs_uncertain.html")
    print("  - umap_3d_nn15_md01.html + umap_3d_certain_vs_uncertain.html")


if __name__ == "__main__":
    main()