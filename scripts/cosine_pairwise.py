from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


EMBEDDINGS_FILE = Path("cosine_similarity_results/all_prompts_with_embeddings.parquet")
OUTPUT_DIR = Path("cosine_pairwise_histograms")

CATEGORIES = [
    "certain",
    "adversarial_nonsense",
    "corrupted",
    "gibberish",
    "nonword",
    "word_salad",
]

UNCERTAIN_CATEGORIES = [
    "adversarial_nonsense",
    "corrupted",
    "gibberish",
    "nonword",
    "word_salad",
]


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_DIR,
        "within": OUTPUT_DIR / "within_category",
        "certain_vs_uncertain": OUTPUT_DIR / "certain_vs_uncertain",
        "uncertain_vs_uncertain": OUTPUT_DIR / "uncertain_vs_uncertain",
        "overlay": OUTPUT_DIR / "overlay_figures",
        "certain_vs_each_separate": OUTPUT_DIR / "certain_vs_each_separate",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    return df, embeddings


def get_category_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    out = {}
    for cat in df["category"].unique():
        idx = df.index[df["category"] == cat].to_numpy()
        out[cat] = embeddings[idx]
    return out


def pairwise_cosine_same_group(a: np.ndarray) -> np.ndarray:
    sims = a @ a.T
    mask = ~np.eye(len(a), dtype=bool)
    return sims[mask]


def pairwise_cosine_between_groups(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sims = a @ b.T
    return sims.ravel()


def make_histogram(
    values: np.ndarray,
    title: str,
    output_path: Path,
    bins: int = 60,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, density=True, alpha=0.8)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_overlay_histogram(
    values_dict: dict[str, np.ndarray],
    title: str,
    output_path: Path,
    bins: int = 60,
) -> None:
    plt.figure(figsize=(10, 6))

    for label, values in values_dict.items():
        plt.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.35,
            label=label,
        )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def make_overlay_histogram_with_ks(
    certain_vals: np.ndarray,
    uncertain_vals: np.ndarray,
    uncertain_name: str,
    output_path: Path,
    bins: int = 60,
) -> None:
    ks_result = ks_2samp(certain_vals, uncertain_vals)
    ks_stat = ks_result.statistic
    p_value = ks_result.pvalue

    plt.figure(figsize=(10, 6))

    plt.hist(
        certain_vals,
        bins=bins,
        density=True,
        alpha=0.5,
        label="certain vs certain",
    )

    plt.hist(
        uncertain_vals,
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"certain vs {uncertain_name}",
    )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(f"Figure B: certain vs certain and certain vs {uncertain_name}")
    plt.legend()

    text = f"KS statistic = {ks_stat:.4f}\np-value = {p_value:.2e}"
    plt.text(
        0.02,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    dirs = ensure_dirs()
    df, embeddings = load_embeddings(EMBEDDINGS_FILE)
    embeddings_by_cat = get_category_embeddings(df, embeddings)

    # ------------------------------------------------------------
    # Individual histograms
    # ------------------------------------------------------------

    for cat in CATEGORIES:
        vals = pairwise_cosine_same_group(embeddings_by_cat[cat])
        make_histogram(
            values=vals,
            title=f"Within-category cosine similarity: {cat} vs {cat}",
            output_path=dirs["within"] / f"{cat}_vs_{cat}.png",
        )

    certain_emb = embeddings_by_cat["certain"]
    for cat in UNCERTAIN_CATEGORIES:
        vals = pairwise_cosine_between_groups(certain_emb, embeddings_by_cat[cat])
        make_histogram(
            values=vals,
            title=f"Cosine similarity: certain vs {cat}",
            output_path=dirs["certain_vs_uncertain"] / f"certain_vs_{cat}.png",
        )

    uncertain_pairs = list(combinations(UNCERTAIN_CATEGORIES, 2))
    for cat_a, cat_b in uncertain_pairs:
        vals = pairwise_cosine_between_groups(
            embeddings_by_cat[cat_a],
            embeddings_by_cat[cat_b],
        )
        make_histogram(
            values=vals,
            title=f"Cosine similarity: {cat_a} vs {cat_b}",
            output_path=dirs["uncertain_vs_uncertain"] / f"{cat_a}_vs_{cat_b}.png",
        )

    # ------------------------------------------------------------
    # Figure A: Within-category overlay
    # ------------------------------------------------------------
    within_values = {}
    for cat in CATEGORIES:
        within_values[f"{cat} vs {cat}"] = pairwise_cosine_same_group(
            embeddings_by_cat[cat]
        )

    make_overlay_histogram(
        values_dict=within_values,
        title="Figure A: Within-category cosine similarity distributions",
        output_path=dirs["overlay"] / "figure_A_within_category_overlay.png",
    )

    # ------------------------------------------------------------
    # Figure B: Separate plots for certain vs each uncertain
    # Each plot overlays:
    #   - certain vs certain
    #   - certain vs one uncertain category
    # and writes KS test results on the plot
    # ------------------------------------------------------------
    certain_vs_certain_vals = pairwise_cosine_same_group(certain_emb)

    for cat in UNCERTAIN_CATEGORIES:
        uncertain_vals = pairwise_cosine_between_groups(
            certain_emb,
            embeddings_by_cat[cat],
        )

        make_overlay_histogram_with_ks(
            certain_vals=certain_vs_certain_vals,
            uncertain_vals=uncertain_vals,
            uncertain_name=cat,
            output_path=dirs["certain_vs_each_separate"] / f"certain_vs_{cat}_overlay.png",
        )

    # ------------------------------------------------------------
    # Figure C, D, E: Uncertain cross-category overlays split
    # ------------------------------------------------------------
    group_c_pairs = [
        ("adversarial_nonsense", "corrupted"),
        ("adversarial_nonsense", "gibberish"),
        ("adversarial_nonsense", "nonword"),
        ("adversarial_nonsense", "word_salad"),
    ]

    group_d_pairs = [
        ("corrupted", "gibberish"),
        ("corrupted", "nonword"),
        ("corrupted", "word_salad"),
    ]

    group_e_pairs = [
        ("gibberish", "nonword"),
        ("gibberish", "word_salad"),
        ("nonword", "word_salad"),
    ]

    group_c_values = {}
    for cat_a, cat_b in group_c_pairs:
        group_c_values[f"{cat_a} vs {cat_b}"] = pairwise_cosine_between_groups(
            embeddings_by_cat[cat_a],
            embeddings_by_cat[cat_b],
        )

    group_d_values = {}
    for cat_a, cat_b in group_d_pairs:
        group_d_values[f"{cat_a} vs {cat_b}"] = pairwise_cosine_between_groups(
            embeddings_by_cat[cat_a],
            embeddings_by_cat[cat_b],
        )

    group_e_values = {}
    for cat_a, cat_b in group_e_pairs:
        group_e_values[f"{cat_a} vs {cat_b}"] = pairwise_cosine_between_groups(
            embeddings_by_cat[cat_a],
            embeddings_by_cat[cat_b],
        )

    make_overlay_histogram(
        values_dict=group_c_values,
        title="Figure C: Uncertain cross-category similarities (group 1)",
        output_path=dirs["overlay"] / "figure_C_uncertain_cross_group1.png",
    )

    make_overlay_histogram(
        values_dict=group_d_values,
        title="Figure D: Uncertain cross-category similarities (group 2)",
        output_path=dirs["overlay"] / "figure_D_uncertain_cross_group2.png",
    )

    make_overlay_histogram(
        values_dict=group_e_values,
        title="Figure E: Uncertain cross-category similarities (group 3)",
        output_path=dirs["overlay"] / "figure_E_uncertain_cross_group3.png",
    )

    print(f"Saved histograms to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()