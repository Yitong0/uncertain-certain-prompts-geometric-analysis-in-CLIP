from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer


PROMPT_COLUMN = "prompt_text"


def load_prompts(csv_path: Path, category: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if PROMPT_COLUMN not in df.columns:
        raise ValueError(
            f"'prompt_text' column not found in {csv_path.name}. "
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()
    out["prompt_text"] = (
        out[PROMPT_COLUMN]
        .astype(str)
        .fillna("")
        .str.strip()
    )

    out = out[out["prompt_text"] != ""].copy()
    out["category"] = category
    out["source_file"] = csv_path.name
    out = out.reset_index(drop=True)

    return out


def build_dataset(data_dir: Path) -> pd.DataFrame:
    files = {
        "certain": data_dir / "base_prompts.csv",
        "adversarial_nonsense": data_dir / "adversarial_nonsense_variants.csv",
        "corrupted": data_dir / "corrupted_variants.csv",
        "gibberish": data_dir / "gibberish_variants.csv",
        "nonword": data_dir / "nonword_variants.csv",
        "word_salad": data_dir / "word_salad_variants.csv",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    dfs = []
    for category, path in files.items():
        dfs.append(load_prompts(path, category))

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["row_id"] = np.arange(len(all_df))
    return all_df


@torch.no_grad()
def encode_texts(
    texts: List[str],
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 77,
) -> np.ndarray:
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            text_features = model.get_text_features(**inputs)
        except Exception:
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
            text_features = model.text_projection(text_outputs.pooler_output)

        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = model.text_projection(text_features.pooler_output)
            else:
                raise TypeError(
                    f"Could not extract text embeddings. Got object of type: {type(text_features)}"
                )

        # Normalize so cosine similarity becomes dot product
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_features.cpu().numpy())

    return np.vstack(all_embeddings)


def centroid(a: np.ndarray) -> np.ndarray:
    c = a.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm == 0:
        return c
    return c / norm


def centroid_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ca = centroid(a)
    cb = centroid(b)
    return float(np.dot(ca, cb))


def make_between_group_summary(
    embeddings_by_cat: Dict[str, np.ndarray],
    categories: List[str],
) -> pd.DataFrame:
    rows = []

    for cat_a, cat_b in combinations_with_replacement(categories, 2):
        a = embeddings_by_cat[cat_a]
        b = embeddings_by_cat[cat_b]

        rows.append(
            {
                "category_a": cat_a,
                "category_b": cat_b,
                "n_a": len(a),
                "n_b": len(b),
                "centroid_cosine": centroid_similarity(a, b),
            }
        )

    return pd.DataFrame(rows)


def save_embeddings_table(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_path: Path,
) -> None:
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)

    out = pd.concat(
        [
            df[["row_id", "category", "source_file", "prompt_text"]].reset_index(drop=True),
            emb_df,
        ],
        axis=1,
    )

    out.to_parquet(output_path, index=False)


def save_centroids(
    centroids: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    rows = []
    for cat, vec in centroids.items():
        row = {"category": cat}
        for i, value in enumerate(vec):
            row[f"centroid_{i}"] = float(value)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = build_dataset(data_dir)
    print(f"Loaded {len(df)} prompts.")

    # Load CLIP model
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()

    # Encode prompts
    texts = df["prompt_text"].tolist()
    embeddings = encode_texts(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Save prompt embeddings
    save_embeddings_table(
        df=df,
        embeddings=embeddings,
        output_path=output_dir / "all_prompts_with_embeddings.parquet",
    )

    # Group embeddings by category
    embeddings_by_cat: Dict[str, np.ndarray] = {}
    for cat, group in df.groupby("category"):
        idx = group.index.to_numpy()
        embeddings_by_cat[cat] = embeddings[idx]

    uncertain_cats = [
        "adversarial_nonsense",
        "corrupted",
        "gibberish",
        "nonword",
        "word_salad",
    ]

    certain = embeddings_by_cat["certain"]
    uncertain_all = np.vstack([embeddings_by_cat[c] for c in uncertain_cats])

    # Build and save centroids
    centroids: Dict[str, np.ndarray] = {}
    for cat, emb in embeddings_by_cat.items():
        centroids[cat] = centroid(emb)
    centroids["uncertain_all"] = centroid(uncertain_all)

    save_centroids(
        centroids=centroids,
        output_path=output_dir / "category_centroids.csv",
    )

    # Certain vs all uncertain summary
    overall_summary = {
        "certain_count": int(len(certain)),
        "uncertain_total_count": int(len(uncertain_all)),
        "certain_vs_all_uncertain_centroid_cosine": centroid_similarity(
            certain, uncertain_all
        ),
    }

    with open(output_dir / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    # Certain vs certain + each uncertain category
    certain_vs_each_rows = []
    certain_vs_each_categories = ["certain"] + uncertain_cats

    for cat_b in certain_vs_each_categories:
        certain_vs_each_rows.append(
            {
                "category_a": "certain",
                "category_b": cat_b,
                "n_a": len(embeddings_by_cat["certain"]),
                "n_b": len(embeddings_by_cat[cat_b]),
                "centroid_cosine": centroid_similarity(
                    embeddings_by_cat["certain"],
                    embeddings_by_cat[cat_b],
                ),
            }
        )

    certain_vs_each = pd.DataFrame(certain_vs_each_rows)
    certain_vs_each.to_csv(
        output_dir / "certain_vs_each_uncertain_category.csv",
        index=False,
    )

    # Uncertain cross-category similarity: unique unordered pairs only
    uncertain_cross = make_between_group_summary(
        embeddings_by_cat=embeddings_by_cat,
        categories=uncertain_cats,
    )
    uncertain_cross.to_csv(
        output_dir / "uncertain_cross_category_similarity.csv",
        index=False,
    )

    print("Done.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()