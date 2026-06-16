"""Resolve a `--features` selector into ordered feature columns.

Shared by `nat viz render` (15m_visualize.py) and `nat viz3d` (viz_mesh.py) so
both accept the same selector grammar: a category name, a named FEATURE_VECTORS
vector, a comma-list of columns, or 'all'/None (every numeric feature). Columns
are returned ordered by category so panels / the y-axis group sensibly.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

META_COLS = {"timestamp_ns", "symbol", "sequence_id", "datetime"}


def category_map():
    """Return (ordered_categories, {column: category}) from the schema."""
    try:
        from data.schema import BASE_FEATURES, OPTIONAL_FEATURES
    except Exception:
        return [], {}
    cats, col2cat = [], {}
    for src in (BASE_FEATURES, OPTIONAL_FEATURES):
        for cat, cols in src.items():
            cats.append(cat)
            for c in cols:
                col2cat[c] = cat
    return cats, col2cat


def select_features(df: pd.DataFrame, features: Optional[str]):
    """Resolve ``features`` into an ordered list of numeric feature columns.

    Raises ValueError if an explicit selector matches no columns.
    """
    numeric = [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    cats, col2cat = category_map()

    chosen = None
    if features and features != "all":
        if "," in features:
            chosen = [c for c in features.split(",") if c in df.columns]
        elif col2cat and features in set(col2cat.values()):
            chosen = [c for c in numeric if col2cat.get(c) == features]
        else:
            try:
                from cluster_pipeline.config import FEATURE_VECTORS, get_vector_columns
                if features in FEATURE_VECTORS:
                    chosen = [c for c in get_vector_columns(features) if c in numeric]
            except Exception:
                chosen = None
        if not chosen:
            raise ValueError(f"--features '{features}' matched no columns")
    else:
        chosen = list(numeric)

    # Order by category (grouped), stable within category.
    order = {cat: i for i, cat in enumerate(cats)}
    chosen.sort(key=lambda c: (order.get(col2cat.get(c), 10_000), c))
    return chosen


def cap_by_variance(df: pd.DataFrame, cols: list, max_features: int):
    """Drop all-NaN columns, then cap to the top-N by variance (preserving the
    incoming category order). Returns (kept_cols, capped: bool)."""
    present = [c for c in cols if c in df.columns and not df[c].isna().all()]
    if max_features and len(present) > max_features:
        top = (
            df[present].var(numeric_only=True)
            .sort_values(ascending=False)
            .head(max_features)
            .index
        )
        keep = set(top)
        return [c for c in present if c in keep], True
    return present, False
