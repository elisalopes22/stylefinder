"""Load datasets from HuggingFace and prepare documents."""

import os
import sys
from collections import namedtuple
from datasets import load_dataset
import pandas as pd

# Ensure project root is on Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import STYLE_DATASET, PRODUCT_DATASET

Doc = namedtuple("Doc", ["doc_id", "text"])


def load_style_data():
    """Load the fashion style instruct dataset."""
    print("Loading style dataset...")
    dataset = load_dataset(STYLE_DATASET, split="train")
    df = dataset.to_pandas()
    print(f"  Loaded: {df.shape}")
    return df


def load_product_data():
    """Load the fashion product catalog."""
    print("Loading product dataset...")
    dataset = load_dataset(PRODUCT_DATASET, split="train")
    df = dataset.to_pandas()
    df = df.dropna(subset=["productDisplayName", "articleType"]).reset_index(drop=True)
    df = df[df['gender'].isin(['Men', 'Women', 'Unisex'])].reset_index(drop=True)  # <-- ADD THIS LINE
    print(f"  Loaded: {df.shape}")
    return df


def create_style_docs(style_df):
    """Create Doc namedtuples from style dataframe."""
    docs = []
    for idx, row in style_df.iterrows():
        text = f"Body: {row['input']}. Occasion: {row['context']}. Recommendation: {row['completion']}"
        docs.append(Doc(doc_id=str(idx), text=text))
    return docs


def create_product_docs(product_df):
    """Create Doc namedtuples from product dataframe.

    Combines all text metadata fields into a single searchable string.
    Images exist in the dataset but indexing is text-based only.
    """
    docs = []
    fields = ["gender", "masterCategory", "subCategory", "articleType",
              "baseColour", "season", "usage"]
    for idx, row in product_df.iterrows():
        parts = [str(row[f]) for f in fields if pd.notna(row.get(f))]
        name = str(row["productDisplayName"])
        text = " ".join(parts) + " - " + name
        docs.append(Doc(doc_id=str(idx), text=text))
    return docs


if __name__ == "__main__":
    style_df = load_style_data()
    product_df = load_product_data()
    style_docs = create_style_docs(style_df)
    product_docs = create_product_docs(product_df)
    print(f"\nStyle docs: {len(style_docs)}")
    print(f"Product docs: {len(product_docs)}")
    print(f"\nStyle example:\n  {style_docs[0].text[:200]}...")
    print(f"\nProduct example:\n  {product_docs[0].text}")
