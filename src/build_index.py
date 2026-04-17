"""Build and save FAISS vector indexes and SQLite metadata database.

Based on Task 1 evaluation, we chose:
- FAISS for semantic vector search (best retrieval quality)
- SQLite for metadata filtering (enables SQL + semantic hybrid queries)

Boolean search and BM25 were discarded because they performed poorly
on natural language fashion queries (0 results for boolean, wrong
body types and irrelevant keyword matches for BM25).

Run: python -m src.build_index
"""

import os
import sys
import pickle
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure project root is on Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    INDEX_DIR, EMBEDDING_MODEL, EMBEDDING_DIM,
    STYLE_FAISS, PRODUCT_FAISS,
    STYLE_ID_MAP, PRODUCT_ID_MAP,
    STYLE_CORPUS, PRODUCT_CORPUS,
    SQLITE_DB,
)
from src.load_data import (
    load_style_data, load_product_data,
    create_style_docs, create_product_docs,
)


def build_faiss_index(docs, model):
    """Encode documents and build a FAISS IndexFlatIP index.

    Uses normalized embeddings so inner product = cosine similarity.
    Each document is indexed as a single vector (document-level granularity).
    """
    texts = [doc.text for doc in docs]
    embeddings = model.encode(
        texts, show_progress_bar=True,
        batch_size=64, normalize_embeddings=True
    )
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings.astype("float32"))
    id_map = {i: doc.doc_id for i, doc in enumerate(docs)}
    corpus = {doc.doc_id: doc.text for doc in docs}
    return index, id_map, corpus


def build_sqlite(style_df, product_df):
    """Create SQLite database with product and style metadata."""
    if os.path.exists(SQLITE_DB):
        os.remove(SQLITE_DB)

    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()

    c.execute("""CREATE TABLE products (
        doc_id INTEGER PRIMARY KEY,
        product_name TEXT,
        gender TEXT,
        master_category TEXT,
        sub_category TEXT,
        article_type TEXT,
        base_colour TEXT,
        season TEXT,
        usage TEXT,
        year REAL
    )""")

    c.execute("""CREATE TABLE styles (
        doc_id INTEGER PRIMARY KEY,
        body_description TEXT,
        occasion TEXT,
        recommendation TEXT
    )""")

    # Create indexes for common filter columns
    c.execute("CREATE INDEX idx_gender ON products(gender)")
    c.execute("CREATE INDEX idx_article ON products(article_type)")
    c.execute("CREATE INDEX idx_colour ON products(base_colour)")
    c.execute("CREATE INDEX idx_season ON products(season)")
    c.execute("CREATE INDEX idx_occasion ON styles(occasion)")

    # Insert products
    n_products = 0
    for idx, row in product_df.iterrows():
        c.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?)", (
            idx, row.get("productDisplayName", ""), row.get("gender", ""),
            row.get("masterCategory", ""), row.get("subCategory", ""),
            row.get("articleType", ""), row.get("baseColour", ""),
            row.get("season", ""), row.get("usage", ""), row.get("year", None)
        ))
        n_products += 1

    # Insert styles
    n_styles = 0
    for idx, row in style_df.iterrows():
        c.execute("INSERT INTO styles VALUES (?,?,?,?)", (
            idx, row.get("input", ""), row.get("context", ""), row.get("completion", "")
        ))
        n_styles += 1

    conn.commit()
    conn.close()
    return n_products, n_styles


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    print("=" * 60)
    print("STYLEFINDER - Building indexes (FAISS + SQLite)")
    print("=" * 60)

    # Load data
    style_df = load_style_data()
    product_df = load_product_data()
    style_docs = create_style_docs(style_df)
    product_docs = create_product_docs(product_df)

    # 1. FAISS vector indexes
    print("\n--- Building FAISS indexes ---")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("  Encoding style documents...")
    style_index, style_id_map, style_corpus = build_faiss_index(style_docs, model)
    faiss.write_index(style_index, STYLE_FAISS)
    save_pickle(style_id_map, STYLE_ID_MAP)
    save_pickle(style_corpus, STYLE_CORPUS)

    print("  Encoding product documents...")
    product_index, product_id_map, product_corpus = build_faiss_index(product_docs, model)
    faiss.write_index(product_index, PRODUCT_FAISS)
    save_pickle(product_id_map, PRODUCT_ID_MAP)
    save_pickle(product_corpus, PRODUCT_CORPUS)

    print(f"  Style FAISS: {style_index.ntotal} vectors")
    print(f"  Product FAISS: {product_index.ntotal} vectors")

    # 2. SQLite metadata database
    print("\n--- Building SQLite database ---")
    n_prod, n_sty = build_sqlite(style_df, product_df)
    print(f"  Products: {n_prod} rows | Styles: {n_sty} rows")

    # Summary
    print("\n" + "=" * 60)
    print("SAVED FILES")
    print("=" * 60)
    total = 0
    for fname in sorted(os.listdir(INDEX_DIR)):
        size = os.path.getsize(os.path.join(INDEX_DIR, fname))
        total += size
        print(f"  {fname:40s} {size / 1024 / 1024:8.2f} MB")
    print(f"  {'TOTAL':40s} {total / 1024 / 1024:8.2f} MB")
    print("\nDone.")


if __name__ == "__main__":
    main()
