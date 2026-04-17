"""Search engine for StyleFinder.

Loads FAISS and SQLite indexes from disk and provides:
- faiss_search: pure semantic search
- filtered_search: SQL metadata filter + FAISS semantic ranking
- stylefinder: two-layer pipeline (style advice -> product matching)

Run:
  python -m src.search "hourglass figure elegant evening dress"
  python -m src.search   (interactive mode)
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
    EMBEDDING_MODEL,
    STYLE_FAISS, PRODUCT_FAISS,
    STYLE_ID_MAP, PRODUCT_ID_MAP,
    STYLE_CORPUS, PRODUCT_CORPUS,
    SQLITE_DB,
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class StyleFinderEngine:
    """Loads FAISS + SQLite indexes from disk and provides search."""

    def __init__(self):
        print("Loading indexes from disk...")

        self.style_faiss = faiss.read_index(STYLE_FAISS)
        self.product_faiss = faiss.read_index(PRODUCT_FAISS)
        self.style_id_map = load_pickle(STYLE_ID_MAP)
        self.product_id_map = load_pickle(PRODUCT_ID_MAP)
        self.style_corpus = load_pickle(STYLE_CORPUS)
        self.product_corpus = load_pickle(PRODUCT_CORPUS)

        self.conn = sqlite3.connect(SQLITE_DB)
        self.cursor = self.conn.cursor()

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  Style: {self.style_faiss.ntotal} vectors")
        print(f"  Product: {self.product_faiss.ntotal} vectors")
        print("Ready.\n")

    def faiss_search(self, query, layer="product", n=5):
        """Semantic search using FAISS."""
        faiss_idx = self.style_faiss if layer == "style" else self.product_faiss
        id_map = self.style_id_map if layer == "style" else self.product_id_map
        corpus = self.style_corpus if layer == "style" else self.product_corpus

        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = faiss_idx.search(q_emb, n)

        return [{"rank": i + 1, "doc_id": id_map[idx], "score": float(score),
                 "text": corpus[id_map[idx]][:150]}
                for i, (score, idx) in enumerate(zip(scores[0], indices[0]))]

    def filtered_search(self, query, gender=None, article_type=None,
                        colour=None, season=None, n=5):
        """SQL metadata filter + FAISS semantic ranking.

        First narrows the candidate set using SQL, then ranks
        the filtered subset by semantic similarity.
        """
        sql = "SELECT doc_id FROM products WHERE 1=1"
        params = []
        if gender:
            sql += " AND gender = ?"; params.append(gender)
        if article_type:
            sql += " AND article_type = ?"; params.append(article_type)
        if colour:
            sql += " AND base_colour = ?"; params.append(colour)
        if season:
            sql += " AND season = ?"; params.append(season)

        self.cursor.execute(sql, params)
        filtered_ids = [row[0] for row in self.cursor.fetchall()]
        if not filtered_ids:
            return []

        # Reconstruct embeddings for filtered products from FAISS
        filtered_embs = np.vstack([
            self.product_faiss.reconstruct(fid) for fid in filtered_ids
        ]).astype("float32")

        temp_index = faiss.IndexFlatIP(filtered_embs.shape[1])
        temp_index.add(filtered_embs)

        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = temp_index.search(q_emb, min(n, len(filtered_ids)))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            orig_id = filtered_ids[idx]
            self.cursor.execute(
                "SELECT product_name, article_type, base_colour, season, usage "
                "FROM products WHERE doc_id = ?", (orig_id,)
            )
            row = self.cursor.fetchone()
            results.append({
                "rank": rank + 1, "doc_id": str(orig_id), "score": float(score),
                "name": row[0], "type": row[1], "colour": row[2],
                "season": row[3], "usage": row[4]
            })
        return results

    def stylefinder(self, query, n_styles=3, n_products=5):
        """Two-layer search: style recommendations + product matching."""
        print(f"\n{'=' * 70}")
        print(f"STYLEFINDER QUERY: {query}")
        print(f"{'=' * 70}")

        # Layer 1: Style recommendations
        print("\n--- LAYER 1: Style Recommendations ---")
        style_results = self.faiss_search(query, layer="style", n=n_styles)
        for r in style_results:
            print(f"  Rank {r['rank']} (score: {r['score']:.4f}): {r['text']}...")

        # Layer 2: Product matching
        print(f"\n--- LAYER 2: Matching Products ---")
        best_style_text = self.style_corpus[style_results[0]["doc_id"]][:300]
        product_query = query + " " + best_style_text
        product_results = self.faiss_search(product_query, layer="product", n=n_products)
        for r in product_results:
            print(f"  Rank {r['rank']} (score: {r['score']:.4f}): {r['text']}")

        return style_results, product_results

    def close(self):
        self.conn.close()


def main():
    engine = StyleFinderEngine()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        engine.stylefinder(query)
    else:
        print("StyleFinder interactive mode. Type 'quit' to exit.\n")
        while True:
            query = input("Query> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                engine.stylefinder(query)

    engine.close()


if __name__ == "__main__":
    main()
