"""Shared paths and constants for StyleFinder."""

import os

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

# FAISS indexes
STYLE_FAISS = os.path.join(INDEX_DIR, "style_faiss.index")
PRODUCT_FAISS = os.path.join(INDEX_DIR, "product_faiss.index")
STYLE_ID_MAP = os.path.join(INDEX_DIR, "style_id_map.pkl")
PRODUCT_ID_MAP = os.path.join(INDEX_DIR, "product_id_map.pkl")
STYLE_CORPUS = os.path.join(INDEX_DIR, "style_corpus.pkl")
PRODUCT_CORPUS = os.path.join(INDEX_DIR, "product_corpus.pkl")

# SQLite
SQLITE_DB = os.path.join(INDEX_DIR, "stylefinder.db")

# Evaluation
EVAL_SET = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_set.json")

# Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# HuggingFace datasets
STYLE_DATASET = "neuralwork/fashion-style-instruct"
PRODUCT_DATASET = "ashraq/fashion-product-images-small"
