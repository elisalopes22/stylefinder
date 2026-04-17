# StyleFinder

A two-layer fashion Information Retrieval system.

- **Layer 1:** Expert style recommendations based on body type and occasion (3,193 entries)
- **Layer 2:** Product matching from a catalog of 44,000+ fashion items

## Task 1: Exploration and Method Comparison

See [`notebooks/Task1_Exploration.ipynb`](notebooks/Task1_Exploration.ipynb) for the initial exploration where we compared Boolean search, BM25, and Semantic search. Key findings:
- Boolean search returned 0 results for all 10 queries
- BM25 matched keywords literally (e.g., "waist pouches" for "cinches at the waist")
- Semantic search (Sentence Transformers) significantly outperformed both methods

These results informed our index choices for Task 2.

## Task 2: Persistent Indexing (this project)

Based on Task 1 findings, we persist our indexes using:
- **FAISS** for semantic vector search (best retrieval quality)
- **SQLite** for metadata filtering (enables hybrid SQL + semantic queries)

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Build indexes (first time only)
python -m src.build_index

# Step 2: Search
python -m src.search "hourglass figure elegant evening dress"
python -m src.search   # interactive mode

# Step 3: Run evaluation
python -m src.evaluate
```

## Project Structure

```
stylefinder/
├── notebooks/
│   └── Task1_Exploration.ipynb   # Task 1: BM25 vs Semantic comparison
├── src/
│   ├── config.py                 # Paths and constants
│   ├── load_data.py              # Load datasets from HuggingFace
│   ├── build_index.py            # Build FAISS + SQLite indexes
│   ├── search.py                 # Search engine + CLI
│   └── evaluate.py               # Run 15 evaluation queries
├── indexes/                      # Persisted index files (gitignored)
├── evaluation_set.json           # Queries with expected doc IDs
├── requirements.txt
└── README.md
```

## Index Methods

Based on our Task 1 evaluation, we chose two complementary index methods:

| Method | Storage | Purpose |
|--------|---------|---------|
| FAISS | `.index` file | Semantic vector search (cosine similarity) |
| SQLite | `.db` file | Metadata filtering (gender, color, category, season) |

Boolean search and BM25 were evaluated in Task 1 but discarded: boolean returned 0 results for all 10 queries, and BM25 produced irrelevant matches (e.g., waist pouches for "dress that cinches at the waist").

The combination of SQLite filtering + FAISS ranking enables hybrid queries: first narrow candidates by metadata (e.g., "only women's black dresses"), then rank by semantic similarity.

## Indexing Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Granularity** | Whole document | Style entries are coherent units (body + occasion + outfits). Products are short metadata strings. Splitting would lose context. |
| **Text parsing** | Lowercase + whitespace split (for BM25 baseline); full text for embeddings | Embeddings handle the semantic understanding; no need for stemming or n-grams. |
| **Product text** | Concatenation of metadata fields: `gender + category + articleType + colour + season + usage + productName` | Creates a rich searchable string from structured fields. Example: `"Women Apparel Dress Dresses Black Fall Formal - Avirate Black Formal Dress"` |
| **Style text** | Concatenation: `"Body: {input}. Occasion: {context}. Recommendation: {completion}"` | Preserves the full context: who the recommendation is for, when, and what. |
| **Embedding model** | `all-MiniLM-L6-v2` (384 dimensions) | Lightweight, fast, good quality. Same model used in course Lab 2. |
| **FAISS index type** | `IndexFlatIP` on normalized vectors | Inner product on L2-normalized vectors equals cosine similarity. Exact search, no approximation. |
| **SQLite metadata** | Products: gender, masterCategory, subCategory, articleType, baseColour, season, usage, year. Styles: body description, occasion, recommendation. | Enables pre-filtering before semantic search. Indexed columns: gender, article_type, base_colour, season, occasion. |
| **Image handling** | Text-based indexing only | Product images exist in the dataset but are used only as visual aids in output. All retrieval is performed on text fields. |

## Datasets

- [neuralwork/fashion-style-instruct](https://huggingface.co/datasets/neuralwork/fashion-style-instruct) — Style recommendations
- [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small) — Product catalog

## Evaluation Set

15 queries with manually curated ground truth in [`evaluation_set.json`](evaluation_set.json):
- 8 style queries (body type + occasion matching)
- 4 product queries (specific item search)
- 3 adversarial queries (wheelchair, budget, sustainability — no relevant docs exist)

Each entry includes the expected document IDs and notes explaining why each document was judged relevant or not. Run `python -m src.evaluate` to measure recall against this ground truth.
