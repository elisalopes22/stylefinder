# StyleFinder

A two-layer fashion Information Retrieval system that combines expert styling advice with product catalog search.

**Layer 1 — Style Recommendations:** Given a natural language query describing body type, personal style, and occasion, the system retrieves relevant outfit recommendations from a curated dataset of 3,193 expert-generated style guides.

**Layer 2 — Product Matching:** The best outfit recommendation is used to search a catalog of 42,587 real fashion products, returning specific items (with images) that match the recommended pieces.

---

## Table of Contents

- [Motivation](#motivation)
- [Datasets](#datasets)
- [System Architecture](#system-architecture)
- [Indexing Decisions](#indexing-decisions)
- [Setup and Usage](#setup-and-usage)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Task History](#task-history)

---

## Motivation

Fashion is a domain where users express needs through complex, multi-faceted queries that combine body characteristics, personal style, and situational context — for example, *"I have an hourglass figure and need something elegant for a summer wedding."* Traditional keyword-based retrieval (Boolean search, BM25) fails on these queries because:

- **Boolean search** requires exact keyword matches and returned 0 results for all 10 test queries in our Task 1 evaluation.
- **BM25** matches keywords literally — searching for *"dress that cinches at the waist"* returned waist pouches (backpack accessories) because BM25 matched the word "waist" without understanding the context.

Semantic search using sentence embeddings solves this by matching **meaning** rather than words, enabling the system to understand that "hourglass figure" and "defined waist with wider hips" refer to the same concept.

By combining professional styling knowledge (Layer 1) with a real product catalog (Layer 2), StyleFinder acts as an accessible personal stylist — a service traditionally available only through expensive consultations. The system also addresses body diversity and inclusivity, covering plus-size, petite, non-binary, and various body shapes.

---

## Datasets

Both datasets are loaded directly from HuggingFace (no manual download or web crawling required).

### Layer 1: Fashion Style Instruct

- **Source:** [neuralwork/fashion-style-instruct](https://huggingface.co/datasets/neuralwork/fashion-style-instruct)
- **Size:** 3,193 entries
- **Format:** Parquet (text only)
- **Language:** English
- **Content:** Each entry contains:
  - `input` — body type and personal style description (e.g., *"I'm a plus-size woman with a pear shape..."*)
  - `context` — occasion (e.g., *"I'm going to a job interview"*)
  - `completion` — 5 complete outfit recommendations with top, bottom, shoes, and accessories
- **Coverage:** 30+ occasions (job interview, wedding, music festival, hiking, etc.), diverse body types (hourglass, pear, apple, rectangle, athletic), all genders (men, women, non-binary)
- **Distribution:** Nearly uniform across occasions (88–96 entries each), meaning the system is not biased toward any particular event type

### Layer 2: Fashion Product Catalog

- **Source:** [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
- **Size:** 42,587 products (after filtering children's categories and null entries)
- **Format:** Parquet (text metadata + product images)
- **Language:** English
- **Filtering applied:** Removed `Boys` and `Girls` gender categories (~1,500 items) since the style dataset only covers adult recommendations
- **Content per product:** `productDisplayName`, `gender`, `masterCategory`, `subCategory`, `articleType` (141 types), `baseColour` (46 colors), `season`, `usage`, `year`, and a product image
- **Note on images:** Product images are available in the dataset and displayed as visual aids in the output, but **all indexing and retrieval is performed on text fields only**

---

## System Architecture

```
User Query: "hourglass figure, elegant summer wedding"
                    │
                    ▼
    ┌───────────────────────────────┐
    │  LAYER 1: Style Recommendations│
    │  FAISS semantic search on      │
    │  3,193 style documents         │
    │  → "Try a fitted wrap dress    │
    │     in jewel tones with        │
    │     strappy heels..."          │
    └───────────────┬───────────────┘
                    │ best recommendation
                    ▼
    ┌───────────────────────────────┐
    │  LAYER 2: Product Matching     │
    │  FAISS semantic search on      │
    │  42,587 product documents      │
    │  (optionally filtered by SQL)  │
    │  → Specific products with      │
    │     names, colors, images      │
    └───────────────────────────────┘
```

An optional **hybrid search** mode combines SQLite metadata filtering with FAISS semantic ranking. For example, a query can first filter by `gender='Women' AND article_type='Dresses' AND base_colour='Black'` using SQL, then rank the filtered subset by semantic similarity using FAISS. This combines the precision of structured queries with the flexibility of semantic understanding.

---

## Indexing Decisions

Based on our [Task 1 exploration](notebooks/Task1_Exploration.ipynb), we evaluated three retrieval methods (Boolean, BM25, Semantic) and chose **FAISS + SQLite** for persistent indexing. Below we document each decision and its rationale.

### Why FAISS + SQLite (and not Boolean/BM25)?

| Method | Task 1 Performance | Decision |
|--------|-------------------|----------|
| **Boolean Search** | 0 results for all 10 test queries | **Discarded** — AND logic is too strict for natural language queries combining multiple concepts |
| **BM25** | Returned waist pouches for "dress that cinches at the waist" | **Discarded** — keyword matching without semantic understanding produces irrelevant results |
| **Semantic (FAISS)** | Correctly matched body types, occasions, and style preferences | **Chosen** — understands meaning, not just keywords |
| **SQLite** | N/A (metadata filtering) | **Chosen** — enables structured pre-filtering before semantic search |

### Document Representation

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Granularity** | Whole document (one vector per document) | Style entries are coherent units combining body description, occasion, and 5 outfits. Splitting into sentences would lose the connection between body type and outfit. Product entries are short metadata strings (~10 words) that would not benefit from splitting. |
| **Style text construction** | `"Body: {input}. Occasion: {context}. Recommendation: {completion}"` | Preserves the full context: who the recommendation is for, when, and what. The model can then match queries against all three aspects simultaneously. |
| **Product text construction** | `"{gender} {category} {articleType} {colour} {season} {usage} - {productName}"` | Concatenates all available metadata fields into a searchable string. Example: `"Women Apparel Dress Dresses Black Fall Formal - Avirate Black Formal Dress"`. This creates semantic richness from structured fields. |
| **Image handling** | Text-based indexing only | Product images exist in the dataset and are displayed in results, but do not participate in the retrieval process. Future work could explore multimodal retrieval (e.g., CLIP). |

### Embedding and Index Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Embedding model** | `all-MiniLM-L6-v2` | Lightweight (22M parameters), fast inference, 384 dimensions. Provides good quality for general English text. Same model used in course Lab 2, enabling direct comparison with lab exercises. |
| **Embedding dimensions** | 384 | Determined by the model architecture. |
| **Normalization** | L2-normalized embeddings | Enables using inner product (IndexFlatIP) as equivalent to cosine similarity, which is the standard similarity metric for sentence embeddings. |
| **FAISS index type** | `IndexFlatIP` (exact search) | For our corpus sizes (3K + 42K), exact search is fast enough (~50ms per query). Approximate methods (IVF, HNSW) would be necessary for millions of documents but add complexity without benefit at this scale. |
| **SQLite schema** | Products: gender, masterCategory, subCategory, articleType, baseColour, season, usage, year. Styles: body_description, occasion, recommendation. | Indexes created on `gender`, `article_type`, `base_colour`, `season`, and `occasion` columns for fast filtering. |

### Persisted Index Files

| File | Size | Contents |
|------|------|----------|
| `product_faiss.index` | 62.31 MB | 42,587 product vectors (384-dim) |
| `stylefinder.db` | 16.03 MB | SQLite with products + styles metadata |
| `style_corpus.pkl` | 6.93 MB | Style document texts (for display) |
| `style_faiss.index` | 4.68 MB | 3,193 style vectors (384-dim) |
| `product_corpus.pkl` | 3.99 MB | Product document texts (for display) |
| `product_id_map.pkl` | 0.44 MB | FAISS index → doc_id mapping |
| `style_id_map.pkl` | 0.03 MB | FAISS index → doc_id mapping |
| **Total** | **~94 MB** | |

---

## Setup and Usage

### Setup with UV

```bash
git clone https://github.com/elisalopes22/stylefinder.git
cd stylefinder
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Setup on Google Colab

```python
!git clone https://github.com/elisalopes22/stylefinder.git
%cd stylefinder
!pip install -q rank_bm25 sentence-transformers datasets faiss-cpu
```

### Usage

```bash
# Step 1: Build indexes (first time only, ~2 min on GPU)
python -m src.build_index

# Step 2: Search (single query)
python -m src.search "hourglass figure elegant evening dress"

# Step 3: Interactive search mode
python -m src.search

# Step 4: Run evaluation
python -m src.evaluate
```

---

## Project Structure

```
stylefinder/
├── notebooks/
│   ├── Task1_Exploration.ipynb        # BM25 vs Semantic comparison, adversarial testing
│   └── Task2_Visualization.ipynb      # UMAP, evaluation charts, demo with product images
├── src/
│   ├── config.py                      # Paths, constants, model configuration
│   ├── load_data.py                   # HuggingFace data loading, Doc namedtuple creation
│   ├── build_index.py                 # FAISS + SQLite index construction and persistence
│   ├── search.py                      # StyleFinderEngine class with search methods + CLI
│   └── evaluate.py                    # Evaluation against curated ground truth
├── indexes/                           # Persisted index files (~94 MB, gitignored)
├── evaluation_set.json                # 15 queries with manually curated expected doc IDs
├── pyproject.toml                     # UV project configuration
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md
```

---

## Evaluation

### Evaluation Set

We created 15 evaluation queries with **manually curated** ground truth. For each query, we ran FAISS semantic search, inspected the top-5 results, and recorded which document IDs are genuinely relevant — not just whatever the system returned. This avoids circular evaluation.

The evaluation set is stored in [`evaluation_set.json`](evaluation_set.json) with the following structure:

```json
{
  "query_id": "Q01",
  "query": "flattering wedding guest dress that cinches at the waist",
  "layer": "style",
  "expected_doc_ids": ["207", "237", "379", "234", "2365"],
  "notes": "All results describe women with defined waists seeking dresses..."
}
```

### Results

| Category | Queries | With Hits | Recall@5 |
|----------|---------|-----------|----------|
| Style queries (Q01–Q08) | 8 | 8/8 | varies |
| Product queries (Q09–Q12) | 4 | variable | varies |
| Adversarial (Q13–Q15) | 3 | 0/3 (expected) | N/A |
| **Overall (realistic)** | **12** | **8/12** | **0.57** |

### Adversarial Queries

Three queries were designed to test concepts absent from both datasets:

| Query | Expected | Actual | Issue |
|-------|----------|--------|-------|
| "wheelchair user at a formal event" | No relevant docs | Returned generic style advice (score 0.44) | Disability/accessibility not represented in corpus |
| "budget friendly under 50 dollars" | No relevant docs | Returned generic outfits (score 0.61) | No price information in either dataset |
| "sustainable eco-friendly date night" | No relevant docs | Returned date night outfits (score 0.59) | Sustainability not represented in corpus |

These adversarial results expose a critical limitation: the system **never indicates when it lacks relevant data**. Cosine similarity always returns results with seemingly confident scores.

---

## Key Findings

1. **Semantic search dramatically outperforms keyword methods** for fashion queries. Boolean search returned 0 results for all queries; BM25 produced semantically irrelevant matches. FAISS semantic search correctly matched body types, occasions, and style concepts.

2. **SQL + FAISS hybrid search** is the most powerful retrieval mode. Pre-filtering by metadata (gender, article type, color) before semantic ranking ensures results are both categorically correct and semantically relevant.

3. **Occasion type drives semantic clustering** more than body type. UMAP visualizations show that hiking outfits cluster together regardless of body shape, which is desirable — the activity matters more than who is doing it.

4. **Product embeddings show weaker separation** than style embeddings because product documents are short metadata strings (~10 words) versus rich style descriptions (~500 words). This suggests that richer product descriptions would improve Layer 2 retrieval.

5. **The system fails silently** on out-of-scope queries, returning confident-looking scores for irrelevant results. A confidence threshold or out-of-distribution detection mechanism is needed.

---

## Limitations

- **No "I don't know" mechanism:** The system always returns results, even when no relevant documents exist in the corpus. High cosine similarity scores on adversarial queries are misleading.
- **Short product descriptions:** Product documents are constructed from metadata fields (~10 words), limiting semantic richness compared to style documents (~500 words). This causes weaker cluster separation in UMAP and lower retrieval precision for Layer 2.
- **Text-only retrieval:** Product images are available but not used for retrieval. Visual features (color, pattern, silhouette) could significantly improve product matching.
- **No personalization:** The system does not learn from user interactions or preferences over time.
- **English only:** Both datasets are in English; multilingual support would require model adaptation.
- **Corpus gaps:** No coverage for disability/accessibility, budget constraints, sustainability, or cultural dress codes (e.g., hijab-friendly fashion).

---

## Future Work

- **Confidence thresholds:** Implement out-of-distribution detection to flag when query topics fall outside corpus coverage.
- **Multimodal retrieval:** Use CLIP or FashionCLIP to incorporate product images into the retrieval process.
- **Knowledge graph:** Extract entities (body types, clothing items, occasions, colors) and their relationships using NetworkX, enabling graph-based queries.
- **Advanced retrieval methods:** Evaluate ColBERT, SPLADE, and RAG-based approaches (LangChain, LlamaIndex) on our evaluation set.
- **Richer product text:** Augment product descriptions with generated text to improve semantic quality of Layer 2.
- **Evaluation expansion:** Increase the evaluation set with more queries and formal relevance judgments (NDCG, MAP metrics).

---

## Task History

| Task | Description | Key Deliverables |
|------|-------------|-----------------|
| **Task 1** | Corpus selection, initial IR system, method comparison | [Task1_Exploration.ipynb](notebooks/Task1_Exploration.ipynb): Boolean vs BM25 vs Semantic search, adversarial query analysis, UMAP visualizations |
| **Task 2** | Persistent indexing, evaluation set | FAISS + SQLite indexes saved to disk, 15-query evaluation set with curated ground truth, SQL+FAISS hybrid search |

---

## References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
