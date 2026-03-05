# E-commerce-RAG-chatbot

url: https://e-commerce-rag-chatbot-flr6vevdzueohmjn5qmb49.streamlit.app/

# Talk to Your Data — LLM-Powered E-Commerce Analytics

A natural language analytics interface for the Olist Brazilian E-Commerce dataset. Ask questions in plain English — no SQL, no pandas required.

**Live demo:** [llm-powered-e-commerce-analytics.streamlit.app](https://llm-powered-e-commerce-analytics.streamlit.app/)

---

## What it does

Type any question about 438,038 e-commerce records and get a grounded, accurate answer in seconds.

```
"What is total revenue?"
"How many orders were delivered late?"
"Show top 5 product categories"
"Average delivery time by state"
"What is the order status of order id abc123def456..."
"How to increase sales?"
```

The system automatically decides how to answer each question — no configuration needed.

---

## Architecture

### Query Router (3-Way)

Every question is dispatched to one of three strategies:

| Route | Trigger | Method |
|---|---|---|
| **Exact Lookup** | Alphanumeric ID detected via regex | Direct dataframe scan — no embedding |
| **Compute** | Aggregations, counts, totals, trends | LLM generates pandas code → executes → auto-retries on failure |
| **RAG** | All other questions | Embed query → cosine similarity → top 10 docs → LLM generates answer |

### Full Pipeline

```
Raw CSVs (Kaggle Olist)
  → AWS S3 (source of truth)
  → Jupyter: clean + feature engineer (delivery_days, is_late, total_revenue, churn_score)
  → S3 (processed CSVs)
  → AWS RDS MySQL (6 tables via SQLAlchemy)
  → SQL Analytics (5 summary tables)
  → Power BI Dashboard (MySQL connector)

S3 (processed CSVs)
  → 438,038 LangChain Documents
  → HuggingFace all-MiniLM-L6-v2 (Kaggle T4 GPU)
  → embeddings.npy + documents.pkl → S3

App startup (Streamlit Cloud)
  → S3 cold-start: download embeddings.npy + documents.pkl + CSVs
  → Load pipeline → serve queries
```

---

## Dataset

[Olist Brazilian E-Commerce — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| Table | Rows |
|---|---|
| orders | 99,441 |
| customers | 99,441 |
| payments | 103,886 |
| reviews | 99,224 |
| products | 32,951 |
| sellers | 3,095 |

Stored in AWS S3 (`brazilian-ecommerce-vs`, `us-west-1`).

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq — Llama 3.1 8B Instant |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Search | NumPy cosine similarity (no vector DB) |
| Orchestration | LangChain LCEL |
| Database | AWS RDS MySQL |
| Storage | AWS S3 |
| BI Dashboard | Microsoft Power BI |
| UI | Streamlit Cloud |
| Embedding Compute | Kaggle T4 GPU (offline, one-time) |

---

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── notebooks/
│   ├── streamlit-e-commerce.ipynb   # Data processing & feature engineering
│   ├── rag-system.ipynb             # Embedding generation (Kaggle T4 GPU)
│   └── talk-to-your-data.ipynb     # Development & testing with pyngrok
└── .streamlit/
    └── secrets.toml        # API keys (not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/talk-to-your-data.git
cd talk-to-your-data
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
AWS_ACCESS_KEY_ID = "your_aws_access_key"
AWS_SECRET_ACCESS_KEY = "your_aws_secret_key"
AWS_DEFAULT_REGION = "us-west-1"
```

### 4. Run the app

```bash
streamlit run app.py
```

On first run, the app downloads `embeddings.npy` and `documents.pkl` from S3 (~cold start). Subsequent runs use `@st.cache_resource`.

---

## How the RAG system was built

The embedding pipeline was run offline on a Kaggle T4 GPU notebook (`rag-system.ipynb`):

1. Load all 6 cleaned CSVs from S3
2. Convert every row to a LangChain `Document` with structured text
3. Embed all 438,038 documents using `all-MiniLM-L6-v2` on GPU
4. Save `embeddings.npy` (float32 matrix) and `documents.pkl` (text list) to S3

At query time, only the user's question is embedded (CPU, <100ms). Cosine similarity is a single NumPy dot product against the pre-normalized matrix.

---

## How the text-to-pandas chain works

For aggregate questions, the LLM receives:

- Full schema: table names, column names, dtypes, sample values, unique value lists
- Table relationships and join keys
- Example code patterns for common query types

It generates executable pandas code, which is run against the in-memory dataframes. If execution fails, the error message is included in a follow-up prompt and the LLM auto-generates a fix.

---

## Potential improvements

- Add `order_items` table for product-level analytics
- Connect directly to RDS MySQL at query time (remove CSV loading)
- Chart generation for visual answers (matplotlib / plotly)
- Multi-turn conversation memory
- User authentication and query history logging
- Export answers to CSV or PDF
- Deploy on AWS ECS with a custom domain

---

## License

MIT
