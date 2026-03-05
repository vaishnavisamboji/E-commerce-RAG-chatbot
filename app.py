import streamlit as st
import pandas as pd
import numpy as np
import boto3
import os
import re
import pickle
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Talk to your Data",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
* { font-family: 'IBM Plex Sans', sans-serif; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #000000; color: #e0e0e0; }
[data-testid="stHeader"] { background-color: #000000; }
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 400;
    letter-spacing: 0.12em; text-transform: uppercase; color: #555555;
    background: transparent; border: none; border-bottom: 1px solid #1a1a1a;
    padding: 12px 24px; transition: all 0.2s; }
[data-testid="stTabs"] button:hover { color: #ffffff; background: transparent; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff; border-bottom: 1px solid #ffffff; background: transparent; }
[data-testid="stTabsContent"] { background: #000000; border: none; padding-top: 32px; }
[data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea, textarea {
    background-color: #0a0a0a; border: 1px solid #222222; border-radius: 0px;
    color: #e0e0e0; font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; padding: 12px 16px; }
[data-testid="stTextInput"] input:focus, textarea:focus {
    border-color: #444444; box-shadow: none; }
[data-testid="stChatInput"] { background-color: #0a0a0a; border: 1px solid #222222; border-radius: 0px; }
[data-testid="stChatInput"] button { color: #ffffff; }
[data-testid="stChatMessage"] {
    background-color: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 0px;
    padding: 16px 20px; margin-bottom: 8px; }
[data-testid="stChatMessage"] p { color: #e0e0e0; font-size: 14px; line-height: 1.7; }
[data-testid="stDataFrame"] { border: 1px solid #1a1a1a; }
.stDataFrame table { background-color: #0a0a0a; color: #e0e0e0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
.stDataFrame th { background-color: #111111; color: #888888;
    border-bottom: 1px solid #222222; font-weight: 400; letter-spacing: 0.05em;
    text-transform: uppercase; font-size: 11px; }
.stDataFrame td { border-bottom: 1px solid #111111; color: #cccccc; }
[data-testid="stSelectbox"] select, [data-testid="stSelectbox"] > div > div {
    background-color: #0a0a0a; border: 1px solid #222222; border-radius: 0px;
    color: #e0e0e0; font-size: 13px; }
[data-testid="stButton"] button {
    background-color: #000000; border: 1px solid #333333; border-radius: 0px;
    color: #e0e0e0; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 8px 20px; transition: all 0.2s; }
[data-testid="stButton"] button:hover {
    background-color: #111111; border-color: #666666; color: #ffffff; }
[data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #1a1a1a; padding: 20px 24px; }
[data-testid="stMetricLabel"] { color: #555555; font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #ffffff; font-size: 28px; font-weight: 300; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #333333; }
::-webkit-scrollbar-thumb:hover { background: #555555; }
hr { border-color: #111111; margin: 24px 0; }
[data-testid="stSpinner"] { color: #555555; }
[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #111111; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div style="border-bottom: 1px solid #111111; padding-bottom: 24px; margin-bottom: 8px;">
    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px;
                 letter-spacing: 0.2em; text-transform: uppercase; color: #444444;">
        LLM-Powered E-Commerce Analytics
    </span>
    <h1 style="font-family: 'IBM Plex Sans', sans-serif; font-size: 28px;
               font-weight: 300; color: #ffffff; margin: 6px 0 0 0; letter-spacing: -0.02em;">
        Talk to your Data 
    </h1>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD PIPELINE
# =============================================================================
@st.cache_resource(show_spinner="Loading pipeline...")
@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline():
    # Read API keys from Streamlit secrets
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
        AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-west-1")
    except KeyError as e:
        st.error(f"Missing secret: {e}. Please add all required secrets in the Streamlit dashboard.")
        st.stop()

    # S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    BUCKET = 'brazilian-ecommerce-vs'
    TABLES = ['orders', 'customers', 'products', 'reviews', 'payments', 'sellers']
    dfs = {}
    for table in TABLES:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=f'{table}.csv')
            dfs[table] = pd.read_csv(obj['Body'])
        except Exception as e:
            st.warning(f'{table}: skipped ({e})')

    # Download embeddings from S3 if not present
    def download_if_missing(local_path, s3_key):
        if not os.path.exists(local_path):
            with st.spinner(f"Downloading {s3_key} from S3..."):
                s3.download_file(BUCKET, s3_key, local_path)

    download_if_missing('embeddings.npy', 'embeddings.npy')
    download_if_missing('documents.pkl', 'documents.pkl')

    # Load pre-built embeddings
    embs_matrix = np.load('embeddings.npy')
    with open('documents.pkl', 'rb') as f:
        texts = pickle.load(f)

    norms = np.linalg.norm(embs_matrix, axis=1, keepdims=True)
    embs_matrix = embs_matrix / np.maximum(norms, 1e-9)

    # Embedding model for query only
    embeddings = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},  # use CPU on Streamlit Cloud; they don't provide GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # LLM
    llm = ChatGroq(
        model='llama-3.1-8b-instant',
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=512
    )


    # Schema
    schema_parts = []
    for name, df in dfs.items():
        col_info = []
        for col in df.columns:
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else 'N/A'
            dtype  = df[col].dtype
            if df[col].nunique() <= 20:
                unique_vals = df[col].dropna().unique().tolist()
                col_info.append(f'    {col} ({dtype}) — unique values: {unique_vals}')
            else:
                col_info.append(f'    {col} ({dtype}) — e.g. {sample}')
        schema_parts.append(f"dfs['{name}'] — {len(df):,} rows:\n" + '\n'.join(col_info))

    SCHEMA = '\n\n'.join(schema_parts) + """

TABLE RELATIONSHIPS:
- dfs['orders']   + dfs['customers'] → join on 'customer_id'
- dfs['orders']   + dfs['payments']  → join on 'order_id'
- dfs['orders']   + dfs['reviews']   → join on 'order_id'
- dfs['products'] is standalone
- dfs['sellers']  is standalone

IMPORTANT RULES:
- All date columns are strings — ALWAYS convert with pd.to_datetime() before date math
- NEVER join tables not listed above
- Product-level order counts require order_items — say so if asked

EXAMPLE CODE PATTERNS:

# Orders by state:
merged = dfs['orders'].merge(dfs['customers'][['customer_id','customer_state']], on='customer_id')
result = merged.groupby('customer_state').size().reset_index(name='order_count').sort_values('order_count', ascending=False)

# Total revenue:
result = dfs['payments']['payment_value'].sum()

# Revenue by state:
merged = dfs['orders'].merge(dfs['customers'][['customer_id','customer_state']], on='customer_id')
merged = merged.merge(dfs['payments'][['order_id','payment_value']], on='order_id')
result = merged.groupby('customer_state')['payment_value'].sum().sort_values(ascending=False)

# Average order value:
result = dfs['payments'].groupby('order_id')['payment_value'].sum().mean()

# Unique customers:
result = dfs['customers']['customer_unique_id'].nunique()

# Repeating customers:
counts = dfs['orders'].merge(dfs['customers'][['customer_id','customer_unique_id']], on='customer_id')
result = (counts.groupby('customer_unique_id').size() > 1).sum()

# Orders by status:
result = dfs['orders']['order_status'].value_counts()

# Average review score:
result = dfs['reviews']['review_score'].mean()

# Average delivery time in days:
o = dfs['orders'].copy()
o['order_purchase_timestamp']      = pd.to_datetime(o['order_purchase_timestamp'], errors='coerce')
o['order_delivered_customer_date'] = pd.to_datetime(o['order_delivered_customer_date'], errors='coerce')
o = o.dropna(subset=['order_purchase_timestamp','order_delivered_customer_date'])
result = (o['order_delivered_customer_date'] - o['order_purchase_timestamp']).dt.days.mean()

# Late orders:
o = dfs['orders'].copy()
o['order_estimated_delivery_date'] = pd.to_datetime(o['order_estimated_delivery_date'], errors='coerce')
o['order_delivered_customer_date'] = pd.to_datetime(o['order_delivered_customer_date'], errors='coerce')
o = o.dropna(subset=['order_estimated_delivery_date','order_delivered_customer_date'])
result = (o['order_delivered_customer_date'] > o['order_estimated_delivery_date']).sum()

# Top product categories:
result = dfs['products']['product_category_name'].value_counts().head(10)

# Sellers by state:
result = dfs['sellers']['seller_state'].value_counts()

# Revenue by month:
merged = dfs['payments'].merge(dfs['orders'][['order_id','order_purchase_timestamp']], on='order_id')
merged['month'] = pd.to_datetime(merged['order_purchase_timestamp'], errors='coerce').dt.to_period('M')
result = merged.groupby('month')['payment_value'].sum()

# Revenue by year:
merged = dfs['payments'].merge(dfs['orders'][['order_id','order_purchase_timestamp']], on='order_id')
merged['year'] = pd.to_datetime(merged['order_purchase_timestamp'], errors='coerce').dt.year
result = merged.groupby('year')['payment_value'].sum()
"""

    # Prompts + chains
    router_prompt = ChatPromptTemplate.from_template("""
You are a routing assistant. Given a question about a database, decide if it needs:
- "compute": counts, totals, averages, rankings, grouping, trends, unique values, distributions
- "lookup": finding details about one specific record or ID
When in doubt, choose "compute".
Reply with ONLY one word: compute or lookup
Question: {question}
""")
    router_chain = router_prompt | llm | StrOutputParser()

    codegen_prompt = ChatPromptTemplate.from_template("""
You are a Python/pandas expert. You have access to a dict `dfs` with these exact tables, columns, dtypes, sample values, relationships and example patterns:
{schema}
Write a Python code snippet that answers the question.
- Use ONLY column names exactly as shown in the schema
- Follow the EXAMPLE CODE PATTERNS above as closely as possible
- Store the final answer in a variable called `result`
- Return ONLY executable Python code, no imports, no markdown, no explanation
Question: {question}
""")
    codegen_chain = codegen_prompt | llm | StrOutputParser()

    fix_prompt = ChatPromptTemplate.from_template("""
You are a Python/pandas expert. The following code produced an error.
Schema: {schema}
Original question: {question}
Broken code: {code}
Error: {error}
Fix the code. Use ONLY column names from the schema. Store result in `result`.
Return ONLY executable Python code, no markdown, no explanation.
""")
    fix_chain = fix_prompt | llm | StrOutputParser()

    rag_prompt = ChatPromptTemplate.from_template("""
You are a friendly, conversational data analyst assistant.
Answer using ONLY the database records below.
Be chatty and engaging like a helpful colleague. Keep it concise. Never invent data.
Context: {context}
Question: {question}
Answer:""")

    def execute_code(code):
        local_ns = {'dfs': dfs, 'pd': pd}
        exec(code, {}, local_ns)
        return local_ns.get('result', 'No result variable set.')

    def run_pandas(question):
        code = codegen_chain.invoke({'schema': SCHEMA, 'question': question})
        code = re.sub(r'```python|```', '', code).strip()
        try:
            result = execute_code(code)
        except Exception as e:
            fixed = fix_chain.invoke({'schema': SCHEMA, 'question': question,
                                      'code': code, 'error': str(e)})
            fixed = re.sub(r'```python|```', '', fixed).strip()
            try:
                result = execute_code(fixed)
            except Exception as e2:
                return "I wasn't able to compute that. Try rephrasing!"
        format_prompt = f"""Question: {question}
Computed result:
{result}
Answer in a friendly, conversational tone. Be concise but engaging.
If the result is a table or list, present it cleanly.
Don't say "based on the data" — just answer naturally like a helpful colleague."""
        return llm.invoke(format_prompt).content

    def is_exact_lookup(question):
        return bool(re.search(r'[a-f0-9]{20,}', question.lower()))

    def run_exact_lookup(question):
        match = re.search(r'([a-f0-9]{20,})', question.lower())
        if not match:
            return "Couldn't find an ID in your question."
        lookup_id = match.group(1)
        found_rows = []
        for table, df in dfs.items():
            for col in df.columns:
                mask = df[col].astype(str).str.lower() == lookup_id
                if mask.any():
                    row = df[mask].iloc[0].to_dict()
                    fields = '\n'.join(f'  {k}: {v}' for k, v in row.items()
                                      if pd.notna(v) and str(v) != 'nan')
                    found_rows.append(f"Table '{table}':\n{fields}")
        if not found_rows:
            return f"Searched everywhere but couldn't find `{lookup_id}`. Double-check the ID."
        all_found = '\n\n'.join(found_rows)
        fmt = f"""Question: {question}
Found these records:
{all_found}
Present this in a friendly, conversational way. Answer the question directly."""
        return llm.invoke(fmt).content

    def rag_answer(question):
        chunks   = retrieve(question, k=10)
        context  = '\n\n'.join(f'[{i+1}] {c}' for i, c in enumerate(chunks))
        messages = rag_prompt.format_messages(context=context, question=question)
        return llm.invoke(messages).content

    def ask(question):
        if is_exact_lookup(question):
            return run_exact_lookup(question)
        route = router_chain.invoke({'question': question}).strip().lower()
        if 'compute' in route:
            return run_pandas(question)
        else:
            return rag_answer(question)

    return dfs, ask, SCHEMA

dfs, ask, SCHEMA = load_pipeline()

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Dataset", "Model", "About", "Dashboard"])

# =============================================================================
# TAB 1 — CHAT
# =============================================================================
with tab1:
    st.markdown("""
    <p style="color: #444444; font-size: 13px; font-family: 'IBM Plex Mono', monospace;
              letter-spacing: 0.05em; margin-bottom: 32px;">
        Ask anything about your data in plain English.
    </p>
    """, unsafe_allow_html=True)

    # Initialize session state for sample question
    if 'sample_question' not in st.session_state:
        st.session_state.sample_question = None

    # Sample questions row
    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #444444;
              letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">
        Try asking:
    </p>
    """, unsafe_allow_html=True)

    sample_qs = [
        "What is total revenue?",
        "Show top 5 product categories",
        "How many orders were delivered late?",
        "Average delivery time",
        "What is the average review score?"
    ]

    # Create columns for buttons (adjust number per row as needed)
    cols = st.columns(len(sample_qs))
    for i, q in enumerate(sample_qs):
        with cols[i]:
            if st.button(q, key=f"sample_{i}"):
                st.session_state.sample_question = q
                st.rerun()

    st.markdown("<hr style='margin: 20px 0; border-color: #1a1a1a;'>", unsafe_allow_html=True)

    # Message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Chat input
    prompt = st.chat_input("Ask a question about your data...")

    # If a sample question was clicked, use it as prompt
    if st.session_state.sample_question:
        prompt = st.session_state.sample_question
        st.session_state.sample_question = None  # clear after use

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            with st.spinner(''):
                response = ask(prompt)
            st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.rerun()

    if st.session_state.messages:
        if st.button('Clear conversation'):
            st.session_state.messages = []
            st.rerun()
# =============================================================================
# TAB 2 — DATASET
# =============================================================================
with tab2:
    st.markdown("""
    <p style="color: #444444; font-size: 13px; font-family: 'IBM Plex Mono', monospace;
              letter-spacing: 0.05em; margin-bottom: 32px;">
        Browse the raw tables loaded from S3.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 5])
    with col1:
        selected_table = st.selectbox('Select table', options=list(dfs.keys()),
                                      label_visibility='collapsed')
    with col2:
        st.markdown(f"""
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px;
                     color: #444444; letter-spacing: 0.1em; text-transform: uppercase;
                     line-height: 38px;">
            {dfs[selected_table].shape[0]:,} rows &nbsp;&nbsp;{dfs[selected_table].shape[1]} columns
        </span>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    df = dfs[selected_table]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', f'{len(df):,}')
    c2.metric('Columns', f'{df.shape[1]}')
    c3.metric('Null values', f'{df.isnull().sum().sum():,}')
    c4.metric('Unique rows', f'{df.drop_duplicates().shape[0]:,}')

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
    n_rows = st.slider('Rows to display', min_value=10, max_value=500,
                       value=50, step=10, label_visibility='collapsed')
    st.dataframe(df.head(n_rows), use_container_width=True, height=420)

    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #444444;
              letter-spacing: 0.1em; text-transform: uppercase;
              margin-top: 32px; margin-bottom: 12px;">Column info</p>
    """, unsafe_allow_html=True)
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-null': df.count().values,
        'Null': df.isnull().sum().values,
        'Unique': df.nunique().values,
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3 — MODEL
# =============================================================================
with tab3:
    st.markdown('<div style="color:#444; font-size:13px; font-family:IBM Plex Mono,monospace; letter-spacing:0.05em; margin-bottom:40px;">Architecture and components powering this application.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px;">Data Layer</div>
            <div style="font-size:18px; font-weight:300; color:#fff; margin-bottom:8px;">AWS S3 + RDS MySQL</div>
            <div style="font-size:13px; color:#555; line-height:1.6;">
                Raw CSVs ingested into S3 as the source of truth. Cleaned and feature-engineered
                via Jupyter — delivery_days, is_late, total_revenue, churn_score added.
                All 6 tables loaded into AWS RDS MySQL via SQLAlchemy. SQL analytics queries
                generate 5 summary tables saved back into RDS for Power BI.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px;">Retrieval</div>
            <div style="font-size:18px; font-weight:300; color:#fff; margin-bottom:8px;">Semantic Search</div>
            <div style="font-size:13px; color:#555; line-height:1.6;">
                HuggingFace all-MiniLM-L6-v2 embeds 438,038 documents on Kaggle T4 GPU.
                Embeddings saved as embeddings.npy + documents.pkl and uploaded to S3.
                Cosine similarity via numpy dot product at query time — no vector DB required.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px;">Generation</div>
            <div style="font-size:18px; font-weight:300; color:#fff; margin-bottom:8px;">Groq — Llama 3.1 8B</div>
            <div style="font-size:13px; color:#555; line-height:1.6;">
                Free tier. Sub-second inference via Groq LPU hardware.
                Temperature 0 for deterministic, factual answers.
                Used across all 3 routes — lookup formatting, pandas result formatting, and RAG generation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px;">Business Intelligence</div>
            <div style="font-size:18px; font-weight:300; color:#fff; margin-bottom:8px;">Power BI via MySQL Connector</div>
            <div style="font-size:13px; color:#555; line-height:1.6;">
                Power BI connects directly to RDS MySQL via the MySQL connector.
                Pulls from 5 pre-computed analytics tables — monthly revenue, customer segments,
                delivery performance, payment breakdown, and product categories.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px;">Query Router</div>
            <div style="font-size:18px; font-weight:300; color:#fff; margin-bottom:8px;">3-Way Routing</div>
            <div style="font-size:13px; color:#555; line-height:1.6;">
                Exact alphanumeric ID detected via regex — bypasses embeddings, scans dataframes directly.
                Aggregate questions (counts, totals, trends) routed to pandas code generation.
                All other questions use RAG retrieval.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px;">Code Generation</div>
            <div style="font-size:18px; font-weight:300; color:#fff; margin-bottom:8px;">Text-to-Pandas</div>
            <div style="font-size:13px; color:#555; line-height:1.6;">
                LLM receives full schema — column names, dtypes, sample values, join relationships,
                and example patterns. Generates pandas code, executes against real dataframes.
                Auto-retries with the error message included if execution fails.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline flow
    st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin:8px 0 16px 0;">Full pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="border:1px solid #1a1a1a; padding:28px; font-family:IBM Plex Mono,monospace; font-size:12px; line-height:2.8;">
        <div>
            <span style="color:#555">Raw CSVs</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">S3</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">Clean + Feature Engineer</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">S3 (processed CSVs)</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">RDS MySQL</span>
        </div>
        <div>
            <span style="color:#555">RDS MySQL</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">SQL Analytics (5 summary tables)</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">Power BI Dashboard</span>
        </div>
        <div>
            <span style="color:#555">S3 (processed CSVs)</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">438,038 docs embedded on GPU</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">embeddings.npy + documents.pkl</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">S3</span>
        </div>
        <div style="margin-top:12px; border-top:1px solid #1a1a1a; padding-top:12px;">
            <span style="color:#555">Question</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#666">ID detected?</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">Direct dataframe scan &#8594; LLM format</span>
        </div>
        <div style="padding-left:120px;">
            <span style="color:#666">Aggregate?</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">Codegen &#8594; exec &#8594; auto-retry &#8594; LLM format</span>
        </div>
        <div style="padding-left:120px;">
            <span style="color:#666">Else</span>
            <span style="color:#333"> &#8594; </span>
            <span style="color:#aaa">Embed query &#8594; cosine similarity &#8594; top 10 docs &#8594; LLM generate</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px; margin-top:8px;">Full Pipeline</div>

    <div style="border:1px solid #1a1a1a; padding:32px; overflow-x:auto;">

      <!-- PHASE 1 — INGESTION -->
      <div style="display:flex; align-items:center; gap:0; margin-bottom:4px;">
        <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.12em; width:100px; text-align:right; padding-right:16px; flex-shrink:0;">Ingestion</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">Raw Olist CSVs<div style="font-size:9px; color:#444; margin-top:3px;">orders · customers · products · reviews · payments · sellers</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#05080d; border:1px solid #1a2a3a; padding:10px 14px; font-size:12px; color:#6aadcf; white-space:nowrap;">AWS S3<div style="font-size:9px; color:#2a4a5a; margin-top:3px;">brazilian-ecommerce-vs · us-west-1</div></div>
      </div>

      <!-- connector -->
      <div style="display:flex; margin-bottom:4px;">
        <div style="width:100px; flex-shrink:0;"></div>
        <div style="width:16px; border-left:1px solid #1a1a1a; margin-left:0; height:20px;"></div>
      </div>

      <!-- PHASE 2 — PROCESSING -->
      <div style="display:flex; align-items:center; gap:0; margin-bottom:4px;">
        <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.12em; width:100px; text-align:right; padding-right:16px; flex-shrink:0;">Processing</div>
        <div style="background:#05080d; border:1px solid #1a2a3a; padding:10px 14px; font-size:12px; color:#6aadcf; white-space:nowrap;">AWS S3</div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">Jupyter Notebook<div style="font-size:9px; color:#444; margin-top:3px;">s3fs · pandas</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">Clean + Feature Engineer<div style="font-size:9px; color:#444; margin-top:3px;">delivery_days · is_late · total_revenue</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#05080d; border:1px solid #1a2a3a; padding:10px 14px; font-size:12px; color:#6aadcf; white-space:nowrap;">S3 (processed CSVs)<div style="font-size:9px; color:#2a4a5a; margin-top:3px;">6 cleaned tables</div></div>
      </div>

      <!-- connector -->
      <div style="display:flex; margin-bottom:4px;">
        <div style="width:100px; flex-shrink:0;"></div>
        <div style="width:16px; border-left:1px solid #1a1a1a; margin-left:0; height:20px;"></div>
      </div>

      <!-- PHASE 3 — SQL ANALYTICS -->
      <div style="display:flex; align-items:center; gap:0; margin-bottom:4px;">
        <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.12em; width:100px; text-align:right; padding-right:16px; flex-shrink:0;">SQL Analytics</div>
        <div style="background:#05080d; border:1px solid #1a2a3a; padding:10px 14px; font-size:12px; color:#6aadcf; white-space:nowrap;">S3 (processed)</div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">AWS RDS MySQL<div style="font-size:9px; color:#444; margin-top:3px;">SQLAlchemy · 6 tables loaded</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">SQL Queries<div style="font-size:9px; color:#444; margin-top:3px;">monthly · segments · delivery · payments · products</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">5 Summary Tables<div style="font-size:9px; color:#444; margin-top:3px;">saved back to MySQL</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#060d06; border:1px solid #1f3a1f; padding:10px 14px; font-size:12px; color:#6dbf6d; white-space:nowrap;">Power BI Dashboard<div style="font-size:9px; color:#2a5a2a; margin-top:3px;">MySQL connector</div></div>
      </div>

      <!-- connector -->
      <div style="display:flex; margin-bottom:4px;">
        <div style="width:100px; flex-shrink:0;"></div>
        <div style="width:16px; border-left:1px solid #1a1a1a; margin-left:0; height:20px;"></div>
      </div>

      <!-- PHASE 4 — EMBEDDING -->
      <div style="display:flex; align-items:center; gap:0; margin-bottom:4px;">
        <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.12em; width:100px; text-align:right; padding-right:16px; flex-shrink:0;">Embedding</div>
        <div style="background:#05080d; border:1px solid #1a2a3a; padding:10px 14px; font-size:12px; color:#6aadcf; white-space:nowrap;">S3 (processed)</div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">Kaggle T4 GPU<div style="font-size:9px; color:#444; margin-top:3px;">438,038 LangChain Documents</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#ccc; white-space:nowrap;">all-MiniLM-L6-v2<div style="font-size:9px; color:#444; margin-top:3px;">HuggingFace · normalize embeddings</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#05080d; border:1px solid #1a2a3a; padding:10px 14px; font-size:12px; color:#6aadcf; white-space:nowrap;">embeddings.npy + documents.pkl<div style="font-size:9px; color:#2a4a5a; margin-top:3px;">uploaded to S3</div></div>
      </div>

      <!-- connector -->
      <div style="display:flex; margin-bottom:4px;">
        <div style="width:100px; flex-shrink:0;"></div>
        <div style="width:16px; border-left:1px solid #1a1a1a; margin-left:0; height:20px;"></div>
      </div>

      <!-- PHASE 5 — APP STARTUP -->
      <div style="display:flex; align-items:center; gap:0; margin-bottom:4px;">
        <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.12em; width:100px; text-align:right; padding-right:16px; flex-shrink:0;">App Startup</div>
        <div style="background:#05080d; border:1px solid #1a2a3a; padding:10px 14px; font-size:12px; color:#6aadcf; white-space:nowrap;">S3 cold-start download<div style="font-size:9px; color:#2a4a5a; margin-top:3px;">embeddings.npy · documents.pkl · 6 CSVs</div></div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#080808; border:1px solid #1e1e1e; padding:10px 14px; font-size:12px; color:#fff; white-space:nowrap;">User Question</div>
        <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
        <div style="background:#0a0800; border:1px solid #2a2010; padding:10px 14px; font-size:12px; color:#c4a35a; white-space:nowrap;">3-Way Router<div style="font-size:9px; color:#5a4a20; margin-top:3px;">LangChain LCEL</div></div>
      </div>

      <!-- connector -->
      <div style="display:flex; margin-bottom:4px;">
        <div style="width:100px; flex-shrink:0;"></div>
        <div style="width:16px; border-left:1px solid #1a1a1a; margin-left:0; height:20px;"></div>
      </div>

      <!-- PHASE 6 — ROUTING (3 branches) -->
      <div style="display:flex; align-items:flex-start; gap:0;">
        <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.12em; width:100px; text-align:right; padding-right:16px; flex-shrink:0; padding-top:14px;">Routing</div>
        <div style="border-left:1px solid #1a1a1a; padding-left:0; display:flex; flex-direction:column; gap:8px;">

          <!-- Route A -->
          <div style="display:flex; align-items:center; gap:0;">
            <div style="width:20px; height:1px; background:#1a1a1a; flex-shrink:0;"></div>
            <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.1em; padding:4px 8px; border:1px solid #151515; background:#050505; white-space:nowrap; margin-right:8px;">ID Detected</div>
            <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Regex match</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Direct dataframe scan</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Groq · Llama 3.1 8B</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#0a0a0a; border:1px solid #222; padding:8px 14px; font-size:12px; color:#fff; white-space:nowrap;">Answer</div>
          </div>

          <!-- Route B -->
          <div style="display:flex; align-items:center; gap:0;">
            <div style="width:20px; height:1px; background:#1a1a1a; flex-shrink:0;"></div>
            <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.1em; padding:4px 8px; border:1px solid #151515; background:#050505; white-space:nowrap; margin-right:8px;">Aggregate</div>
            <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Schema + prompt</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Codegen &#8594; exec</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Auto-retry on error</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Groq · Llama 3.1 8B</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#0a0a0a; border:1px solid #222; padding:8px 14px; font-size:12px; color:#fff; white-space:nowrap;">Answer</div>
          </div>

          <!-- Route C -->
          <div style="display:flex; align-items:center; gap:0;">
            <div style="width:20px; height:1px; background:#1a1a1a; flex-shrink:0;"></div>
            <div style="font-family:IBM Plex Mono,monospace; font-size:9px; color:#333; text-transform:uppercase; letter-spacing:0.1em; padding:4px 8px; border:1px solid #151515; background:#050505; white-space:nowrap; margin-right:8px;">RAG</div>
            <div style="color:#2a2a2a; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Embed query</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Cosine similarity</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Top 10 docs</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#080808; border:1px solid #161616; padding:8px 12px; font-size:11px; color:#555; white-space:nowrap;">Groq · Llama 3.1 8B</div>
            <div style="color:#222; padding:0 8px; font-size:13px; flex-shrink:0;">&#8594;</div>
            <div style="background:#0a0a0a; border:1px solid #222; padding:8px 14px; font-size:12px; color:#fff; white-space:nowrap;">Answer</div>
          </div>

        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:16px;">Loaded data</div>', unsafe_allow_html=True)
    cols = st.columns(len(dfs))
    for i, (name, df) in enumerate(dfs.items()):
        cols[i].metric(name, f'{len(df):,}')
# =============================================================================
# TAB 4 — ABOUT
# =============================================================================
with tab4:
    st.markdown('<div style="color:#444; font-size:13px; font-family:IBM Plex Mono,monospace; letter-spacing:0.05em; margin-bottom:40px;">What this is, why it matters, and where it goes next.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:

        # What this is
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;">What this is</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300; margin-bottom:12px;">LLM-Powered E-Commerce Analytics is a full end-to-end data project built on the Olist Brazilian E-Commerce dataset — from raw ingestion and SQL analytics to a Power BI dashboard and a natural language query interface backed by RAG and LLM-generated pandas code.</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300; margin-bottom:40px;">The goal: eliminate the need for SQL or pandas to get answers from structured data. Ask a plain English question and receive a grounded, accurate answer drawn directly from 438,038 records across 6 tables.</div>', unsafe_allow_html=True)

        # How it was built
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;">How it was built</div>', unsafe_allow_html=True)

        # Step 1
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#555; letter-spacing:0.12em; text-transform:uppercase; margin:20px 0 8px 0;">1 — Data Ingestion</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300;">Raw Olist CSVs (orders, customers, products, reviews, payments, sellers) were uploaded to an AWS S3 bucket (<code style="background:transparent; color:#888; font-size:12px;">brazilian-ecommerce-vs</code>) as the single source of truth for the entire project.</div>', unsafe_allow_html=True)

        # Step 2
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#555; letter-spacing:0.12em; text-transform:uppercase; margin:20px 0 8px 0;">2 — Data Processing &amp; Feature Engineering</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300;">A Jupyter notebook connected to S3 via <code style="background:transparent; color:#888; font-size:12px;">s3fs</code> and loaded all 6 tables. Data quality checks were run across all tables — null counts, duplicate detection, and referential integrity checks (orders without customers, orders without payments, orders without reviews).</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300; margin-top:10px;">Cleaning steps included: converting all date columns to datetime, filling missing delivery dates with estimated dates, filling null review titles and messages, and imputing missing product dimensions with column medians. Feature engineering added <code style="background:transparent; color:#888; font-size:12px;">delivery_days</code>, <code style="background:transparent; color:#888; font-size:12px;">is_late</code>, <code style="background:transparent; color:#888; font-size:12px;">total_revenue</code> per order, and a customer-level <code style="background:transparent; color:#888; font-size:12px;">churn_score</code> (customers with only one order flagged as likely churned). Processed tables were uploaded back to S3 as cleaned CSVs.</div>', unsafe_allow_html=True)

        # Step 3
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#555; letter-spacing:0.12em; text-transform:uppercase; margin:20px 0 8px 0;">3 — AWS RDS MySQL + SQL Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300;">An AWS RDS MySQL instance was provisioned and all 6 cleaned tables were loaded from the notebook via SQLAlchemy. SQL queries were written directly in the notebook to generate 5 analytics summary tables: monthly revenue &amp; order trends, customer segments by state (High / Mid / Low Value), delivery performance by state, payment method breakdown, and product category analysis. All summary tables were saved back into MySQL for Power BI consumption.</div>', unsafe_allow_html=True)

        # Step 4
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#555; letter-spacing:0.12em; text-transform:uppercase; margin:20px 0 8px 0;">4 — Power BI Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300;">Power BI connected directly to RDS MySQL via the MySQL connector and queried the analytics summary tables to build the business intelligence dashboard — covering revenue trends, delivery performance, payment methods, customer segments, and state-level breakdowns.</div>', unsafe_allow_html=True)

        # Step 5
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#555; letter-spacing:0.12em; text-transform:uppercase; margin:20px 0 8px 0;">5 — RAG + LLM Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300;">Built on a Kaggle T4 GPU notebook. All 438,038 rows were converted into LangChain Documents and embedded with <code style="background:transparent; color:#888; font-size:12px;">all-MiniLM-L6-v2</code> on GPU. The embeddings matrix was saved as <code style="background:transparent; color:#888; font-size:12px;">embeddings.npy</code> and document texts as <code style="background:transparent; color:#888; font-size:12px;">documents.pkl</code>, then uploaded to S3. At runtime, cosine similarity is computed entirely in numpy — no vector database required.</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300; margin-top:10px;">A 3-way query router dispatches every question to the right strategy: exact alphanumeric IDs bypass embeddings and scan dataframes directly; aggregate questions (counts, totals, averages, trends) route to a text-to-pandas chain that generates executable code, runs it against the real dataframes, and auto-retries with error context if execution fails; all other questions use RAG — top 10 records retrieved by cosine similarity, answered by the LLM from that context.</div>', unsafe_allow_html=True)

        # Step 6
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#555; letter-spacing:0.12em; text-transform:uppercase; margin:20px 0 8px 0;">6 — Streamlit App, GitHub &amp; Deployment</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300; margin-bottom:40px;">The app was written and tested inside the Kaggle notebook using <code style="background:transparent; color:#888; font-size:12px;">pyngrok</code> to expose a live public URL. The final <code style="background:transparent; color:#888; font-size:12px;">app.py</code> was pushed to GitHub and deployed on Streamlit Cloud, where it downloads embeddings from S3 on cold start and runs the full pipeline at zero infrastructure cost.</div>', unsafe_allow_html=True)

        # Business value
        st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;">Business value</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300; margin-bottom:12px;">Analysts spend significant time writing repetitive queries for stakeholders. This removes that bottleneck — a product manager can query delivery performance, a finance team can pull monthly revenue by state, and a support agent can look up any order by ID, all without writing a single line of code.</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px; color:#ccc; line-height:1.8; font-weight:300; margin-bottom:40px;">The architecture is database-agnostic. Swapping the data source for a production MySQL, Postgres, or Snowflake instance requires only changes to the loading layer — the router, codegen chain, and RAG pipeline remain unchanged.</div>', unsafe_allow_html=True)

    with col2:

        # Dataset
        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:20px;">Dataset</div>
            <div style="font-size:13px; color:#888; line-height:2.2;">
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px; color:#ccc;">Olist Brazilian E-Commerce (Kaggle)</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">438,038 documents embedded</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">99,441 orders &nbsp;&#183;&nbsp; 99,441 customers</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">103,886 payments &nbsp;&#183;&nbsp; 99,224 reviews</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">32,951 products &nbsp;&#183;&nbsp; 3,095 sellers</div>
                <div>Stored in AWS S3 (us-west-1)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Improvements
        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px; margin-bottom:16px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:20px;">Improvements</div>
            <div style="font-size:13px; color:#888; line-height:2.2;">
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px; color:#ccc;">Add order_items for product-level analytics</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">Connect directly to RDS MySQL at query time</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">Chart generation for visual answers</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">Multi-turn conversation memory</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">User authentication and query history</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">Export answers to CSV or PDF</div>
                <div>Deploy on AWS ECS with a custom domain</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stack
        st.markdown("""
        <div style="border:1px solid #1a1a1a; padding:28px;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:20px;">Stack</div>
            <div style="font-family:IBM Plex Mono,monospace; font-size:12px; color:#666; line-height:2.4;">
                <div><span style="color:#333">LLM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">Groq / Llama 3.1 8B</span></div>
                <div><span style="color:#333">Embeddings&nbsp;</span><span style="color:#ccc">HuggingFace all-MiniLM-L6-v2</span></div>
                <div><span style="color:#333">Vector DB&nbsp;&nbsp;</span><span style="color:#ccc">Numpy cosine similarity</span></div>
                <div><span style="color:#333">Framework&nbsp;&nbsp;</span><span style="color:#ccc">LangChain LCEL</span></div>
                <div><span style="color:#333">Database&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">AWS RDS MySQL</span></div>
                <div><span style="color:#333">Storage&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">AWS S3</span></div>
                <div><span style="color:#333">BI&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">Microsoft Power BI</span></div>
                <div><span style="color:#333">UI&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">Streamlit Cloud</span></div>
                <div><span style="color:#333">Compute&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">Kaggle T4 GPU (embedding)</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# =============================================================================
# TAB 5 — POWER BI DASHBOARD
# =============================================================================
with tab5:
    st.markdown("""
    <p style="color: #444444; font-size: 13px; font-family: 'IBM Plex Mono', monospace;
              letter-spacing: 0.05em; margin-bottom: 32px;">
        Business intelligence dashboard — E-Commerce Analytics.
    </p>
    """, unsafe_allow_html=True)

    # ============================================================
    # ORDER STATUS LOOKUP (NOW AT THE TOP)
    # ============================================================
    st.subheader("Order Status Lookup")

    # Initialize session state for lookup results
    if 'lookup_result' not in st.session_state:
        st.session_state.lookup_result = None

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        customer_id_input = st.text_input("Enter Customer ID", key="cid")
    with col_l2:
        order_id_input = st.text_input("Enter Order ID", key="oid")

    if st.button("Check Status"):
        result = None
        if order_id_input:
            # Direct order ID lookup
            order_row = dfs['orders'][dfs['orders']['order_id'] == order_id_input]
            if not order_row.empty:
                status = order_row.iloc[0]['order_status']
                result = f"**Order ID:** {order_id_input}  \n**Status:** `{status}`"
            else:
                result = f"Order ID `{order_id_input}` not found."
        elif customer_id_input:
            # Find all orders for this customer
            customer_orders = dfs['orders'][dfs['orders']['customer_id'] == customer_id_input]
            if not customer_orders.empty:
                statuses = customer_orders['order_status'].value_counts().to_dict()
                result = f"**Customer ID:** {customer_id_input}  \n**Orders found:** {len(customer_orders)}  \n**Status breakdown:** {statuses}"
            else:
                result = f"Customer ID `{customer_id_input}` not found."
        else:
            result = "Please enter an Order ID or Customer ID."

        st.session_state.lookup_result = result

    # Display result
    if st.session_state.lookup_result:
        st.markdown(st.session_state.lookup_result)

    st.markdown("<hr style='margin: 40px 0; border-color: #1a1a1a;'>", unsafe_allow_html=True)

    # ============================================================
    # CHARTS (REVENUE & DELIVERY)
    # ============================================================
    # Compute monthly revenue
    orders_df = dfs['orders']
    payments_df = dfs['payments']
    merged = pd.merge(orders_df, payments_df, on='order_id')
    merged['order_purchase_timestamp'] = pd.to_datetime(merged['order_purchase_timestamp'])
    merged['month'] = merged['order_purchase_timestamp'].dt.to_period('M').astype(str)
    monthly_revenue = merged.groupby('month')['payment_value'].sum().reset_index()
    monthly_revenue = monthly_revenue.sort_values('month')

    # Compute average delivery days by state
    orders = dfs['orders'].copy()
    customers = dfs['customers']
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['delivery_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    delivery_merged = pd.merge(orders, customers, on='customer_id').dropna(subset=['delivery_days'])
    state_delivery = delivery_merged.groupby('customer_state')['delivery_days'].mean().reset_index()
    state_delivery = state_delivery.sort_values('delivery_days', ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Revenue by Month")
        st.line_chart(monthly_revenue.set_index('month'))
    with col2:
        st.subheader("Average Delivery Days by State (Top 10)")
        st.bar_chart(state_delivery.set_index('customer_state'))

    st.markdown("<hr style='margin: 40px 0; border-color: #1a1a1a;'>", unsafe_allow_html=True)

    # ============================================================
    # PRODUCT CATEGORY CHART
    # ============================================================
    st.subheader("Product Category Distribution")
    products_df = dfs['products']
    cat_counts = products_df['product_category_name'].value_counts().reset_index()
    cat_counts.columns = ['category', 'count']
    cat_counts = cat_counts.head(15)  # top 15 for readability
    st.bar_chart(cat_counts.set_index('category'))

    # st.image('images/dashboard.png', use_container_width=True)

    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #333333;
              letter-spacing: 0.08em; margin-top: 16px;">
        Data sourced from AWS S3.
    </p>
    """, unsafe_allow_html=True)
