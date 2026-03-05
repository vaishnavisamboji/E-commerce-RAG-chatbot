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
    page_title="DataChat",
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
        Brazilian E-Commerce
    </span>
    <h1 style="font-family: 'IBM Plex Sans', sans-serif; font-size: 28px;
               font-weight: 300; color: #ffffff; margin: 6px 0 0 0; letter-spacing: -0.02em;">
        DataChat
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
    st.markdown("""
    <p style="color: #444444; font-size: 13px; font-family: 'IBM Plex Mono', monospace;
              letter-spacing: 0.05em; margin-bottom: 40px;">
        Architecture and components powering this application.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="border: 1px solid #1a1a1a; padding: 28px; margin-bottom: 16px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">Retrieval</p>
            <p style="font-size: 18px; font-weight: 300; color: #ffffff; margin-bottom: 8px;">
                Semantic Search</p>
            <p style="font-size: 13px; color: #555555; line-height: 1.6;">
                HuggingFace all-MiniLM-L6-v2 embeds 438,000+ documents.
                Embeddings pre-built on GPU and saved to disk.
                Cosine similarity via numpy dot product at query time.
            </p>
        </div>
        <div style="border: 1px solid #1a1a1a; padding: 28px; margin-bottom: 16px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">Generation</p>
            <p style="font-size: 18px; font-weight: 300; color: #ffffff; margin-bottom: 8px;">
                Groq — Llama 3.1 8B</p>
            <p style="font-size: 13px; color: #555555; line-height: 1.6;">
                Free tier. Sub-second inference via Groq LPU hardware.
                Temperature 0 for deterministic, factual answers.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border: 1px solid #1a1a1a; padding: 28px; margin-bottom: 16px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">Query Router</p>
            <p style="font-size: 18px; font-weight: 300; color: #ffffff; margin-bottom: 8px;">
                3-Way Routing</p>
            <p style="font-size: 13px; color: #555555; line-height: 1.6;">
                Exact ID via regex — bypasses embeddings entirely.
                Aggregate keywords route to pandas code generation.
                All other questions use RAG retrieval.
            </p>
        </div>
        <div style="border: 1px solid #1a1a1a; padding: 28px; margin-bottom: 16px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">Code Generation</p>
            <p style="font-size: 18px; font-weight: 300; color: #ffffff; margin-bottom: 8px;">
                Text-to-Pandas</p>
            <p style="font-size: 13px; color: #555555; line-height: 1.6;">
                LLM receives full schema with types, samples and join relationships.
                Generates pandas code, executes against real dataframes, auto-retries on error.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
              letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">Pipeline flow</p>
    <div style="border: 1px solid #1a1a1a; padding: 28px; font-family: 'IBM Plex Mono', monospace;
                font-size: 12px; color: #666666; line-height: 2.4;">
        Question &nbsp;<span style="color:#333">—</span>&nbsp;
        <span style="color:#888">ID detected?</span> &nbsp;<span style="color:#333">—&gt;</span>&nbsp;
        <span style="color:#ccc">Direct dataframe scan</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <span style="color:#888">Aggregate?</span> &nbsp;<span style="color:#333">—&gt;</span>&nbsp;
        <span style="color:#ccc">Codegen &rarr; exec &rarr; format</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <span style="color:#888">Else</span> &nbsp;<span style="color:#333">—&gt;</span>&nbsp;
        <span style="color:#ccc">Embed &rarr; retrieve &rarr; generate</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
              letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">Loaded data</p>
    """, unsafe_allow_html=True)
    cols = st.columns(len(dfs))
    for i, (name, df) in enumerate(dfs.items()):
        cols[i].metric(name, f'{len(df):,}')

# =============================================================================
# TAB 4 — ABOUT
# =============================================================================
with tab4:
    st.markdown("""
    <p style="color: #444444; font-size: 13px; font-family: 'IBM Plex Mono', monospace;
              letter-spacing: 0.05em; margin-bottom: 40px;">
        What this is, why it matters, and where it goes next.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div style="margin-bottom: 40px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">What this is</p>
            <p style="font-size: 14px; color: #cccccc; line-height: 1.8; font-weight: 300;">
                DataChat is a natural language interface for structured business data.
                Instead of writing SQL or pandas, analysts type plain English questions
                and receive accurate, grounded answers drawn directly from the data.
            </p>
            <p style="font-size: 14px; color: #cccccc; line-height: 1.8; font-weight: 300; margin-top: 16px;">
                It combines semantic retrieval with code generation — using the right tool
                for each type of question. Aggregate questions trigger real pandas execution
                against the full dataset. Lookup questions retrieve the most relevant records.
                Exact IDs bypass embeddings entirely for instant results.
            </p>
        </div>
        <div style="margin-bottom: 40px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">Business value</p>
            <p style="font-size: 14px; color: #cccccc; line-height: 1.8; font-weight: 300;">
                Analysts spend significant time writing repetitive queries for stakeholders.
                This tool removes that bottleneck — a product manager can query delivery
                performance, a finance team can pull monthly revenue, and a support agent
                can look up an order status, all without engineering involvement.
            </p>
            <p style="font-size: 14px; color: #cccccc; line-height: 1.8; font-weight: 300; margin-top: 16px;">
                The architecture is database-agnostic. Connecting to a production MySQL,
                Postgres, or Snowflake instance requires minimal changes to the data loading
                layer — the rest of the pipeline remains unchanged.
            </p>
        </div>
        <div style="margin-bottom: 40px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;">How it was built</p>
            <p style="font-size: 14px; color: #cccccc; line-height: 1.8; font-weight: 300;">
                Built over 6 hours on a Kaggle T4 GPU notebook. Data from the Olist
                Brazilian E-Commerce dataset stored in AWS S3. The pipeline evolved
                through several iterations — from Phi-2 with ChromaDB, to FAISS,
                to pure numpy, before landing on the current architecture:
                LangChain + Groq + numpy vector search with a hybrid query router.
            </p>
            <p style="font-size: 14px; color: #cccccc; line-height: 1.8; font-weight: 300; margin-top: 16px;">
                Each component was chosen for zero-cost operation on free tiers:
                Groq for inference, HuggingFace for embeddings, S3 for storage.
                Total infrastructure cost to run this demo: $0.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border: 1px solid #1a1a1a; padding: 28px; margin-bottom: 16px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 20px;">Improvements</p>
            <div style="font-size: 13px; color: #888888; line-height: 2.2;">
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px; color:#cccccc;">
                    Add order_items for product analytics</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">
                    Connect directly to MySQL / Postgres</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">
                    Chart generation for visual answers</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">
                    Multi-turn conversation memory</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">
                    User authentication and query history</div>
                <div style="border-bottom:1px solid #111; padding-bottom:10px; margin-bottom:10px;">
                    Export answers to CSV or PDF</div>
                <div>Deploy on AWS ECS with a custom domain</div>
            </div>
        </div>
        <div style="border: 1px solid #1a1a1a; padding: 28px;">
            <p style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444444;
                      letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 20px;">Stack</p>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #666666; line-height: 2.4;">
                <div><span style="color:#333">LLM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">Groq / Llama 3.1</span></div>
                <div><span style="color:#333">Embeddings&nbsp;</span><span style="color:#ccc">HuggingFace MiniLM</span></div>
                <div><span style="color:#333">Vector DB&nbsp;&nbsp;</span><span style="color:#ccc">Numpy dot product</span></div>
                <div><span style="color:#333">Framework&nbsp;&nbsp;</span><span style="color:#ccc">LangChain LCEL</span></div>
                <div><span style="color:#333">Storage&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">AWS S3</span></div>
                <div><span style="color:#333">UI&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">Streamlit</span></div>
                <div><span style="color:#333">Compute&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#ccc">Kaggle T4 GPU</span></div>
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
