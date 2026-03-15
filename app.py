import streamlit as st
import pandas as pd
import json
import time

# Import RAG Systems
from prolog_rag_project.baselines.naive_rag import NaiveRAG
from prolog_rag_project.baselines.graph_rag import SOTAGraphRAG
from prolog_rag_project.baselines.corrective_rag import CorrectiveRAG
from prolog_rag_project.baselines.contextual_rag import ContextualRAG
from prolog_rag_project.core.prolog_rag import PrologRAG

# Page Config
st.set_page_config(
    page_title="Prolog-RAG vs SOTA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    .proof-box {
        background-color: #1E2329;
        border-left: 4px solid #00E676;
        padding: 1rem;
        border-radius: 4px;
        font-family: monospace;
        color: #E2E8F0;
        margin-bottom: 1rem;
    }
    .baseline-box {
        background-color: #1A1F2B;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .prolog-badge {
        background-color: #00E676;
        color: #000000;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Models (Cached to avoid reloading on every UI interaction)
@st.cache_resource
def load_models():
    return {
        "Prolog-RAG (Reasoning)": PrologRAG(),
        "Contextual RAG (Anthropic)": ContextualRAG(),
        "Corrective RAG (CRAG)": CorrectiveRAG(),
        "GraphRAG (SOTA)": SOTAGraphRAG(),
        "Naive RAG (Baseline)": NaiveRAG()
    }

def load_sample_questions():
    try:
        with open("test_questions.json", "r") as f:
            return json.load(f)
    except:
        return []

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Prolog-logo.png", width=100)
    st.markdown("### Systems Loaded")
    st.markdown("🟢 Prolog-RAG (Active)\n\n🔵 Contextual RAG\n\n🔵 CRAG\n\n🔵 GraphRAG\n\n🔵 Naive RAG")
    
    st.markdown("### Settings")
    show_proofs = st.toggle("Show Prolog Proof Traces", value=True)
    
    st.markdown("---")
    st.markdown("*Built with Streamlit & SWI-Prolog*")

# Main UI
st.markdown('<p class="main-header">🧠 Prolog-RAG Arena</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Evaluating Symbolic Mathematical Reasoning vs. LLM Baselines in Financial RAG</p>', unsafe_allow_html=True)

# Load context
st.info("System is ready. Connected to ChromaDB 'finqa' & SWI-Prolog Knowledge Base.")

try:
    with st.spinner("Initializing AI Models..."):
        models = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
    
samples = load_sample_questions()
sample_texts = [q['question'] for q in samples]

# Query Input Area
col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input("Enter a financial question:", placeholder="e.g., What was the revenue growth rate between 2022 and 2023?")
    
    # Quick select
    if samples:
        st.markdown("**Or select a benchmark question:**")
        selected_sample = st.selectbox("", ["-- Choose a question --"] + sample_texts, label_visibility="collapsed")
        if selected_sample != "-- Choose a question --":
            user_query = selected_sample

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    baseline_select = st.selectbox("Compare against:", list(models.keys())[1:])
    run_btn = st.button("🚀 Execute Query", use_container_width=True, type="primary")

# Execution and Results
if run_btn and user_query:
    st.markdown("---")
    
    res_col1, res_col2 = st.columns(2)
    
    # 1. Prolog-RAG Execution
    with res_col1:
        st.markdown("### <span class='prolog-badge'>Prolog-RAG</span>", unsafe_allow_html=True)
        
        start_time = time.time()
        with st.spinner("Executing Prolog reasoning..."):
            try:
                prolog_res = models["Prolog-RAG (Reasoning)"].query(user_query)
                p_time = time.time() - start_time
                
                method = prolog_res.get('route', 'UNKNOWN')
                st.caption(f"⏱️ {p_time:.2f}s | Route: {method}")
                
                st.markdown(f"**Answer:**\n\n{prolog_res.get('answer', 'Failed to generate answer.')}")
                
                if show_proofs and 'proof_trace' in prolog_res and prolog_res['proof_trace']:
                    st.markdown("#### 🔍 Logical Proof Trace")
                    st.markdown('<div class="proof-box">', unsafe_allow_html=True)
                    st.code(prolog_res['proof_trace'], language="prolog")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prolog-RAG Error: {str(e)}")

    # 2. Baseline Execution
    with res_col2:
        st.markdown(f"### 🔵 {baseline_select}")
        
        start_time = time.time()
        with st.spinner(f"Executing {baseline_select}..."):
            try:
                base_res = models[baseline_select].query(user_query)
                b_time = time.time() - start_time
                
                method = base_res.get('method', 'UNKNOWN')
                st.caption(f"⏱️ {b_time:.2f}s | Strategy: {method}")
                
                st.markdown(f"**Answer:**\n\n{base_res.get('answer', 'Failed to generate answer.')}")
                
            except Exception as e:
                st.error(f"Baseline Error: {str(e)}")
