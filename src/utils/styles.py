import streamlit as st

def apply_custom_css():
    """Apply custom CSS for better styling."""
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
        }
        
        /* Improved header styling */
        h1 {
            color: #1E88E5;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        
        h2 {
            color: #0D47A1;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: #1565C0;
            font-weight: 600;
        }
        
        /* Card styling */
        div.stMarkdown > div[data-testid="stMarkdownContainer"] > div:has(h3) {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #1E88E5 !important;
        }
    </style>
    """, unsafe_allow_html=True) 