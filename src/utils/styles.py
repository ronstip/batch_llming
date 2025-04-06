import streamlit as st

def apply_custom_css():
    """Apply custom CSS for better styling."""
    st.markdown("""
    <style>
        /* Base styles and layout */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            max-width: 1200px;
        }
        
        /* Improved tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #f0f2f6;
            padding: 0.5rem 0.5rem 0;
            border-radius: 0.8rem 0.8rem 0 0;
            border-bottom: 1px solid #e0e3e9;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            border-radius: 0.6rem 0.6rem 0 0;
            font-weight: 500;
            padding: 0 20px;
            transition: all 0.2s ease;
            border: 1px solid transparent;
            border-bottom: none;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #ffffff;
            color: #1a73e8;
            font-weight: 600;
            border-color: #e0e3e9;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            background-color: white;
            border: 1px solid #e0e3e9;
            border-top: none;
            border-radius: 0 0 0.8rem 0.8rem;
            padding: 1rem;
        }
        
        /* Modern typography */
        h1 {
            color: #1a73e8;
            font-weight: 700;
            margin-bottom: 1.5rem;
            font-size: 2rem !important;
        }
        
        h2 {
            color: #174ea6;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-size: 1.5rem !important;
        }
        
        h3 {
            color: #1967d2;
            font-weight: 600;
            font-size: 1.2rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.75rem !important;
        }
        
        /* Clean, modern card styling */
        div.stMarkdown > div[data-testid="stMarkdownContainer"] > div:has(h3) {
            background-color: white;
            padding: 1.25rem;
            border-radius: 0.6rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05), 0 0 0 1px rgba(0,0,0,0.03);
            margin-bottom: 1rem;
            transition: all 0.2s ease;
            border-left: 3px solid #1a73e8;
        }
        
        div.stMarkdown > div[data-testid="stMarkdownContainer"] > div:has(h3):hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.02);
            transform: translateY(-2px);
        }
        
        /* Enhanced metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #1a73e8 !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: 500 !important;
            color: #5f6368 !important;
        }
        
        /* Button styling */
        button[kind="primary"] {
            background-color: #1a73e8 !important;
            border: none !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
        }
        
        button[kind="primary"]:hover {
            background-color: #1765cc !important;
            box-shadow: 0 2px 6px rgba(26, 115, 232, 0.4) !important;
        }
        
        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border-radius: 0.5rem !important;
            overflow: hidden !important;
        }
        
        /* Image styling */
        img {
            border-radius: 0.4rem;
            transition: transform 0.2s ease;
            max-width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
            margin: 0 auto;
            display: block;
        }
        
        .stImage {
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .stImage > div {
            display: flex;
            justify-content: center;
        }
        
        /* Improve caption alignment */
        .caption {
            text-align: center;
            margin-top: 0.2rem;
            font-size: 0.8rem;
            color: #666;
        }
        
        .hover-zoom img:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #1a73e8;
            background-color: #f8f9fa;
            border-radius: 0.4rem;
        }
        
        /* More compact layout for visualizations */
        .stPlotlyChart, .stDataFrame {
            margin-bottom: 1rem !important;
        }
        
        /* Custom tag styling for labels */
        .tag {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            margin: 0.1rem;
            border-radius: 1rem;
            background-color: #e8f0fe;
            color: #1967d2;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .tag-positive {
            background-color: #e6f4ea;
            color: #137333;
        }
        
        .tag-negative {
            background-color: #fce8e6;
            color: #c5221f;
        }
        
        .tag-neutral {
            background-color: #f1f3f4;
            color: #5f6368;
        }
        
        /* Image gallery styling */
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .image-card {
            border-radius: 0.4rem;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
            aspect-ratio: 1 / 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .image-card img {
            max-height: 120px;
            width: auto;
            object-fit: contain;
        }
        
        .image-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
    </style>
    """, unsafe_allow_html=True) 