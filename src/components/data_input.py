import streamlit as st
import pandas as pd

def render_data_input():
    """Render the data input section of the app."""
    st.header("Input Data")
    
    input_col1, input_col2 = st.columns([3, 2])
    
    with input_col1:
        input_mode = st.radio("Input Mode", ["Upload CSV", "Write Query"], horizontal=True)
        
        uploaded_file = None
        query_text = None
        df = None
        
        if input_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ CSV loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
                    
                except Exception as e:
                    st.error(f"❌ Error loading CSV: {str(e)}")
        else:
            query_text = st.text_area("Enter your query:", "Analyze 10 social media posts for sentiment and themes.")
    
    with input_col2:
        if df is not None and not df.empty:
            st.subheader("Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Display column info in an expander
            with st.expander("Column Information"):
                for col in df.columns:
                    st.write(f"- **{col}**: {df[col].dtype}")
    
    return df, query_text 