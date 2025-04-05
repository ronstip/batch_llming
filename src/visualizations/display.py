import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import List, Dict, Any

def display_results(results_df, fields):
    """Display the analysis results in a dashboard format."""
    st.header("Analysis Results")
    
    # Skip if DataFrame is empty
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Get all field keys
    field_keys = [field['key'] for field in fields]
    
    # Key metrics at the top
    st.subheader("Key Metrics")
    metric_cols = st.columns(4)
    
    # Post count
    with metric_cols[0]:
        st.metric("Posts Analyzed", len(results_df))
    
    # Sentiment distribution if present
    if 'sentiment' in field_keys and 'sentiment' in results_df.columns:
        with metric_cols[1]:
            sentiment_counts = results_df['sentiment'].value_counts(normalize=True)
            positive_pct = sentiment_counts.get('positive', 0) * 100
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    
    # Engagement potential if present
    if 'engagement_potential' in field_keys and 'engagement_potential' in results_df.columns:
        with metric_cols[2]:
            engagement = results_df['engagement_potential'].value_counts(normalize=True)
            high_engagement = engagement.get('high', 0) * 100
            st.metric("High Engagement", f"{high_engagement:.1f}%")
    else:
        with metric_cols[2]:
            st.metric("Average Engagement", "N/A")

    # Most common themes if present
    if 'themes' in field_keys and 'themes' in results_df.columns:
        with metric_cols[3]:
            # Extract all themes and count them
            all_themes = []
            for themes_str in results_df['themes']:
                if isinstance(themes_str, str):
                    themes = [t.strip() for t in themes_str.split(',')]
                    all_themes.extend(themes)
            
            if all_themes:
                from collections import Counter
                most_common_theme = Counter(all_themes).most_common(1)[0][0]
                st.metric("Top Theme", most_common_theme)
    
    # Display tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Card View"])
    
    with tab1:
        st.dataframe(results_df, use_container_width=True)
        
        # Download button for results
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv",
        )
    
    with tab2:
        # Display cards for each post
        st.write("showing first 20 results")
        for i in range(0, min(len(results_df), 20), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < min(len(results_df), 20):
                    row = results_df.iloc[i + j]
                    with cols[j]:
                        st.markdown("### Post Analysis")
                        st.markdown(f"**Title:** {row.get('original_title', 'N/A')}")
                        
                        # Display analysis fields
                        for field in fields:
                            field_key = field['key']
                            if field_key in row:
                                st.markdown(f"**{field_key.replace('_', ' ').title()}:** {row[field_key]}")
                        
                        st.markdown("---")

def create_visualizations(df, fields):
    """Create visualizations based on the data and field types."""
    st.subheader("Visualizations")
    
    # Skip if DataFrame is empty
    if df.empty:
        st.warning("No data to visualize")
        return
    
    # Get all field keys that should be in the current results
    field_keys = [field['key'] for field in fields]
    available_fields = [field for field in fields if field['key'] in df.columns]
    
    # If no fields to visualize, show a message
    if not available_fields:
        st.warning("No fields to visualize in the current results")
        return
    
    # Create columns for visualizations
    viz_cols = st.columns(2)
    
    col_index = 0
    
    for field in available_fields:
        field_key = field['key']
        field_type = field['type']
        
        try:
            with viz_cols[col_index % 2]:
                if field_type == 'enum' or field_type == 'str':
                    # Bar chart for categorical data
                    st.subheader(f"{field_key} Distribution")
                    value_counts = df[field_key].value_counts().reset_index()
                    value_counts.columns = [field_key, 'count']
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=value_counts, x=field_key, y='count', ax=ax)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                elif field_type == 'int' or field_type == 'float':
                    # Histogram for numerical data
                    st.subheader(f"{field_key} Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(df[field_key], kde=True, ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Also show basic statistics
                    st.write(f"**Mean:** {df[field_key].mean():.2f}")
                    st.write(f"**Median:** {df[field_key].median():.2f}")
                    st.write(f"**Min:** {df[field_key].min():.2f}")
                    st.write(f"**Max:** {df[field_key].max():.2f}")
                
                col_index += 1
                
        except Exception as e:
            st.error(f"Error creating visualization for {field_key}: {str(e)}")

def display_errors(results_df):
    """Display any errors encountered during processing."""
    if 'error' not in results_df.columns:
        return
    
    error_rows = results_df[results_df['error'].notna()]
    
    if not error_rows.empty:
        with st.expander("⚠️ Errors detected in processing", expanded=True):
            st.error(f"Found {len(error_rows)} errors in processing")
            for i, (idx, row) in enumerate(error_rows.iterrows()):
                st.write(f"### Error {i+1}:")
                st.write(f"**Error**: {row.get('error', 'Unknown error')}")
                st.write(f"**Type**: {row.get('error_type', 'Unknown')}")
                if 'traceback' in row:
                    st.code(row['traceback'], language='python') 