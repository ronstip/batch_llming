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
    """Create visualizations based on the data and field types with modern styling."""
    st.header("📊 Insights & Visualizations")
    
    # Skip if DataFrame is empty
    if df.empty:
        st.info("No data available for visualizations")
        return
    
    # Get all field keys that should be in the current results
    field_keys = [field['key'] for field in fields]
    available_fields = [field for field in fields if field['key'] in df.columns]
    
    # If no fields to visualize, show a message
    if not available_fields:
        st.info("No fields available for visualization in the current results")
        return
    
    # Modern color palette
    colors = ['#1a73e8', '#34a853', '#fbbc04', '#ea4335', '#9c27b0', '#3f51b5', '#03a9f4', '#009688']
    
    # Create a tabbed interface for different visualization categories
    viz_tabs = st.tabs(["Categorical", "Numerical", "Text Analysis"])
    
    with viz_tabs[0]:
        st.markdown("### Categorical Distributions")
        
        # Filter categorical fields
        categorical_fields = [field for field in available_fields 
                             if field['type'] == 'enum' or field['type'] == 'str']
        
        if not categorical_fields:
            st.info("No categorical fields found in the data")
        else:
            # Create grid layout
            for i in range(0, len(categorical_fields), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(categorical_fields):
                        field = categorical_fields[i + j]
                        field_key = field['key']
                        
                        with cols[j]:
                            try:
                                # Calculate value counts
                                value_counts = df[field_key].value_counts().reset_index()
                                value_counts.columns = [field_key, 'count']
                                
                                # Set title with field name
                                st.markdown(f"#### {field_key.replace('_', ' ').title()}")
                                
                                # Create bar chart with modern styling
                                fig, ax = plt.subplots(figsize=(8, 4))
                                
                                # Use the color palette
                                color_idx = (i + j) % len(colors)
                                bars = sns.barplot(
                                    data=value_counts, 
                                    x=field_key, 
                                    y='count', 
                                    ax=ax,
                                    color=colors[color_idx],
                                    alpha=0.8
                                )
                                
                                # Add value labels on top of bars
                                for bar in bars.patches:
                                    bars.annotate(
                                        f'{int(bar.get_height())}',
                                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                        ha='center', va='bottom',
                                        fontsize=9, color='#555'
                                    )
                                
                                # Clean up the chart
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_color('#ddd')
                                ax.spines['bottom'].set_color('#ddd')
                                ax.tick_params(axis='x', labelrotation=45, labelsize=9, colors='#555')
                                ax.tick_params(axis='y', labelsize=9, colors='#555')
                                ax.set_title(f"{field_key.replace('_', ' ').title()} Distribution", fontsize=12, pad=10)
                                ax.set_xlabel('')
                                ax.set_ylabel('Count', fontsize=10, color='#666')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show percentage breakdown
                                total = value_counts['count'].sum()
                                value_counts['percentage'] = (value_counts['count'] / total * 100).round(1)
                                
                                # Format as colored tags
                                tags_html = ""
                                for _, row in value_counts.iterrows():
                                    color_class = ""
                                    if field_key == 'sentiment':
                                        if row[field_key].lower() == 'positive':
                                            color_class = "tag-positive"
                                        elif row[field_key].lower() == 'negative':
                                            color_class = "tag-negative"
                                        else:
                                            color_class = "tag-neutral"
                                    
                                    tags_html += f'<span class="tag {color_class}">{row[field_key]}: {row["percentage"]}%</span> '
                                
                                st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error visualizing {field_key}: {e}")
    
    with viz_tabs[1]:
        st.markdown("### Numerical Analysis")
        
        # Filter numerical fields
        numerical_fields = [field for field in available_fields 
                          if field['type'] == 'int' or field['type'] == 'float']
        
        if not numerical_fields:
            st.info("No numerical fields found in the data")
        else:
            # Create grid layout
            for i in range(0, len(numerical_fields), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(numerical_fields):
                        field = numerical_fields[i + j]
                        field_key = field['key']
                        
                        with cols[j]:
                            try:
                                # Set title with field name
                                st.markdown(f"#### {field_key.replace('_', ' ').title()}")
                                
                                # Create modern histogram with KDE
                                fig, ax = plt.subplots(figsize=(8, 4))
                                
                                # Use the color palette with transparency for density
                                color_idx = (i + j) % len(colors)
                                main_color = colors[color_idx]
                                
                                # Draw histogram with KDE
                                sns.histplot(
                                    df[field_key], 
                                    kde=True, 
                                    ax=ax,
                                    color=main_color,
                                    kde_kws={'color': '#444', 'lw': 1, 'alpha': 0.8},
                                    alpha=0.6
                                )
                                
                                # Clean up the chart
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_color('#ddd')
                                ax.spines['bottom'].set_color('#ddd')
                                ax.tick_params(axis='x', labelsize=9, colors='#555')
                                ax.tick_params(axis='y', labelsize=9, colors='#555')
                                ax.set_title(f"{field_key.replace('_', ' ').title()} Distribution", fontsize=12)
                                ax.set_xlabel(field_key.replace('_', ' ').title(), fontsize=10, color='#666')
                                ax.set_ylabel('Count', fontsize=10, color='#666')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show statistics with modern cards
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    mean_val = df[field_key].mean()
                                    median_val = df[field_key].median()
                                    
                                    st.markdown(f"""
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:10px;">
                                        <div style="font-size:0.8rem; color:#666;">Mean</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{mean_val:.2f}</div>
                                    </div>
                                    
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px;">
                                        <div style="font-size:0.8rem; color:#666;">Median</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{median_val:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    min_val = df[field_key].min()
                                    max_val = df[field_key].max()
                                    
                                    st.markdown(f"""
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:10px;">
                                        <div style="font-size:0.8rem; color:#666;">Min</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{min_val:.2f}</div>
                                    </div>
                                    
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px;">
                                        <div style="font-size:0.8rem; color:#666;">Max</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{max_val:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error visualizing {field_key}: {e}")
    
    with viz_tabs[2]:
        st.markdown("### Text & Theme Analysis")
        
        # Look for themes field or other text fields
        themes_field = next((field for field in available_fields if field['key'] == 'themes'), None)
        
        if themes_field and 'themes' in df.columns:
            try:
                # Extract all themes and count them
                all_themes = []
                for themes_str in df['themes']:
                    if isinstance(themes_str, str):
                        themes = [t.strip() for t in themes_str.split(',')]
                        all_themes.extend(themes)
                
                if all_themes:
                    from collections import Counter
                    theme_counts = Counter(all_themes).most_common(15)  # Top 15 themes
                    
                    # Create DataFrame for visualization
                    theme_df = pd.DataFrame(theme_counts, columns=['theme', 'count'])
                    
                    st.markdown("#### Most Common Themes")
                    
                    # Create horizontal bar chart for themes
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Sort by count
                    theme_df = theme_df.sort_values('count')
                    
                    # Plot horizontal bars with gradient colors
                    bars = sns.barplot(
                        data=theme_df,
                        y='theme',
                        x='count',
                        ax=ax,
                        palette=sns.color_palette("Blues_d", len(theme_df))
                    )
                    
                    # Add count labels
                    for i, bar in enumerate(bars.patches):
                        bars.text(
                            bar.get_width() + 0.3,
                            bar.get_y() + bar.get_height()/2,
                            f'{int(bar.get_width())}',
                            ha='left', va='center',
                            fontsize=9, color='#555'
                        )
                    
                    # Clean up the chart
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#ddd')
                    ax.spines['bottom'].set_color('#ddd')
                    ax.tick_params(axis='y', labelsize=10, colors='#444')
                    ax.tick_params(axis='x', labelsize=9, colors='#555')
                    ax.set_title("Top Themes Mentioned", fontsize=14)
                    ax.set_xlabel('Count', fontsize=10, color='#666')
                    ax.set_ylabel('')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Visual display of themes as tags
                    st.markdown("#### Theme Cloud")
                    
                    # Generate HTML for theme tags with size based on frequency
                    theme_html = ""
                    max_count = max(t[1] for t in theme_counts)
                    
                    for theme, count in theme_counts:
                        # Scale size between 0.8 and 1.5rem based on count
                        size = 0.8 + (count / max_count) * 0.7
                        # Calculate color intensity based on count
                        intensity = int(50 + (count / max_count) * 200)
                        theme_html += f'<span class="tag" style="font-size:{size}rem; background-color:rgba(25, 103, 210, {count/max_count*0.8+0.2});">{theme} ({count})</span> '
                    
                    st.markdown(f"<div style='line-height: 2.2; padding: 15px;'>{theme_html}</div>", unsafe_allow_html=True)
                else:
                    st.info("No theme data available for analysis")
            
            except Exception as e:
                st.error(f"Error analyzing themes: {e}")
        
        # Look for other text fields to analyze (like image_description)
        text_field = next((field for field in available_fields 
                        if field['key'] in ['image_description', 'target_audience'] 
                        and field['key'] in df.columns), None)
        
        if text_field:
            field_key = text_field['key']
            
            st.markdown(f"#### {field_key.replace('_', ' ').title()} Word Cloud")
            
            try:
                # Combine all text from the field
                all_text = " ".join([str(text) for text in df[field_key] if pd.notna(text)])
                
                if all_text.strip():
                    # Generate word cloud (if wordcloud library is available)
                    try:
                        from wordcloud import WordCloud
                        import matplotlib.pyplot as plt
                        
                        wordcloud = WordCloud(
                            width=800, height=400,
                            background_color='white',
                            colormap='viridis',
                            max_words=100,
                            contour_width=1,
                            contour_color='#ddd'
                        ).generate(all_text)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f"{field_key.replace('_', ' ').title()} Word Cloud", fontsize=14, pad=20)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except ImportError:
                        st.info("WordCloud library not available. Install with: pip install wordcloud")
                else:
                    st.info(f"No text data available in the {field_key} field")
            
            except Exception as e:
                st.error(f"Error creating word cloud: {e}")

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