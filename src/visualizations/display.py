import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import List, Dict, Any
import os
from PIL import Image
import io
import re
import numpy as np

def display_results(results_df, fields):
    """Display the analysis results in a dashboard format."""
    st.header("Analysis Results")
    
    # Skip if DataFrame is empty
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Get all field keys
    field_keys = [field['key'] for field in fields]
    
    # Detect image columns - columns that might contain image paths or image data
    image_columns = []
    
    # First check if any columns are explicitly passed via session state
    if 'display_image_columns' in st.session_state and st.session_state.display_image_columns:
        image_columns = st.session_state.display_image_columns
    
    # If no columns found yet, check if any fields are explicitly marked as image type
    if not image_columns:
        for field in fields:
            if field.get('type') == 'image' and field['key'] in results_df.columns:
                image_columns.append(field['key'])
    
    # Then try to auto-detect image columns if none are explicitly marked
    if not image_columns:
        for column in results_df.columns:
            # Skip columns that are clearly not image paths/data
            if results_df[column].dtype != 'object':
                continue
                
            # Check if column values look like image paths
            sample_values = results_df[column].dropna().head(10).tolist()
            if sample_values:
                # Check if values look like image paths or URLs
                if any(isinstance(val, str) and (
                       any(val.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']) or
                       re.match(r'https?://.+\.(jpg|jpeg|png|gif|webp|bmp)', val.lower())
                      ) for val in sample_values):
                    image_columns.append(column)
    
    st.write(f"Debug: Found image columns: {image_columns}")
    
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
    if image_columns:
        tab1, tab2, tab3 = st.tabs(["Table View", "Card View", "Image Gallery"])
    else:
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
    
    def is_valid_image_path(path):
        """Check if a path is a valid image path that can be displayed."""
        if not isinstance(path, str):
            return False
        
        # Handle URLs
        if path.startswith(('http://', 'https://')):
            return any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'])
        
        # Handle local files - try both absolute path and relative to current directory
        return (os.path.exists(path) and any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'])) or \
               (os.path.exists(os.path.join(os.getcwd(), path)) and any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']))
    
    with tab2:
        # Display cards for each post
        st.write("showing first 20 results")
        
        # If no valid images, display text for debugging
        valid_images_found = False
        
        for i in range(0, min(len(results_df), 20), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < min(len(results_df), 20):
                    row = results_df.iloc[i + j]
                    with cols[j]:
                        st.markdown("### Post Analysis")
                        
                        # Display images if available
                        for img_col in image_columns:
                            if img_col in row and row[img_col]:
                                try:
                                    if is_valid_image_path(row[img_col]):
                                        st.image(row[img_col], width=150, caption=f"{img_col}")
                                        valid_images_found = True
                                    else:
                                        # To reduce clutter, print to console instead of UI
                                        print(f"Image path not accessible: {row[img_col]}")
                                except Exception as e:
                                    print(f"Could not display image: {e}")
                        
                        st.markdown(f"**Title:** {row.get('original_title', 'N/A')}")
                        
                        # Display analysis fields
                        for field in fields:
                            field_key = field['key']
                            if field_key in row and field_key not in image_columns:
                                st.markdown(f"**{field_key.replace('_', ' ').title()}:** {row[field_key]}")
                        
                        st.markdown("---")
        
        if not valid_images_found and image_columns:
            st.info("No valid images found in the data. Please verify image paths.")
    
    # Image Gallery Tab
    with tab3:
        # Check for image columns in the gallery
        if not image_columns:
            st.info("No images available in the current dataset")
        else:
            st.markdown("### Image Gallery")
            
            # Show what image columns were detected
            st.info(f"Found image columns: {', '.join(image_columns)}")
            
            # Group by image type
            img_cols_per_page = 4
            img_rows_per_page = 4
            items_per_page = img_cols_per_page * img_rows_per_page
            
            # Get all valid image paths from the dataset
            all_images = []
            for _, row in results_df.iterrows():
                for col in image_columns:
                    if col in row:
                        img_path = row[col]
                        if pd.notna(img_path) and str(img_path).strip():
                            all_images.append({
                                'path': img_path,
                                'column': col.replace('original_', ''),
                                'row_idx': _
                            })
                            
            if not all_images:
                st.warning("Found image columns but no valid image paths in the data")
                st.write("Check if your image paths are correct and accessible")
            else:
                st.success(f"Found {len(all_images)} images in the dataset")
                
                # Allow selecting how many columns to display
                cols_per_row = st.slider("Images per row", min_value=2, max_value=6, value=4)
                
                # Create rows of images
                rows = []
                current_row = []
                
                for img_info in all_images:
                    current_row.append(img_info)
                    
                    if len(current_row) == cols_per_row:
                        rows.append(current_row)
                        current_row = []
                
                # Add any remaining images
                if current_row:
                    rows.append(current_row)
                
                # Display the images in rows
                for row_images in rows:
                    cols = st.columns(cols_per_row)
                    for i, img_info in enumerate(row_images):
                        with cols[i]:
                            try:
                                img_path = img_info['path']
                                st.image(img_path, width=150)
                                st.caption(f"Column: {img_info['column']}")
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                                st.code(img_path, language=None)

def create_visualizations(df, fields):
    """Create visualizations based on the data and field types with modern styling."""
    st.header("📊 Insights & Visualizations")
    
    # Skip if DataFrame is empty
    if df.empty:
        st.info("No data available for visualizations")
        return
    
    # Import required libraries upfront to avoid scoping issues
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    
    # Get all field keys that should be in the current results
    field_keys = [field['key'] for field in fields]
    available_fields = [field for field in fields if field['key'] in df.columns]
    
    # If no fields to visualize, show a message
    if not available_fields:
        st.info("No fields available for visualization in the current results")
        return
    
    # Modern color palette
    colors = ['#1a73e8', '#34a853', '#fbbc04', '#ea4335', '#9c27b0', '#3f51b5', '#03a9f4', '#009688']
    
    # Clean the dataframe - handle nulls and convert types appropriately
    df_viz = df.copy()
    
    # Helper function to detect if a column has numeric data despite being object type
    def is_numeric_column(column):
        try:
            # Try to convert to numeric, but handle mixed data gracefully
            pd.to_numeric(df_viz[column], errors='coerce')
            # If less than 30% of values are NaN after conversion, consider it numeric
            null_pct = df_viz[column].isna().mean()
            return null_pct < 0.3
        except:
            return False
    
    # Create a tabbed interface for different visualization categories
    viz_tabs = st.tabs(["Categorical", "Numerical", "Relationships", "Text Analysis", "Advanced"])
    
    # === CATEGORICAL VISUALIZATIONS ===
    with viz_tabs[0]:
        st.markdown("### Categorical Distributions")
        
        # Filter categorical fields
        categorical_fields = [field for field in available_fields 
                             if field['type'] == 'enum' or field['type'] == 'str']
        
        if not categorical_fields:
            st.info("No categorical fields found in the data")
        else:
            # Add visualization type selector
            viz_type = st.radio(
                "Chart type:",
                ["Bar Chart", "Pie Chart", "Donut Chart"],
                horizontal=True
            )
            
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
                                value_counts = df_viz[field_key].value_counts().reset_index()
                                value_counts.columns = [field_key, 'count']
                                
                                # Calculate percentages
                                total = value_counts['count'].sum()
                                value_counts['percentage'] = (value_counts['count'] / total * 100).round(1)
                                
                                # Limit to top 10 categories if there are too many
                                if len(value_counts) > 10:
                                    top_n = value_counts.iloc[:9].copy()
                                    other_count = value_counts.iloc[9:]['count'].sum()
                                    other_row = pd.DataFrame({
                                        field_key: ['Other'],
                                        'count': [other_count],
                                        'percentage': [(other_count / total * 100).round(1)]
                                    })
                                    value_counts = pd.concat([top_n, other_row], ignore_index=True)
                                
                                # Set title with field name
                                st.markdown(f"#### {field_key.replace('_', ' ').title()}")
                                
                                if viz_type == "Bar Chart":
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
                                    ax.set_xlabel('')
                                    ax.set_ylabel('Count', fontsize=10, color='#666')
                                
                                elif viz_type in ["Pie Chart", "Donut Chart"]:
                                    # Create pie/donut chart
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    
                                    # Create color palette with enough colors
                                    color_palette = sns.color_palette("husl", len(value_counts))
                                    
                                    # Plot the pie chart
                                    wedges, texts, autotexts = ax.pie(
                                        value_counts['count'],
                                        labels=value_counts[field_key],
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=color_palette,
                                        wedgeprops={'width': 0.5 if viz_type == "Donut Chart" else 1, 'edgecolor': 'w', 'linewidth': 1},
                                        textprops={'fontsize': 9}
                                    )
                                    
                                    # Style the percentages
                                    for autotext in autotexts:
                                        autotext.set_fontsize(8)
                                        autotext.set_weight('bold')
                                    
                                    # Equal aspect ratio ensures that pie is drawn as a circle
                                    ax.set_aspect('equal')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Format as colored tags
                                tags_html = ""
                                for _, row in value_counts.iterrows():
                                    val = row[field_key]
                                    color_class = ""
                                    if field_key == 'sentiment':
                                        if val.lower() == 'positive':
                                            color_class = "tag-positive"
                                        elif val.lower() == 'negative':
                                            color_class = "tag-negative"
                                        else:
                                            color_class = "tag-neutral"
                                    
                                    tags_html += f'<span class="tag {color_class}">{val}: {row["percentage"]}%</span> '
                                
                                st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error visualizing {field_key}: {e}")
    
    # === NUMERICAL VISUALIZATIONS ===
    with viz_tabs[1]:
        st.markdown("### Numerical Analysis")
        
        # Filter numerical fields - include both declared numeric and those that can be converted
        numerical_fields = [field for field in available_fields 
                          if field['type'] in ['int', 'float']]
        
        # Add auto-detected numeric fields
        for field in available_fields:
            if field['type'] not in ['int', 'float'] and field['key'] in df_viz.columns:
                if is_numeric_column(field['key']):
                    numerical_fields.append(field)
        
        if not numerical_fields:
            st.info("No numerical fields found in the data")
        else:
            # Add visualization type selector
            num_viz_type = st.radio(
                "Distribution chart type:",
                ["Histogram", "Box Plot", "Violin Plot"],
                horizontal=True
            )
            
            # Create grid layout
            for i in range(0, len(numerical_fields), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(numerical_fields):
                        field = numerical_fields[i + j]
                        field_key = field['key']
                        
                        with cols[j]:
                            try:
                                # Convert to numeric if needed
                                if df_viz[field_key].dtype == 'object':
                                    df_viz[field_key] = pd.to_numeric(df_viz[field_key], errors='coerce')
                                
                                # Set title with field name
                                st.markdown(f"#### {field_key.replace('_', ' ').title()}")
                                
                                # Create visualization based on selected type
                                fig, ax = plt.subplots(figsize=(8, 4))
                                
                                # Use the color palette with transparency for density
                                color_idx = (i + j) % len(colors)
                                main_color = colors[color_idx]
                                
                                if num_viz_type == "Histogram":
                                    # Draw histogram with KDE
                                    sns.histplot(
                                        df_viz[field_key].dropna(), 
                                        kde=True, 
                                        ax=ax,
                                        color=main_color,
                                        kde_kws={'color': '#444', 'lw': 1, 'alpha': 0.8},
                                        alpha=0.6
                                    )
                                elif num_viz_type == "Box Plot":
                                    # Draw box plot
                                    sns.boxplot(
                                        x=df_viz[field_key].dropna(),
                                        ax=ax,
                                        color=main_color,
                                        width=0.4
                                    )
                                    
                                    # Add strip plot on top for individual points
                                    sns.stripplot(
                                        x=df_viz[field_key].dropna(),
                                        ax=ax,
                                        color='#333',
                                        alpha=0.3,
                                        size=3
                                    )
                                elif num_viz_type == "Violin Plot":
                                    # Draw violin plot
                                    sns.violinplot(
                                        x=df_viz[field_key].dropna(),
                                        ax=ax,
                                        color=main_color,
                                        inner="quartile"
                                    )
                                
                                # Clean up the chart
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_color('#ddd')
                                ax.spines['bottom'].set_color('#ddd')
                                ax.tick_params(axis='x', labelsize=9, colors='#555')
                                ax.tick_params(axis='y', labelsize=9, colors='#555')
                                ax.set_title(f"{field_key.replace('_', ' ').title()} Distribution", fontsize=12)
                                
                                if num_viz_type == "Histogram":
                                    ax.set_xlabel(field_key.replace('_', ' ').title(), fontsize=10, color='#666')
                                    ax.set_ylabel('Count', fontsize=10, color='#666')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Calculate and show statistics with modern cards
                                stats = df_viz[field_key].describe()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:10px;">
                                        <div style="font-size:0.8rem; color:#666;">Mean</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{stats['mean']:.2f}</div>
                                    </div>
                                    
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px;">
                                        <div style="font-size:0.8rem; color:#666;">Median</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{stats['50%']:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:10px;">
                                        <div style="font-size:0.8rem; color:#666;">Min</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{stats['min']:.2f}</div>
                                    </div>
                                    
                                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px;">
                                        <div style="font-size:0.8rem; color:#666;">Max</div>
                                        <div style="font-size:1.2rem; font-weight:bold; color:#1a73e8;">{stats['max']:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error visualizing {field_key}: {e}")
    
    # === RELATIONSHIPS TAB ===
    with viz_tabs[2]:
        st.markdown("### Relationships Between Fields")
        
        # Check if we have at least 2 numeric fields for correlation
        numeric_columns = []
        for field in available_fields:
            if field['type'] in ['int', 'float'] and field['key'] in df_viz.columns:
                numeric_columns.append(field['key'])
            elif field['key'] in df_viz.columns and is_numeric_column(field['key']):
                # Try to convert to numeric
                df_viz[field['key']] = pd.to_numeric(df_viz[field['key']], errors='coerce')
                numeric_columns.append(field['key'])
        
        if len(numeric_columns) >= 2:
            st.subheader("Correlation Heatmap")
            try:
                # Calculate correlation matrix
                corr_matrix = df_viz[numeric_columns].corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
                
                # Generate a custom diverging colormap
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                
                # Draw the heatmap with the mask and correct aspect ratio
                sns.heatmap(
                    corr_matrix, 
                    mask=mask,
                    cmap=cmap, 
                    vmax=1.0, 
                    vmin=-1.0,
                    center=0,
                    square=True, 
                    linewidths=.5, 
                    cbar_kws={"shrink": .8},
                    annot=True,
                    fmt='.2f',
                    annot_kws={"size": 9}
                )
                
                plt.title('Correlation Between Numeric Fields', fontsize=14, pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Allow exploration of relationships between two fields
                st.subheader("Explore Relationships")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_field = st.selectbox("X-axis field:", numeric_columns, key="x_field")
                with col2:
                    y_field = st.selectbox("Y-axis field:", [f for f in numeric_columns if f != x_field], key="y_field")
                
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Draw scatter plot with regression line
                scatter = sns.regplot(
                    x=x_field,
                    y=y_field,
                    data=df_viz,
                    scatter_kws={"alpha": 0.6, "s": 80, "color": colors[0]},
                    line_kws={"color": colors[3], "lw": 2, "alpha": 0.7},
                    ax=ax
                )
                
                # Calculate correlation coefficient
                corr = df_viz[[x_field, y_field]].corr().iloc[0, 1]
                
                # Clean up the chart
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#ddd')
                ax.spines['bottom'].set_color('#ddd')
                ax.tick_params(colors='#555')
                ax.set_title(f"Relationship between {x_field.replace('_', ' ').title()} and {y_field.replace('_', ' ').title()}\nCorrelation: {corr:.2f}", fontsize=12, pad=20)
                ax.set_xlabel(x_field.replace('_', ' ').title(), fontsize=11, color='#444')
                ax.set_ylabel(y_field.replace('_', ' ').title(), fontsize=11, color='#444')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error creating correlation analysis: {e}")
        else:
            st.info("Need at least 2 numeric fields for correlation analysis")
        
        # Categorical relationship exploration
        categorical_fields = [field['key'] for field in available_fields if field['type'] == 'enum' or field['type'] == 'str']
        
        if len(categorical_fields) >= 2 and len(categorical_fields) <= 10:
            st.subheader("Categorical Field Relationships")
            
            col1, col2 = st.columns(2)
            with col1:
                cat_x = st.selectbox("First category:", categorical_fields, key="cat_x")
            with col2:
                cat_y = st.selectbox("Second category:", [f for f in categorical_fields if f != cat_x], key="cat_y")
            
            try:
                # Create cross-tabulation
                cross_tab = pd.crosstab(df_viz[cat_x], df_viz[cat_y])
                
                # Normalize to show percentages
                cross_tab_norm = pd.crosstab(df_viz[cat_x], df_viz[cat_y], normalize='index') * 100
                
                # Create heatmap visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.heatmap(
                    cross_tab_norm,
                    annot=cross_tab.values,  # Show raw counts
                    fmt='d',  # Display as integers
                    cmap='YlGnBu',
                    ax=ax,
                    linewidths=0.5,
                    cbar_kws={'label': 'Percentage (%)'}
                )
                
                plt.title(f'Relationship Between {cat_x.replace("_", " ").title()} and {cat_y.replace("_", " ").title()}', fontsize=14, pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error analyzing categorical relationships: {e}")
        
    # === TEXT ANALYSIS TAB ===
    with viz_tabs[3]:
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
        text_fields = [field for field in available_fields 
                     if field['type'] == 'str' and field['key'] in df.columns 
                     and field['key'] not in ['themes']]
        
        for text_field in text_fields:
            field_key = text_field['key']
            
            st.markdown(f"#### {field_key.replace('_', ' ').title()} Word Cloud")
            
            try:
                # Combine all text from the field
                all_text = " ".join([str(text) for text in df[field_key] if pd.notna(text)])
                
                if all_text.strip():
                    # Generate word cloud (if wordcloud library is available)
                    try:
                        from wordcloud import WordCloud
                        
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
    
    # === ADVANCED TAB ===
    with viz_tabs[4]:
        st.markdown("### Advanced Analysis")
        
        # Summary statistics for all fields
        st.subheader("Summary Statistics")
        
        # Get statistics for numeric fields
        numeric_stats = df_viz.describe().T.reset_index()
        numeric_stats.columns = ['field'] + list(numeric_stats.columns[1:])
        
        if not numeric_stats.empty:
            st.dataframe(numeric_stats, use_container_width=True)
            
            # Option to download stats
            csv_stats = numeric_stats.to_csv(index=False)
            b64_stats = base64.b64encode(csv_stats.encode()).decode()
            st.download_button(
                label="Download Statistics CSV",
                data=csv_stats,
                file_name="statistics_summary.csv",
                mime="text/csv",
            )
        
        # Missing data visualization
        st.subheader("Missing Data Analysis")
        
        # Calculate missing values
        missing_data = df_viz.isnull().sum().reset_index()
        missing_data.columns = ['Field', 'Missing Values']
        missing_data['Percentage'] = (missing_data['Missing Values'] / len(df_viz) * 100).round(1)
        missing_data = missing_data.sort_values('Missing Values', ascending=False)
        
        if missing_data['Missing Values'].sum() > 0:
            # Create bar chart for missing values
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = sns.barplot(
                data=missing_data,
                y='Field',
                x='Percentage',
                ax=ax,
                color=colors[0],
                alpha=0.7
            )
            
            # Add percentage labels
            for i, bar in enumerate(bars.patches):
                bars.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height()/2,
                    f'{missing_data.iloc[i]["Missing Values"]} ({missing_data.iloc[i]["Percentage"]}%)',
                    ha='left', va='center',
                    fontsize=9, color='#555'
                )
            
            # Clean up the chart
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#ddd')
            ax.spines['bottom'].set_color('#ddd')
            ax.set_title("Missing Values by Field", fontsize=14)
            ax.set_xlabel('Percentage (%)', fontsize=10, color='#666')
            ax.set_ylabel('')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No missing values in the dataset")

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