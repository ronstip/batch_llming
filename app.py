import traceback
import streamlit as st
import pandas as pd
import base64
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, get_type_hints, Dict, Any, Type, List, Optional
from pydantic import BaseModel, Field, create_model
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
import time
import os
from PIL import Image
import io
import requests
from io import BytesIO
# from langchain_community.chat_models import ChatOpenAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback to community package if needed
    from langchain_community.chat_models import ChatOpenAI
    st.warning("Using deprecated ChatOpenAI from langchain_community. Consider installing langchain_openai package.")

from src.components.data_input import render_data_input
from src.components.configuration import render_sidebar_config, render_analysis_config
from src.models.pydantic_models import create_dynamic_model
from src.utils.processing import process_data_batch, load_image_from_path, process_post
from src.utils.styles import apply_custom_css
from src.visualizations.display import display_results, create_visualizations, display_errors

# Set page config
st.set_page_config(page_title="Social Media Post Analyzer", layout="wide")

# Apply custom CSS
apply_custom_css()

def load_and_resize_image(image_path, max_width=120):
    """Load an image from path and resize it to create a thumbnail with caching for performance"""
    # Use Streamlit's caching to avoid reloading the same image multiple times
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_image_loader(path, width):
        try:
            # Handle both URLs and local file paths
            if path.startswith(('http://', 'https://')):
                try:
                    response = requests.get(path, timeout=5)  # Add timeout for better performance
                    response.raise_for_status()  # Raise exception for bad responses
                    image = Image.open(BytesIO(response.content))
                except (requests.RequestException, IOError) as e:
                    # Avoid showing warning repeatedly - just print to console
                    print(f"Failed to load image from URL: {path}")
                    return None
            else:
                # For local file paths
                if os.path.exists(path):
                    try:
                        image = Image.open(path)
                    except IOError:
                        print(f"Failed to open image file: {path}")
                        return None
                else:
                    return None
            
            # Validate image dimensions to avoid division by zero
            if image.width <= 0 or image.height <= 0:
                return None
                
            # Calculate new height while maintaining aspect ratio
            width_percent = width / float(image.width)
            new_height = int(float(image.height) * width_percent)
            
            # Ensure reasonable dimensions (prevent excessively tall images)
            max_height = width * 1.5  # Max height is 1.5x the width (reduced from 2x)
            if new_height > max_height:
                new_height = max_height
            
            # Resize the image with high quality
            resized_image = image.resize((width, new_height), Image.LANCZOS)
            return resized_image
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            return None
    
    return _cached_image_loader(image_path, max_width)

# Function to display results with metrics and tables
def display_results(results_df, fields):
    st.header("Analysis Results")
    
    # Create metrics row with modern styling
    st.markdown('<div style="padding: 10px 0 20px 0;">', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    
    # Count total posts
    with metric_cols[0]:
        st.metric("Total Posts", len(results_df))
    
    # Get all field keys
    field_keys = [field['key'] for field in fields]
    
    # Find most common sentiment if present
    if 'sentiment' in field_keys and 'sentiment' in results_df.columns:
        with metric_cols[1]:
            most_common = results_df['sentiment'].value_counts().idxmax()
            st.metric("Most Common Sentiment", most_common)
            
            # Format sentiment as tag for later use
            def format_sentiment(sentiment):
                if sentiment.lower() == 'positive':
                    return f'<span class="tag tag-positive">{sentiment}</span>'
                elif sentiment.lower() == 'negative':
                    return f'<span class="tag tag-negative">{sentiment}</span>'
                else:
                    return f'<span class="tag tag-neutral">{sentiment}</span>'
            
            # Add formatted sentiment to dataframe for display
            results_df['formatted_sentiment'] = results_df['sentiment'].apply(format_sentiment)
    
    # Find average engagement if present
    if 'engagement_potential' in field_keys and 'engagement_potential' in results_df.columns:
        with metric_cols[2]:
            engagement_map = {'high': 3, 'medium': 2, 'low': 1}
            try:
                avg_engagement = results_df['engagement_potential'].map(engagement_map).mean()
                if avg_engagement > 2:
                    engagement_level = "High"
                elif avg_engagement > 1:
                    engagement_level = "Medium"
                else:
                    engagement_level = "Low"
                st.metric("Average Engagement", engagement_level)
            except:
                st.metric("Average Engagement", "Not Available")
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Identify image path columns in the dataframe
    image_path_columns = [col for col in results_df.columns if col.startswith('original_') and 
                          any(path_col in col for path_col in st.session_state.get('image_path_columns', []))]
    
    # Debug info in an expander
    with st.expander("🔍 Debug Information", expanded=False):
        st.write("#### Current Fields:")
        st.json(fields)
        st.write("#### Available Columns:")
        st.write(list(results_df.columns))
        if image_path_columns:
            st.write("#### Image Path Columns:")
            st.write(image_path_columns)
    
    # Create tabs for different views - now adding Image Gallery tab
    tab1, tab2, tab3 = st.tabs(["Table View", "Card View", "Image Gallery"])
    
    with tab1:
        # Filter controls
        if len(results_df) > 5:
            with st.expander("🔍 Filter Results", expanded=False):
                filter_cols = st.columns(3)
                
                # Add filters based on field types
                active_filters = {}
                
                for i, field in enumerate(fields):
                    field_key = field['key']
                    if field_key in results_df.columns:
                        with filter_cols[i % 3]:
                            if field['type'] == 'enum':
                                options = ['All'] + list(results_df[field_key].unique())
                                selected = st.selectbox(f"Filter by {field_key}", options)
                                if selected != 'All':
                                    active_filters[field_key] = selected
                
                # Apply filters if any
                filtered_df = results_df.copy()
                for field, value in active_filters.items():
                    filtered_df = filtered_df[filtered_df[field] == value]
                
                if active_filters:
                    st.success(f"Showing {len(filtered_df)} of {len(results_df)} results")
                else:
                    filtered_df = results_df
        else:
            filtered_df = results_df
        
        # Display the main dataframe
        st.dataframe(filtered_df, use_container_width=True)
        
        # Add image thumbnails directly below the table if images exist
        # if image_path_columns:
        #     st.markdown("### Post Images")
            
        #     # Pagination for image display
        #     items_per_page = 5
        #     total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page
            
        #     col1, col2, col3 = st.columns([1, 3, 1])
        #     with col2:
        #         page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1) - 1
            
        #     start_idx = page * items_per_page
        #     end_idx = min(start_idx + items_per_page, len(filtered_df))
            
        #     for idx in range(start_idx, end_idx):
        #         row = filtered_df.iloc[idx]
                
        #         # Create a row with information and images
        #         col1, col2 = st.columns([1, 3])
                
        #         with col1:
        #             st.markdown(f"**Post #{idx+1}**")
        #             title = row.get('original_title', '')
        #             if title:
        #                 st.markdown(f"**Title:** {title}")
                    
        #             # Show key insights inline
        #             for field in fields[:3]:  # Show first 3 fields only
        #                 field_key = field['key']
        #                 if field_key in row and field_key not in ['image_description']:
        #                     if field_key == 'sentiment' and 'formatted_sentiment' in row:
        #                         st.markdown(f"**Sentiment:** {row['formatted_sentiment']}", unsafe_allow_html=True)
        #                     else:
        #                         st.markdown(f"**{field_key.title()}:** {row[field_key]}")
                
        #         with col2:
        #             # Create image gallery for this row
        #             valid_image_cols = []
        #             for col in image_path_columns:
        #                 img_path = row[col]
        #                 if pd.notna(img_path) and str(img_path).strip():
        #                     valid_image_cols.append((col, img_path))
                    
        #             if valid_image_cols:
        #                 image_display_cols = st.columns(min(3, len(valid_image_cols)))
                        
        #                 for i, (col, img_path) in enumerate(valid_image_cols):
        #                     col_name = col.replace('original_', '')
                            
        #                     with image_display_cols[i % len(image_display_cols)]:
        #                         # Load and display the image
        #                         img = load_and_resize_image(img_path, max_width=160)
        #                         if img:
        #                             st.image(img, caption=col_name)
        #                         else:
        #                             st.markdown(f"[{col_name}]({img_path})")
        #             else:
        #                 st.info("No images available for this post")
                
        #         st.markdown("---")
        
        # Download button for results
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv",
        )
    
    with tab2:
        # Card View - Improved with better image integration
        if len(filtered_df) > 20:
            st.info(f"Showing first 20 of {len(filtered_df)} results")
            display_df = filtered_df.head(20)
        else:
            display_df = filtered_df
        
        for i in range(0, len(display_df), 3):
            card_cols = st.columns(3)
            for j in range(3):
                if i + j < len(display_df):
                    row = display_df.iloc[i + j]
                    with card_cols[j]:
                        # Add post number for reference
                        st.markdown(f"### Post #{i+j+1}")
                        
                        # Title with better styling - only show if exists
                        title = row.get('original_title', row.get('original_query', ''))
                        if title:
                            st.markdown(f"<strong style='color:#1967d2;'>{title}</strong>", unsafe_allow_html=True)
                        
                        # Display images first if they exist
                        image_cols = [col for col in row.index if col.startswith('original_') and 
                                    any(path_col in col for path_col in st.session_state.get('image_path_columns', []))]
                        
                        if image_cols:
                            # Show first image prominently
                            first_img_col = image_cols[0]
                            img_path = row.get(first_img_col)
                            
                            if pd.notna(img_path) and str(img_path).strip():
                                img = load_and_resize_image(img_path, max_width=180)
                                if img:
                                    st.image(img, use_container_width=False)
                            
                            # Show additional images as small thumbnails if more than one
                            if len(image_cols) > 1:
                                st.markdown("**More Images:**")
                                thumb_cols = st.columns(min(3, len(image_cols)-1))
                                
                                for idx, img_col in enumerate(image_cols[1:]):
                                    if idx < len(thumb_cols):
                                        img_path = row.get(img_col)
                                        if pd.notna(img_path) and str(img_path).strip():
                                            with thumb_cols[idx]:
                                                img = load_and_resize_image(img_path, max_width=120)
                                                if img:
                                                    st.image(img, caption=img_col.replace('original_', ''))
                        
                        # Display analysis results with styled tags
                        st.markdown("#### Analysis")
                        
                        # Format sentiment with colored tag if present
                        if 'sentiment' in row and 'formatted_sentiment' in row:
                            st.markdown(f"**Sentiment:** {row['formatted_sentiment']}", unsafe_allow_html=True)
                        
                        # Format other fields
                        for field in fields:
                            field_key = field['key']
                            if field_key in row and field_key != 'sentiment':
                                # Format themes as individual tags
                                if field_key == 'themes' and isinstance(row[field_key], str):
                                    themes = [t.strip() for t in row[field_key].split(',')]
                                    themes_html = ""
                                    for theme in themes:
                                        themes_html += f'<span class="tag">{theme}</span> '
                                    st.markdown(f"**Themes:** {themes_html}", unsafe_allow_html=True)
                                # Format engagement with colored indicators
                                elif field_key == 'engagement_potential':
                                    engagement = row[field_key]
                                    if engagement.lower() == 'high':
                                        st.markdown(f"**Engagement:** <span class='tag tag-positive'>{engagement}</span>", 
                                                   unsafe_allow_html=True)
                                    elif engagement.lower() == 'low':
                                        st.markdown(f"**Engagement:** <span class='tag tag-negative'>{engagement}</span>", 
                                                   unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"**Engagement:** <span class='tag tag-neutral'>{engagement}</span>", 
                                                   unsafe_allow_html=True)
                                # Other fields with normal formatting
                                else:
                                    field_value = row[field_key] if pd.notna(row[field_key]) else "Not available"
                                    
                                    # Apply tag styling to relevant field types
                                    if isinstance(field_value, str) and (',' in field_value or field_value.lower() in ['high', 'medium', 'low', 'positive', 'negative', 'neutral']):
                                        # If comma-separated values, create multiple tags
                                        if ',' in field_value:
                                            values = [v.strip() for v in field_value.split(',')]
                                            values_html = ""
                                            for val in values:
                                                # Apply color based on value sentiment
                                                if val.lower() in ['high', 'positive', 'excellent', 'good']:
                                                    values_html += f'<span class="tag tag-positive">{val}</span> '
                                                elif val.lower() in ['low', 'negative', 'poor', 'bad']:
                                                    values_html += f'<span class="tag tag-negative">{val}</span> '
                                                else:
                                                    values_html += f'<span class="tag">{val}</span> '
                                            st.markdown(f"**{field_key.replace('_', ' ').title()}:** {values_html}", unsafe_allow_html=True)
                                        # Single value with sentiment-based coloring
                                        else:
                                            tag_class = ""
                                            if field_value.lower() in ['high', 'positive', 'excellent', 'good']:
                                                tag_class = "tag-positive"
                                            elif field_value.lower() in ['low', 'negative', 'poor', 'bad']:
                                                tag_class = "tag-negative"
                                            
                                            st.markdown(f"**{field_key.replace('_', ' ').title()}:** <span class='tag {tag_class}'>{field_value}</span>", 
                                                      unsafe_allow_html=True)
                                    else:
                                        # Default display for other fields
                                        st.markdown(f"**{field_key.replace('_', ' ').title()}:** {field_value}")
    
    # Image Gallery Tab
    with tab3:
        if not st.session_state.get('image_path_columns', []):  #image_path_columns:
            st.info("No images available in the current dataset")
        else:
            st.markdown("### Image Gallery")
            
            # Group by image type
            img_cols_per_page = 4
            img_rows_per_page = 4
            items_per_page = img_cols_per_page * img_rows_per_page
            
            # Get all valid image paths from the dataset
            all_images = []
            for _, row in filtered_df.iterrows():
                for col in image_path_columns:
                    img_path = row[col]
                    if pd.notna(img_path) and str(img_path).strip():
                        all_images.append({
                            'path': img_path,
                            'column': col.replace('original_', ''),
                            'row_idx': _
                        })
            
            if not all_images:
                st.info("No valid images found in the dataset")
            else:
                # Pagination controls
                total_pages = (len(all_images) + items_per_page - 1) // items_per_page
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    page = st.slider("Gallery Page", 1, max(2, total_pages), 1)
                
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(all_images))
                
                # Display images in a matrix layout using columns
                rows = []
                current_row = []
                
                for i in range(start_idx, end_idx):
                    current_row.append(all_images[i])
                    if len(current_row) == img_cols_per_page:
                        rows.append(current_row)
                        current_row = []
                
                # Add any remaining images
                if current_row:
                    rows.append(current_row)
                
                # Display each row of images
                for row_images in rows:
                    cols = st.columns(img_cols_per_page)
                    for i, img_info in enumerate(row_images):
                        with cols[i]:
                            img_path = img_info['path']
                            img = load_and_resize_image(img_path, max_width=180)
                            if img:
                                st.image(img, caption=f"{img_path}")
                            else:
                                st.markdown(f"[{img_info['column']}]({img_path})")

# Helper function to get base64 encoded image for HTML display
def get_image_base64(image_path, max_width=120):
    """Convert image to base64 for HTML embedding"""
    try:
        img = load_and_resize_image(image_path, max_width=max_width)
        if img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", optimize=True, quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
        return ""
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def display_errors(error_rows):
    """Display errors from the results dataframe"""
    if not error_rows.empty:
        st.subheader("❌ Processing Errors")
        st.error(f"{len(error_rows)} items failed to process. Showing details below.")
        
        for i, (_, row) in enumerate(error_rows.iterrows()):
            with st.expander(f"Error {i+1}: {row.get('error_type', 'Unknown error')}"):
                st.write(f"**Error message:** {row.get('error', 'No error message available')}")
                
                if 'traceback' in row:
                    st.code(row['traceback'], language="python")

# Main app
def main():
    # Initialize session state for results
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    if 'raw_results' not in st.session_state:
        st.session_state.raw_results = None
    
    if 'schema_fields' not in st.session_state:
        st.session_state.schema_fields = []
        
    if 'need_rerun' not in st.session_state:
        st.session_state.need_rerun = False
        
    if 'image_path_columns' not in st.session_state:
        st.session_state.image_path_columns = []
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Analysis", "Results"])
    
    # Sidebar config
    llm, api_key, image_detail, n_workers = render_sidebar_config()
    
    # Main content
    with tab1:
        # Data input
        uploaded_file, df, image_data, image_path_columns, input_mode, query_text = render_data_input()
        
        # Store image_path_columns in session state
        if image_path_columns:
            st.session_state.image_path_columns = image_path_columns
        
        # Prompt and schema configuration
        prompt_text, schema_fields = render_analysis_config()
        
        # Create a dynamic output model
        output_model = None
        if schema_fields:
            try:
                output_model = create_dynamic_model("SocialMediaAnalysis", schema_fields)
            except Exception as e:
                st.error(f"Error creating output model: {str(e)}")
        
        # Process button
        process_col1, process_col2 = st.columns([3, 1])
        
        with process_col1:
            process_placeholder = st.empty()
            process_button = process_placeholder.button(
                "🚀 Process Data", 
                type="primary",
                use_container_width=True,
                disabled=(llm is None or df is None or df.empty or output_model is None)
            )
        
        with process_col2:
            # Clear results button
            if st.button("🧹 Clear Results", use_container_width=True):
                st.session_state.results_df = None
                st.session_state.raw_results = None
                st.rerun()
    
        # Check if conditions are met for processing
        if not llm and process_button:
            st.error("Please enter a valid OpenAI API key in the sidebar first.")
        
        if not df is not None and process_button:
            st.error("Please upload a CSV file first.")
        
        if not output_model and process_button:
            st.error("Please configure the output schema first.")
    
        # Process data when button is clicked
        if process_button and llm and df is not None and not df.empty and output_model:
            # Process the data
            with st.spinner("Processing data..."):
                try:
                    # Use prompt template with dynamic fields
                    prompt_template = prompt_text
                    
                    # Process the data in batches
                    results = process_data_batch(
                        df, 
                        prompt_template, 
                        llm, 
                        output_model, 
                        n_workers=n_workers,  # Use the user-configured number of workers
                        image_data=image_data, 
                        image_path_columns=image_path_columns,
                        image_detail=image_detail
                    )
                    
                    # Check for errors
                    error_count = sum(1 for result in results if 'error' in result)
                    
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} out of {len(results)} items failed to process. See results for details.")
                    
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Store results in session state
                    st.session_state.results_df = results_df
                    st.session_state.raw_results = results
                    
                    st.success(f"✅ Successfully processed {len(results) - error_count} out of {len(results)} items.")
                    
                    # Display a visual summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'sentiment' in results_df.columns:
                            positive_count = (results_df['sentiment'] == 'positive').sum()
                            positive_percent = (positive_count / len(results_df)) * 100
                            st.metric(label="Positive Sentiment", value=f"{positive_percent:.1f}%")
                    
                    with col2:
                        if 'engagement_potential' in results_df.columns:
                            high_count = (results_df['engagement_potential'] == 'high').sum()
                            high_percent = (high_count / len(results_df)) * 100
                            st.metric(label="High Engagement", value=f"{high_percent:.1f}%")
                    
                    with col3:
                        st.metric(label="Processing Time", value=f"{results_df['_debug_info'].apply(lambda x: x.get('start_time', 0)).mean()/1000000000:.2f}s/post")
                        # total processing time
                        st.metric(label="Total Processing Time", value=f"{results_df['_debug_info'].apply(lambda x: x.get('end_time', 0) - x.get('start_time', 0)).sum()/-1000000000:.2f}s")
                    
                    # Auto-switch to Results tab
                    st.markdown('<script>var tab_btn = parent.window.document.querySelectorAll(".stTabs button")[1]; tab_btn.click();</script>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    # Results tab content
    with tab2:
        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            # Store any image path columns for display
            if st.session_state.image_path_columns:
                # Get the image columns with 'original_' prefix added
                image_path_columns = [f"original_{col}" for col in st.session_state.image_path_columns 
                                     if f"original_{col}" in st.session_state.results_df.columns]
                
                # Also include the original column names if they exist
                image_path_columns.extend([col for col in st.session_state.image_path_columns 
                                          if col in st.session_state.results_df.columns])
                
                # Make sure these are passed to the display function
                st.session_state.display_image_columns = image_path_columns
            
            # Display results
            display_results(st.session_state.results_df, schema_fields)
            
            # Create visualizations
            create_visualizations(st.session_state.results_df, schema_fields)
            
            # Display errors if any
            if 'error' in st.session_state.results_df.columns:
                error_rows = st.session_state.results_df[st.session_state.results_df['error'].notna()]
                if not error_rows.empty:
                    display_errors(error_rows)
        else:
            st.info("No results to display. Process data first in the Analysis tab.")
    
    # Check if we need to rerun the app (used when schema fields are updated)
    if st.session_state.get('need_rerun', False):
        st.session_state.need_rerun = False
        st.rerun()

if __name__ == "__main__":
    main()