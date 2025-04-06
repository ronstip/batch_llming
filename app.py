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
from src.utils.processing import process_data_batch, load_image_from_path
from src.utils.styles import apply_custom_css
from src.visualizations.display import display_results, create_visualizations, display_errors

# Set page config
st.set_page_config(page_title="Social Media Post Analyzer", layout="wide")

# Apply custom CSS
apply_custom_css()

# Create dynamic pydantic model
def create_dynamic_model(name: str, fields: list[dict[str, Any]]) -> Type[BaseModel]:
    """
    Create a dynamic Pydantic model using langchain's pydantic_v1.

    Args:
        name: Name of the model.
        fields: A list of dictionaries, each with 'key', 'type', and 'description'.
                If type is 'enum', provide 'options' as a list of values.

    Returns:
        A dynamically created Pydantic model class.
    """
    annotations: Dict[str, tuple] = {}

    for field in fields:
        key = field['key']
        field_type = field['type']
        description = field['description']

        if field_type == "str":
            annotations[key] = (str, Field(..., description=description))
        elif field_type == "int":
            annotations[key] = (int, Field(..., description=description))
        elif field_type == "float":
            annotations[key] = (float, Field(..., description=description))
        elif field_type == 'enum':
            options = field.get('options')
            if not options:
                raise ValueError(f"Enum field '{key}' must have 'options'.")
            annotations[key] = (Literal[tuple(options)], Field(..., description=description))
        else:
            raise TypeError(f"Unsupported type: {field_type}")

    return create_model(name, **annotations)

def process_post(row, prompt_template, model, output_model, image_data=None, image_path_columns=None):
    try:
        # Create a log dictionary to capture each step
        debug_info = {}
        
        # Extract values from the row as strings
        post_values = {k: str(v) if v is not None else "" for k, v in row.to_dict().items()}
        debug_info["post_values"] = post_values
        
        # Format the prompt with the post values
        try:
            formatted_prompt = prompt_template.format(**post_values)
            debug_info["formatted_prompt"] = formatted_prompt
        except Exception as format_err:
            debug_info["format_error"] = str(format_err)
            raise format_err
        
        # Determine image to use - prioritize image_path_columns if provided
        row_image_data = None
        
        # Check if we have image path columns and try to load the first valid image
        if image_path_columns:
            debug_info["image_path_columns"] = image_path_columns
            for col in image_path_columns:
                if col in post_values and post_values[col].strip():
                    image_path = post_values[col]
                    debug_info["image_path_attempted"] = image_path
                    
                    # Try to load the image from the path
                    row_image_data = load_image_from_path(image_path)
                    if row_image_data:
                        debug_info["image_loaded_from_path"] = image_path
                        break
        
        # If no image was loaded from paths, use the global image_data if available
        if not row_image_data and image_data:
            row_image_data = image_data
            debug_info["using_global_image"] = True
        
        # Create a chat prompt template based on whether we have an image
        if row_image_data:
            # Create a multimodal message with image
            content = [
                {"type": "text", "text": formatted_prompt}
            ]
            
            # Add image as content part
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{row_image_data['type']};base64,{row_image_data['base64']}"
                }
            })
            
            # Create a multimodal message
            message = HumanMessage(content=content)
            
            # Create chat prompt with multimodal content
            prompt = ChatPromptTemplate.from_messages([message])
            debug_info["prompt_type"] = "multimodal"
        else:
            # Standard text-only prompt
            prompt = ChatPromptTemplate.from_messages([
                ("human", formatted_prompt)
            ])
            debug_info["prompt_type"] = "text-only"
            
        debug_info["prompt_created"] = True
        
        # Check if model is None
        if model is None:
            raise ValueError("Model is None - API key may not be set correctly")
        debug_info["model_available"] = True
        
        # Log model type
        debug_info["model_type"] = str(type(model))
        
        # Verify output_model
        debug_info["output_model_type"] = str(type(output_model))
        debug_info["output_model_schema"] = output_model.schema()
        
        # Call the LLM with structured output
        try:
            chain = prompt | model.with_structured_output(output_model)
            debug_info["chain_created"] = True
            
            result = chain.invoke({})
            debug_info["llm_result_received"] = True
            
            # Convert pydantic model to dict
            parsed_result = result.dict()
            debug_info["parsed_result"] = True
        except Exception as llm_err:
            debug_info["llm_error"] = str(llm_err)
            raise llm_err
        
        # Add the original post data to the result
        for key, value in post_values.items():
            parsed_result[f"original_{key}"] = value
        
        # Add debug info to result
        parsed_result["_debug_info"] = debug_info
                
        return parsed_result
        
    except Exception as e:
        error_msg = f"Error processing post: {str(e)}"
        st.error(error_msg)
        
        # Return error with details
        return {
            "error": str(e),
            "error_type": str(type(e)),
            "traceback": traceback.format_exc()
        }

# Function to create visualizations based on field types
def create_visualizations(df, fields):
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
                    st.write(f"Mean: {df[field_key].mean():.2f}")
                    st.write(f"Median: {df[field_key].median():.2f}")
                    st.write(f"Min: {df[field_key].min():.2f}")
                    st.write(f"Max: {df[field_key].max():.2f}")
                
                col_index += 1
        except Exception as e:
            st.error(f"Error visualizing {field_key}: {str(e)}")

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

# Main app
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # App header with logo and description
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="font-size: 2.5rem; margin-right: 0.5rem;">🔄</div>
        <div>
            <h1 style="margin: 0; padding: 0;">Batch LLM Processing</h1>
            <p style="margin: 0; color: #666;">Process multiple inputs using LLMs with structured outputs & image analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'llm_initialized' not in st.session_state:
        st.session_state.llm_initialized = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()  # Initialize as empty DataFrame instead of None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    # Sidebars and configuration
    llm, api_key, image_detail = render_sidebar_config()
    
    # Make sure LLM is initialized with better error messaging
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue")
        # Show helpful instructions
        st.info("""
        ### Getting Started
        1. Enter your OpenAI API key in the sidebar
        2. Upload a CSV file or write a query
        3. Configure your prompt and output schema
        4. Click 'Analyze Data' to process
        """)
        st.stop()
    
    st.session_state.llm_initialized = llm is not None
    
    if not st.session_state.llm_initialized:
        st.error("❌ Language model initialization failed. Please check your API key and selected model.")
        st.stop()
    
    # Create tabs for different sections of the app
    input_tab, config_tab, analyze_tab, results_tab = st.tabs(["Data Input", "Analysis Configuration", "Analyze Data", "Results & Insights"])
    
    with input_tab:
        # Load data input component
        uploaded_file, df, image_data, image_path_columns, input_mode, query_text = render_data_input()
        st.session_state.image_path_columns = image_path_columns
        
        # Show image data status
        if image_data:
            st.success("✅ Global image loaded successfully and will be used for all rows")
        
        if image_path_columns:
            st.success(f"✅ Image path columns selected: {', '.join(image_path_columns)}")
    
    with config_tab:
        # Load configuration components
        prompt_template, fields = render_analysis_config()
        
        # Generate schema model from fields
        schema_model = create_dynamic_model("OutputSchema", fields)
        
        # Preview the schema in a cleaner format
        with st.expander("📋 Preview Generated Schema", expanded=False):
            schema_json = schema_model.schema()
            st.json(schema_json)
            
    with analyze_tab:
        st.header("Analysis Controls")
        
        # Process button and worker selection in a clean layout
        col1, col2 = st.columns([3, 1])
        with col2:
            # Number of workers selection with better UI
            n_workers = st.number_input(
                "Parallel Workers", 
                min_value=1, 
                max_value=40, 
                value=10,
                help="Higher values process faster but use more resources"
            )
            
        with col1:
            # Improved analyze button with status indication
            if st.session_state.processing:
                analyze_button = st.button(
                    "⏳ Processing...", 
                    type="primary", 
                    disabled=True,
                    use_container_width=True
                )
            else:
                analyze_button = st.button(
                    "🚀 Analyze Data", 
                    type="primary", 
                    use_container_width=True
                )
        
        # Display errors from processing if they exist
        if hasattr(st.session_state, 'has_errors') and st.session_state.has_errors:
            st.markdown("---")
            st.subheader("⚠️ Processing Errors")
            
            if hasattr(st.session_state, 'error_results') and st.session_state.error_results:
                st.warning(f"{len(st.session_state.error_results)} items failed to process properly.")
                
                with st.expander("View Error Details", expanded=False):
                    for i, error_result in enumerate(st.session_state.error_results):
                        st.markdown(f"**Error {i+1}:** {error_result.get('error', 'Unknown error')}")
                        if 'traceback' in error_result:
                            st.markdown("**Traceback:**")
                            st.code(error_result['traceback'], language='python')
                        st.markdown("---")
    
    # Handle data processing
    query_mode = input_mode == "Write Query" and query_text
    file_mode = uploaded_file is not None
    
    # Process when button is clicked
    if analyze_button and (file_mode or query_mode):
        # Set processing state for UI feedback
        st.session_state.processing = True
        st.session_state.error_message = None
        
        # Rerun to update UI immediately
        st.rerun()
    
    # Continue processing after rerun if in processing state
    if st.session_state.processing and (file_mode or query_mode):
        try:
            if file_mode and df is not None and df.shape[0] > 0:
                with st.spinner("🔄 Processing data batch..."):
                    # Create progress components
                    progress_col1, progress_col2 = st.columns([3, 1])
                    with progress_col1:
                        progress_bar = st.progress(0)
                    with progress_col2:
                        progress_text = st.empty()
                    
                    # Process the data with progress updates
                    start_time = time.time()
                    
                    def progress_callback(current, total):
                        """Callback to update progress during processing"""
                        progress = current / total
                        progress_bar.progress(progress)
                        progress_text.markdown(f"**{current}/{total}** items")
                    
                    # Process data with progress tracking
                    results = process_data_batch(
                        df, 
                        prompt_template, 
                        llm, 
                        schema_model, 
                        n_workers,
                        image_data,
                        image_path_columns,
                        image_detail
                    )
                    
                    # Check if any results were returned
                    if results:
                        # Separate successful results from errors
                        success_results = []
                        error_results = []
                        
                        for result in results:
                            if 'error' in result:
                                error_results.append(result)
                            else:
                                success_results.append(result)
                        
                        # Convert successful results to DataFrame
                        if success_results:
                            results_df = pd.DataFrame(success_results)
                            
                            # Store in session state
                            st.session_state.results_df = results_df
                            st.session_state.has_errors = len(error_results) > 0
                            st.session_state.error_results = error_results
                        else:
                            # All results had errors
                            st.session_state.results_df = pd.DataFrame()
                            st.session_state.has_errors = True
                            st.session_state.error_results = error_results
                            st.session_state.error_message = "All processing attempts resulted in errors. Check the error details."
                        
                        # Show processing time
                        elapsed_time = time.time() - start_time
                        time_per_item = elapsed_time / len(results)
                        
                        st.success(f"✅ Processed {len(results)} items in {elapsed_time:.2f} seconds ({time_per_item:.2f}s per item)")
                        
                        if success_results:
                            st.success(f"Successfully processed {len(success_results)} of {len(results)} items")
                            
                            # Switch to results tab automatically only if we have successful results
                            results_tab.active = True
                        
                        if error_results:
                            st.warning(f"⚠️ {len(error_results)} of {len(results)} items had errors. See the 'Analyze Data' tab for details.")
                    else:
                        st.session_state.error_message = "No results were generated. Check your data and try again."
                        st.error(st.session_state.error_message)
            
            elif query_mode:
                with st.spinner("🔄 Processing query..."):
                    # Create a single row DataFrame from the query
                    df = pd.DataFrame([{"query": query_text}])
                    
                    # Process the query
                    start_time = time.time()
                    results = process_data_batch(
                        df, 
                        prompt_template, 
                        llm, 
                        schema_model, 
                        n_workers,
                        image_data,
                        image_path_columns,
                        image_detail
                    )
                    
                    # Check if any results were returned
                    if results:
                        # Separate successful results from errors
                        success_results = []
                        error_results = []
                        
                        for result in results:
                            if 'error' in result:
                                error_results.append(result)
                            else:
                                success_results.append(result)
                        
                        # Convert successful results to DataFrame
                        if success_results:
                            results_df = pd.DataFrame(success_results)
                            
                            # Store in session state
                            st.session_state.results_df = results_df
                            st.session_state.has_errors = len(error_results) > 0
                            st.session_state.error_results = error_results
                        else:
                            # All results had errors
                            st.session_state.results_df = pd.DataFrame()
                            st.session_state.has_errors = True
                            st.session_state.error_results = error_results
                            st.session_state.error_message = "All processing attempts resulted in errors. Check the error details."
                        
                        # Show processing time
                        elapsed_time = time.time() - start_time
                        time_per_item = elapsed_time / len(results)
                        
                        st.success(f"✅ Query processed in {elapsed_time:.2f} seconds")
                        
                        if success_results:
                            st.success(f"Successfully processed {len(success_results)} of {len(results)} items")
                            
                            # Switch to results tab automatically only if we have successful results
                            results_tab.active = True
                        
                        if error_results:
                            st.warning(f"⚠️ {len(error_results)} of {len(results)} items had errors. See the 'Analyze Data' tab for details.")
                    else:
                        st.session_state.error_message = "No results were generated. Try adjusting your query and try again."
                        st.error(st.session_state.error_message)
        
        except Exception as e:
            st.session_state.error_message = f"Error during processing: {str(e)}"
            st.error(st.session_state.error_message)
            st.exception(e)
        
        finally:
            # Reset processing state
            st.session_state.processing = False
    
    # Display error message if set
    if st.session_state.error_message:
        with results_tab:
            st.error(st.session_state.error_message)
    
    # Display results if available
    with results_tab:
        if hasattr(st.session_state, 'results_df') and not st.session_state.results_df.empty:
                
            # Display results and visualizations
            display_results(st.session_state.results_df, fields)
            create_visualizations(st.session_state.results_df, fields)
        elif not st.session_state.processing and st.session_state.error_message is None:
            # Show help message when no results are available
            st.info("No results to display yet. Configure your analysis in the previous tabs and click 'Analyze Data'.")
            
            # Show sample results screenshot
            st.markdown("""
            ### What to expect
            
            After processing, you'll see:
            
            1. **Table View** - All analysis results in a searchable table
            2. **Card View** - Visual cards showing analysis with images
            3. **Image Gallery** - Browse all images in a grid layout
            4. **Visualizations** - Charts and insights based on the analysis
            """)
        
        # Display any errors embedded in the results
        if hasattr(st.session_state, 'results_df') and not st.session_state.results_df.empty:
            # We've moved error handling to the Analyze Data tab
            pass

if __name__ == "__main__":
    main()