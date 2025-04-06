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

def load_and_resize_image(image_path, max_width=150):
    """Load an image from path and resize it to create a thumbnail"""
    try:
        # Handle both URLs and local file paths
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            # For local file paths
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                return None
        
        # Calculate new height while maintaining aspect ratio
        width_percent = max_width / float(image.width)
        new_height = int(float(image.height) * width_percent)
        
        # Resize the image
        resized_image = image.resize((max_width, new_height), Image.LANCZOS)
        return resized_image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to display results with metrics and tables
def display_results(results_df, fields):
    st.header("Analysis Results")
    
    # Create metrics row
    metric_cols = st.columns(4)
    
    # Count total posts
    with metric_cols[0]:
        st.metric("Total Posts", len(results_df))
    
    # Get all field keys
    field_keys = [field['key'] for field in fields]
    
    # Display debug info in an expander
    with st.expander("Debug Information", expanded=False):
        st.write("Current fields:")
        st.json(fields)
        st.write("Available columns in results:")
        st.write(list(results_df.columns))
    
    # Find most common sentiment if present
    if 'sentiment' in field_keys and 'sentiment' in results_df.columns:
        with metric_cols[1]:
            most_common = results_df['sentiment'].value_counts().idxmax()
            st.metric("Most Common Sentiment", most_common)
    
    # Find average engagement if present
    if 'engagement_potential' in field_keys and 'engagement_potential' in results_df.columns:
        with metric_cols[2]:
            engagement_map = {'high': 3, 'medium': 2, 'low': 1}
            try:
                if results_df['engagement_potential'].map(engagement_map).mean() > 2:
                    engagement_level = "High"
                elif results_df['engagement_potential'].map(engagement_map).mean() > 1:
                    engagement_level = "Medium"
                else:
                    engagement_level = "Low"
                st.metric("Average Engagement", engagement_level)
            except:
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
        # Identify image path columns in the dataframe
        image_path_columns = [col for col in results_df.columns if col.startswith('original_') and 
                            any(path_col in col for path_col in st.session_state.get('image_path_columns', []))]
        
        # Display the main dataframe
        st.dataframe(results_df, use_container_width=True)
        
        # If we have image path columns, create an image viewer section
        if image_path_columns:
            with st.expander("🖼️ View Images", expanded=True):
                # Display images in a grid layout - 5 rows per page
                rows_per_page = 5
                total_pages = (len(results_df) + rows_per_page - 1) // rows_per_page
                
                page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1) - 1
                start_idx = page * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(results_df))
                
                for idx in range(start_idx, end_idx):
                    row = results_df.iloc[idx]
                    st.write(f"**Row {idx}:**")
                    
                    # Create a grid of images
                    valid_image_cols = []
                    for col in image_path_columns:
                        img_path = row[col]
                        if pd.notna(img_path) and str(img_path).strip():
                            valid_image_cols.append(col)
                    
                    if valid_image_cols:
                        # Calculate grid dimensions
                        cols_per_row = 4  # Display 4 images per row
                        num_rows = (len(valid_image_cols) + cols_per_row - 1) // cols_per_row
                        
                        for r in range(num_rows):
                            image_cols = st.columns(cols_per_row)
                            
                            for c in range(cols_per_row):
                                col_idx = r * cols_per_row + c
                                if col_idx < len(valid_image_cols):
                                    col = valid_image_cols[col_idx]
                                    img_path = row[col]
                                    col_name = col.replace('original_', '')
                                    
                                    with image_cols[c]:
                                        # Load and display the image
                                        img = load_and_resize_image(img_path)
                                        if img:
                                            st.image(img, caption=col_name)
                                        else:
                                            st.write(f"[{col_name}]({img_path})")
                    else:
                        st.write("No images available")
                    
                    st.write("---")
        
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
        for i in range(0, len(results_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(results_df.head(20)):
                    row = results_df.iloc[i + j]
                    with cols[j]:
                        st.markdown("### Post Analysis")
                        st.markdown(f"**Title:** {row.get('original_title', 'N/A')}")
                        
                        # Display analysis fields
                        for field in fields:
                            field_key = field['key']
                            if field_key in row:
                                st.markdown(f"**{field_key.replace('_', ' ').title()}:** {row[field_key]}")
                        
                        # Display images if path exists
                        image_cols = [col for col in row.index if col.startswith('original_') and 
                                    any(path_col in col for path_col in st.session_state.get('image_path_columns', []))]
                        
                        if image_cols:
                            st.markdown("**Images:**")
                            
                            # Create a 2x2 grid for thumbnails
                            img_grid_size = min(2, len(image_cols))
                            if img_grid_size > 0:
                                for img_idx in range(0, len(image_cols), img_grid_size):
                                    img_row = st.columns(img_grid_size)
                                    
                                    for grid_idx in range(img_grid_size):
                                        col_idx = img_idx + grid_idx
                                        if col_idx < len(image_cols):
                                            img_col = image_cols[col_idx]
                                            img_path = row.get(img_col)
                                            
                                            if pd.notna(img_path) and str(img_path).strip():
                                                col_name = img_col.replace('original_', '')
                                                
                                                with img_row[grid_idx]:
                                                    # Load and display thumbnails
                                                    img = load_and_resize_image(img_path, max_width=120)
                                                    if img:
                                                        st.image(img, caption=col_name)
                                                    else:
                                                        st.write(f"[{col_name}]({img_path})")
                        
                        st.markdown("---")

# Main app
def main():
    # Apply custom CSS
    apply_custom_css()
    
    st.title("🔄 Batch LLM Processing")
    st.write("Process multiple inputs using Large Language Models with structured outputs")

    # Initialize session state
    if 'llm_initialized' not in st.session_state:
        st.session_state.llm_initialized = False
    
    # Sidebars and configuration
    llm, api_key, image_detail = render_sidebar_config()
    
    # Make sure LLM is initialized
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue")
        st.stop()
    
    st.session_state.llm_initialized = llm is not None
    
    if not st.session_state.llm_initialized:
        st.error("The language model failed to initialize. Please check your API key and settings.")
        st.stop()
    
    # Main sections
    uploaded_file, df, image_data, image_path_columns, input_mode, query_text = render_data_input()
    prompt_template, fields = render_analysis_config()
    
    # Debugging info for image data
    if image_data:
        st.sidebar.success("✅ Global image loaded for all rows")
    
    # Create the processing button
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("🚀 Analyze Data", type="primary", use_container_width=True)
        
    with col2:
        # Number of workers selection
        n_workers = st.number_input("Parallel Workers", min_value=1, max_value=10, value=2,
                                  help="Number of parallel processing workers. Higher values process faster but use more resources.")
    
    # Generate a schema model from the fields
    schema_model = create_dynamic_model("OutputSchema", fields)
    
    # Process data when the button is clicked
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    query_mode = input_mode == "Write Query" and query_text
    file_mode = uploaded_file is not None
    
    if analyze_button and (file_mode or query_mode):
        if file_mode and df is not None and df.shape[0] > 0:
            with st.spinner("Processing data..."):
                # Process the data using the configured LLM
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
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Store in session state
                    st.session_state.results_df = results_df
                    
                    # Show processing time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Processed {len(results)} items in {elapsed_time:.2f} seconds")
                else:
                    st.error("No results were generated. Check your data and try again.")
        elif query_mode:
            with st.spinner("Processing query..."):
                # Create a single row DataFrame from the query
                df = pd.DataFrame([{"query": query_text}])
                
                # Process the data using the configured LLM
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
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Store in session state
                    st.session_state.results_df = results_df
                    
                    # Show processing time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Processed query in {elapsed_time:.2f} seconds")
                else:
                    st.error("No results were generated. Try adjusting your query and try again.")
        else:
            st.error("Please upload a file with data or enter a query first")
    elif analyze_button:
        st.error("Please upload a CSV file or enter a query first")
    
    # Display results if available
    if st.session_state.results_df is not None:
        display_results(st.session_state.results_df, fields)
        create_visualizations(st.session_state.results_df, fields)
    
if __name__ == "__main__":
    main()