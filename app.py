import traceback
import streamlit as st
import pandas as pd
import base64
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, get_type_hints, Dict, Any, Type, List, Optional
from langchain.pydantic_v1 import BaseModel, Field, create_model
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
import time
import os
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
from src.utils.processing import process_data_batch
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

def process_post(row, prompt_template, model, output_model):
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
        
        # Create a chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("human", formatted_prompt)
        ])
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
                        
                        st.markdown("---")

# Main app
def main():
    st.title("Social Media Post Analysis with LLM")
    
    # Create a horizontal layout for the top section
    header_col1, header_col2 = st.columns([2, 1])
    
    with header_col2:
        st.image("https://img.icons8.com/color/96/000000/analytics.png", width=100)
    
    with header_col1:
        st.markdown("""
        This app analyzes social media posts using OpenAI's language models to extract insights about sentiment, 
        themes, target audience, and engagement potential.
        """)
    
    # Initialize the model in the sidebar
    llm, api_key = render_sidebar_config()
    
    # Create tabs for main workflow
    tabs = ["📊 Data Input", "🔧 Configuration", "📈 Results"]
    tab1, tab2, tab3 = st.tabs(tabs)
    
    # Logic to select the active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # Default to first tab
        
    # Data Input tab
    with tab1:
        df, query_text = render_data_input()
    
    # Configuration tab
    with tab2:
        prompt_text, fields = render_analysis_config()
        
        # Create the output model if possible
        try:
            OutputModel = create_dynamic_model("OutputModel", fields)
            with st.expander("Output Model Schema"):
                st.code(OutputModel.schema_json(indent=2), language="json")
        except Exception as e:
            st.error(f"❌ Error creating output model: {str(e)}")
            OutputModel = None
        
        # Process button - centered
        process_button_col = st.columns([1, 2, 1])[1]
        with process_button_col:
            process_button = st.button("🚀 Process Data", type="primary", 
                                     use_container_width=True,
                                     disabled=not api_key)
    
    # Results tab
    with tab3:
        # This tab will be filled with results after processing
        if 'results_df' not in st.session_state:
            st.info("Process your data to see results here")
    
    # Process data when button is clicked
    use_threading = st.checkbox("Use threaded processing", value=False, 
                          help="Disable for debugging if you encounter issues")
    test_only_first_row = st.checkbox("Test with only first row", value=True)
    
    if process_button and api_key:
        try:
            if df is not None and not df.empty:
                # Process the uploaded data
                with st.spinner("Processing posts..."):
                    # Save the fields to session state before processing
                    st.session_state['current_processing_fields'] = fields.copy()
                    
                    # Create the output model
                    OutputModel = create_dynamic_model("OutputModel", fields)
                    
                    # Process the data
                    results_df = process_data_batch(
                        df,
                        prompt_text,
                        llm,
                        OutputModel,
                        use_threading=use_threading,
                        test_only_first_row=test_only_first_row
                    )
                    
                    # Store in session state
                    st.session_state['results_df'] = results_df
                    st.session_state['fields'] = fields.copy()
                    
                    # Set active tab to Results (index 2)
                    st.session_state.active_tab = 2
                    
                    # Notify completion
                    st.container().success("✅ Processing complete! Navigate to the Results tab to see the analysis.")
                    st.rerun()
                    
            else:
                st.error("No data to process. Please upload a CSV file or write a query.")
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
    
    # Display results if available
    if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
        with tab3:
            # Display any errors
            display_errors(st.session_state['results_df'])
            
            # Look for debug info
            if '_debug_info' in st.session_state['results_df'].columns:
                with st.expander("🔍 Debug Information", expanded=False):
                    sample_debug = st.session_state['results_df']['_debug_info'].iloc[0]
                    st.json(sample_debug)
            
            # Display results
            display_results(st.session_state['results_df'], st.session_state['fields'])
            
            # Create visualizations
            create_visualizations(st.session_state['results_df'], st.session_state['fields'])

if __name__ == "__main__":
    main()