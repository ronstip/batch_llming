import streamlit as st
import json
import time
import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI

def render_sidebar_config():
    """Render the sidebar configuration section."""
    with st.sidebar:
        st.header("Configuration")
        
        # API key input with clear instructions
        st.markdown("#### 🔑 API Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Your OpenAI API key is required to use the language models")
        
        st.markdown("#### 🤖 Model Settings")
        
        # Define model options with information about multimodal support
        model_options = [
            {"name": "gpt-4o-2024-08-06", "multimodal": True, "description": "Latest GPT-4o with multimodal capabilities"},
            {"name": "gpt-4o-mini-2024-07-18", "multimodal": True, "description": "Mini version of GPT-4o with multimodal capabilities"},
            {"name": "gpt-4", "multimodal": False, "description": "Powerful but text-only model"},
            {"name": "gpt-3.5-turbo", "multimodal": False, "description": "Faster text-only model"},
            {"name": "chatgpt-4o-latest", "multimodal": True, "description": "Latest GPT-4o release with multimodal capabilities"},
        ]
        
        # Format the model options for display
        model_display_names = [f"{'🖼️ ' if model['multimodal'] else '📝 '}{model['name']}" for model in model_options]
        model_names = [model["name"] for model in model_options]
        
        selected_display_name = st.selectbox("LLM Model", model_display_names, index=0,
                                           help="Models with 🖼️ support image input for multimodal analysis")
        
        # Extract the actual model name from the selection
        selected_index = model_display_names.index(selected_display_name)
        model_name = model_names[selected_index]
        
        # Show multimodal information if a multimodal model is selected
        is_multimodal = model_options[selected_index]["multimodal"]
        if is_multimodal:
            st.success("✅ Selected model supports multimodal (text + image) input")
            
            # Add image detail level option for multimodal models
            st.markdown("#### 🖼️ Image Settings")
            image_detail = st.selectbox(
                "Image Detail Level", 
                ["high", "low"], 
                index=0,
                help="High: Better image analysis with higher token usage. Low: More efficient with lower token usage."
            )
        else:
            st.warning("⚠️ Selected model is text-only and does not support image input")
            # Set default image detail level for non-multimodal models
            image_detail = "low"
        
        model_options_col1, model_options_col2 = st.columns(2)
        with model_options_col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                                  help="Higher values make output more random, lower values more deterministic")
        with model_options_col2:
            max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=1000, step=100,
                                       help="Maximum number of tokens in the response")
        
        # Initialize model if API key is provided
        llm = None
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            try:
                # Create model_kwargs dictionary for additional parameters
                model_kwargs = {}
                if is_multimodal:
                    # The detail parameter is no longer supported by the OpenAI API
                    # Keeping image_detail for backward compatibility with other parts of the code
                    pass
                
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_kwargs=model_kwargs
                )
                st.success("✅ Model initialized")
            except Exception as e:
                st.error(f"Failed to initialize model: {str(e)}")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")
    
    return llm, api_key, image_detail if is_multimodal else None

def render_analysis_config():
    """Render the analysis configuration section."""
    st.header("Analysis Configuration")
    
    # Prompt Configuration
    st.subheader("Prompt Configuration")
    
    # Inform users about multimodal capabilities
    st.info("""
    🖼️ **Multimodal Support**: 
    
    - If you specified image path columns, the system will automatically load images from those paths for each row.
    - If you uploaded a single image, it will be used for all rows in your dataset.
    
    The prompt below will be sent along with any images to multimodal-capable models.
    """)
    
    # Improved prompt template with simpler format
    prompt_text = st.text_area(
        label="Enter your prompt template:",
        height=500,
        value="""Analyze the following social media post:

Title: {title}

Please provide the following analysis:
- Overall sentiment (positive, negative, or neutral)
- Key themes or topics
- Target audience
- Engagement potential (high, medium, low)

If there's an image included with this post, analyze the visual content and how it relates to the text.
"""
    )
    
    # Schema Builder
    st.subheader("Output Schema")
    
    # Default fields with image analysis field
    default_fields = [
        {"key": "sentiment", "type": "enum", "description": "The sentiment of the post", "options": ["positive", "negative", "neutral"]},
        {"key": "themes", "type": "str", "description": "Main themes in the post separated by commas"},
        {"key": "target_audience", "type": "str", "description": "Description of the target audience"},
        {"key": "engagement_potential", "type": "enum", "description": "Expected engagement level", "options": ["high", "medium", "low"]},
        {"key": "image_description", "type": "str", "description": "Description of the image content if an image is included"}
    ]
    
    # Initialize schema fields in session state if not present
    if 'schema_fields' not in st.session_state:
        st.session_state.schema_fields = default_fields.copy()
    
    # Dynamic schema builder with UI components
    with st.expander("Schema Builder", expanded=True):
        # Button to add a new field
        if st.button("➕ Add New Field"):
            # Generate a unique key for the session state
            new_field_id = f"field_{len(st.session_state.schema_fields) + 1}_{int(time.time())}"
            
            st.session_state.schema_fields.append({
                "key": new_field_id,
                "type": "str",
                "description": "New field description",
                "options": []
            })
            
            # Rerun to show the new field
            st.rerun()
        
        # Iterate through existing fields
        fields_to_remove = []
        for i, field in enumerate(st.session_state.schema_fields):
            st.markdown(f"### Field {i+1}")
            
            # Create a unique key for each input field
            key_prefix = f"field_{i}_"
            
            # Field key
            field["key"] = st.text_input(
                "Field Key (no spaces, lowercase)", 
                value=field["key"], 
                key=f"{key_prefix}key"
            )
            
            # Field type
            field["type"] = st.selectbox(
                "Field Type", 
                ["str", "int", "float", "enum"], 
                index=["str", "int", "float", "enum"].index(field["type"]) if field["type"] in ["str", "int", "float", "enum"] else 0,
                key=f"{key_prefix}type"
            )
            
            # Field description
            field["description"] = st.text_input(
                "Description", 
                value=field["description"], 
                key=f"{key_prefix}desc"
            )
            
            # If enum type, show options input
            if field["type"] == "enum":
                options_str = st.text_input(
                    "Options (comma separated)", 
                    value=", ".join(field.get("options", [])), 
                    key=f"{key_prefix}options"
                )
                field["options"] = [opt.strip() for opt in options_str.split(",") if opt.strip()]
        
            # Remove button with unique key
            remove_button_id = f"remove_{i}_{field.get('key', '')}"
            if st.button("🗑️ Remove", key=remove_button_id):
                fields_to_remove.append(i)
                # Set a flag to rerun after removing fields
                st.session_state.need_rerun = True
            
            st.markdown("---")
        
        # Remove marked fields (in reverse to avoid index issues)
        if fields_to_remove:
            for idx in sorted(fields_to_remove, reverse=True):
                if idx < len(st.session_state.schema_fields):
                    st.session_state.schema_fields.pop(idx)
            
            # Clear previous results when schema changes
            if 'results_df' in st.session_state:
                del st.session_state['results_df']
            
            # Force a rerun to update the UI
            st.rerun()
    
    # Check if we need to rerun after removing fields
    if st.session_state.get('need_rerun', False):
        st.session_state.need_rerun = False
        st.rerun()
    
    # Use the current fields from session state and ensure we're using a copy
    fields = st.session_state.schema_fields.copy()
    
    # Force a session state update when fields change
    if 'fields_hash' not in st.session_state:
        st.session_state.fields_hash = ""
    
    # Create a hash of the current fields to detect changes
    current_fields_hash = json.dumps(fields)
    if current_fields_hash != st.session_state.fields_hash:
        st.session_state.fields_hash = current_fields_hash
        # Clear previous results when schema changes
        if 'results_df' in st.session_state:
            del st.session_state['results_df']
    
    # Display JSON representation in an expander
    with st.expander("JSON Representation"):
        st.code(json.dumps(fields, indent=2), language="json")
    
    return prompt_text, fields 