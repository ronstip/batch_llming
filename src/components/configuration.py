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
        
        # API key input with clear instructions and visual cues
        st.markdown("#### 🔑 API Configuration")
        api_key_container = st.container()
        
        with api_key_container:
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                help="Your OpenAI API key is required to use the language models",
                placeholder="sk-..."
            )
            
            # Visual indicator for API key status
            if api_key:
                if api_key.startswith('sk-') and len(api_key) > 20:
                    st.success("✅ API key format is valid")
                else:
                    st.error("❌ API key format appears invalid. It should start with 'sk-'")
            else:
                st.info("ℹ️ Enter your OpenAI API key to get started")
        
        st.markdown("#### 🤖 Model Settings")
        
        # Enhanced model options with better information about capabilities
        model_options = [
            {"name": "gpt-4o-2024-08-06", "multimodal": True, "description": "Latest GPT-4o with multimodal capabilities", "tokens": "High token limit"},
            {"name": "gpt-4o-mini-2024-07-18", "multimodal": True, "description": "Mini version of GPT-4o with multimodal capabilities", "tokens": "Medium token limit"},
            {"name": "gpt-4", "multimodal": False, "description": "Powerful but text-only model", "tokens": "High token limit"},
            {"name": "gpt-3.5-turbo", "multimodal": False, "description": "Faster text-only model", "tokens": "Medium token limit"},
            {"name": "chatgpt-4o-latest", "multimodal": True, "description": "Latest GPT-4o release with multimodal capabilities", "tokens": "High token limit"},
        ]
        
        # Format the model options for display with more info
        model_display_names = [f"{'🖼️ ' if model['multimodal'] else '📝 '}{model['name']} - {model['tokens']}" for model in model_options]
        model_names = [model["name"] for model in model_options]
        
        selected_display_name = st.selectbox(
            "LLM Model", 
            model_display_names, 
            index=0,
            help="Models with 🖼️ support image input for multimodal analysis"
        )
        
        # Extract the actual model name from the selection
        selected_index = model_display_names.index(selected_display_name)
        model_name = model_names[selected_index]
        selected_model = model_options[selected_index]
        
        # Show model details in an expander
        with st.expander("Model Details"):
            st.markdown(f"**Selected Model:** {model_name}")
            st.markdown(f"**Description:** {selected_model['description']}")
            st.markdown(f"**Multimodal Support:** {'Yes ✅' if selected_model['multimodal'] else 'No ❌'}")
            st.markdown(f"**Token Capacity:** {selected_model['tokens']}")
            
            if selected_model['multimodal']:
                st.markdown("**Capabilities:** Can analyze both text and images")
            else:
                st.markdown("**Capabilities:** Text analysis only")
        
        # Show multimodal information if a multimodal model is selected
        is_multimodal = selected_model["multimodal"]
        if is_multimodal:            
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
        
        st.markdown("#### ⚙️ Advanced Settings")
        model_options_col1, model_options_col2 = st.columns(2)
        with model_options_col1:
            temperature = st.slider(
                "Temperature", 
                0.0, 1.0, 0.7, 0.1, 
                help="Higher values make output more random, lower values more deterministic"
            )
            st.caption("↑ Creativity vs Consistency ↓")
        with model_options_col2:
            max_tokens = st.number_input(
                "Max Tokens", 
                min_value=100, 
                max_value=4000, 
                value=1000, 
                step=100,
                help="Maximum number of tokens in the response"
            )
            st.caption("Response size limit")
        
        # Initialize model if API key is provided with better error handling
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
                
                with st.spinner("Initializing model..."):
                    llm = ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        model_kwargs=model_kwargs
                    )
                    st.success("✅ Model initialized successfully")
            except Exception as e:
                st.error(f"Failed to initialize model: {str(e)}")
                st.info("Common issues: Invalid API key, network problems, or model availability. Please check your API key and try again.")
        else:
            st.warning("⚠️ Please enter your OpenAI API key to proceed")
            
        # Add a section for batch processing settings
        st.markdown("#### ⚡ Performance Settings")
        n_workers = st.slider(
            "Concurrent Workers", 
            min_value=1, 
            max_value=40, 
            value=5,
            help="Number of concurrent workers for processing data. Higher values may process faster but could hit rate limits."
        )
        st.caption("Adjust based on your API rate limits")
    
    return llm, api_key, image_detail if is_multimodal else None, n_workers

def render_analysis_config():
    """Render the analysis configuration section."""
    st.header("Analysis Configuration")
    
    # Prompt Configuration
    st.subheader("Prompt Configuration")
    
    # Improved prompt template with simpler format
    prompt_text = st.text_area(
        label="Enter your prompt template:",
        height=300,
        value="""Analyze the following social media post:

Title: {title}
"""
    )
    
    # Schema Builder
    st.subheader("Output Schema")
    
    # Default fields with image analysis field
    default_fields = [
        {"key": "sentiment", "type": "enum", "description": "The sentiment of the post", "options": ["positive", "negative", "neutral"]},
        {"key": "themes", "type": "str", "description": "Main themes in the post separated by commas"},
        # {"key": "target_audience", "type": "str", "description": "Description of the target audience"},
        # {"key": "engagement_potential", "type": "enum", "description": "Expected engagement level", "options": ["high", "medium", "low"]},
        # {"key": "image_description", "type": "str", "description": "Description of the image content if an image is included"}
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
        
        # Create two columns for field display
        field_columns = st.columns(2)
        
        # Iterate through existing fields
        fields_to_remove = []
        for i, field in enumerate(st.session_state.schema_fields):
            # Alternate between left and right columns
            with field_columns[i % 2]:
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
                st.session_state.results_df = None
                st.session_state.raw_results = None
            
    # Return the prompt template and schema fields
    return prompt_text, st.session_state.schema_fields 