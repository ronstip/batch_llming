import traceback
import pandas as pd
from typing import Dict, Any, Type
from langchain.pydantic_v1 import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

def process_post(row, prompt_template, model, output_model):
    """Process a single post using the LLM and structured output."""
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

def process_data_batch(df, prompt_template, llm, output_model, use_threading=False, test_only_first_row=False):
    """Process a batch of data using either threaded or sequential processing."""
    results = []
    total_rows = len(df)
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    if use_threading:
        # Define max workers based on data size
        max_workers = min(32, total_rows)
        
        # Original threaded processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit all tasks
            for i, (_, row) in enumerate(df.iterrows()):
                future = executor.submit(
                    process_post, 
                    row, 
                    prompt_template, 
                    llm, 
                    output_model
                )
                futures.append(future)
            
            # Process results as they complete
            for i, future in enumerate(futures):
                results.append(future.result())
                progress_bar.progress((i + 1) / total_rows)
    else:
        # Sequential processing for debugging
        st.info("Using sequential processing for debugging")
        
        rows_to_process = df.iterrows()
        if test_only_first_row:
            # Get only the first row
            rows_to_process = [next(df.iterrows())]
        
        for i, (_, row) in enumerate(rows_to_process):
            st.write(f"Processing row {i+1}")
            
            # Process with more detailed logging
            try:
                result = process_post(row, prompt_template, llm, output_model)
                st.write("✅ Row processed successfully")
                if "_debug_info" in result:
                    with st.expander(f"Debug info for row {i+1}"):
                        st.json(result["_debug_info"])
            except Exception as e:
                st.error(f"Error in main processing loop: {str(e)}")
                st.code(traceback.format_exc(), language="python")
                result = {"error": str(e), "error_type": str(type(e))}
            
            results.append(result)
            progress_bar.progress((i + 1) / (1 if test_only_first_row else total_rows))
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Filter columns based on the current fields plus original_ columns
    field_keys = [f["key"] for f in st.session_state.get('current_processing_fields', [])]
    original_columns = [col for col in results_df.columns if col.startswith("original_")]
    valid_columns = field_keys + original_columns + ["error"]
    
    # Only keep relevant columns that exist in the dataframe
    filtered_columns = [col for col in valid_columns if col in results_df.columns]
    if filtered_columns:
        results_df = results_df[filtered_columns]
    
    return results_df 