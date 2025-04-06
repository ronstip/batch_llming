import traceback
import pandas as pd
from typing import Dict, Any, Type, List, Optional
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage
import os
from PIL import Image
import io
import base64
import time

def load_image_from_path(image_path: str) -> Optional[Dict[str, Any]]:
    """Load an image from a file path and return it in the required format."""
    try:
        if not os.path.exists(image_path):
            return None
            
        # Open and process the image
        image = Image.open(image_path)
        
        # Determine the image type from path
        image_type = f"image/{os.path.splitext(image_path)[1][1:].lower()}"
        if image_type == "image/jpg":
            image_type = "image/jpeg"
            
        # Convert to bytes
        buffered = io.BytesIO()
        image.save(buffered, format=image_type.split('/')[1].upper())
        
        # Create image data dict
        image_data = {
            "bytes": buffered.getvalue(),
            "type": image_type,
            "name": os.path.basename(image_path),
            "base64": base64.b64encode(buffered.getvalue()).decode()
        }
        
        return image_data
    except Exception as e:
        st.warning(f"Failed to load image from {image_path}: {str(e)}")
        return None

def process_post(row, prompt_template, model, output_model, image_data=None, image_path_columns=None, image_detail=None):
    """Process a single post using the LLM and structured output.
    
    Args:
        row: The row of data to process
        prompt_template: The prompt template to use
        model: The LLM model to use
        output_model: The pydantic model for structured output
        image_data: Optional global image data to use
        image_path_columns: Optional list of columns containing image paths
        image_detail: Deprecated parameter, kept for compatibility
    """
    try:
        # Create a dictionary to hold debug info
        debug_info = {
            "start_time": time.time()
        }
        
        # Convert row to a dictionary of post values
        post_values = row.to_dict()
        
        # Format the prompt template
        try:
            formatted_prompt = prompt_template.format(**post_values)
            debug_info["prompt_formatted"] = True
        except Exception as format_err:
            debug_info["prompt_format_error"] = str(format_err)
            formatted_prompt = prompt_template  # Fall back to unformatted template
        
        debug_info["prompt"] = formatted_prompt
        
        # Handle image data loading from specific column if specified
        row_image_data = None
        
        if image_path_columns:
            debug_info["image_path_columns"] = image_path_columns
            
            # Try to load image from any of the specified columns
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
            
            # Add image as content part with detail level if specified
            image_url_data = {
                "url": f"data:{row_image_data['type']};base64,{row_image_data['base64']}"
            }
            
            # Remove detail parameter as it's no longer supported
            if image_detail:
                debug_info["image_detail_level"] = image_detail
                # Note: detail parameter removed as it's not supported by the API
            
            content.append({
                "type": "image_url",
                "image_url": image_url_data
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

def process_data_batch(df, prompt_template, llm, output_model, n_workers=2, image_data=None, image_path_columns=None, image_detail=None):
    """Process a batch of data using parallel processing.
    
    Args:
        df: The dataframe to process
        prompt_template: The prompt template to use
        llm: The LLM model to use
        output_model: The pydantic model for structured output
        n_workers: Number of parallel workers to use
        image_data: Optional global image data to use
        image_path_columns: Optional list of columns containing image paths
        image_detail: Deprecated parameter, kept for compatibility
    """
    results = []
    total_rows = len(df)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Show processing information
    status_text.text(f"Processing {total_rows} items with {n_workers} workers...")
    
    # Define max workers based on user selection and data size
    max_workers = min(n_workers, total_rows)
    
    # Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Submit all tasks
        for i, (_, row) in enumerate(df.iterrows()):
            future = executor.submit(
                process_post, 
                row, 
                prompt_template, 
                llm, 
                output_model,
                image_data,
                image_path_columns,
                None  # Pass None instead of image_detail
            )
            futures.append(future)
        
        # Process results as they complete
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                st.error(f"Error processing item {i+1}: {str(e)}")
                results.append({
                    "error": str(e),
                    "error_type": str(type(e)),
                })
            
            # Update progress
            progress = (i + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{total_rows} items ({progress:.1%})")
    
    return results 