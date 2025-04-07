import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import os

def render_data_input():
    """Render the data input section of the app."""
    st.header("Input Data")
    
    # Create a better layout for input options
    input_col1, input_col2 = st.columns([3, 2])
    
    with input_col1:
        input_mode = st.radio(
            "Input Mode", 
            ["Upload CSV", "Write Query"], 
            horizontal=True,
            help="Choose how you want to provide data for analysis"
        )
        
        uploaded_file = None
        query_text = None
        df = None
        
        if input_mode == "Upload CSV":
            # Add instructions above the uploader
            st.info("📋 Upload a CSV file containing your social media posts. The CSV should have a column that contains the text of the posts.")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file", 
                type=['csv'],
                help="Make sure your CSV has proper headers and contains the content to analyze"
            )
            
            if uploaded_file is not None:
                try:
                    # Add a spinner while loading CSV
                    with st.spinner("Loading CSV..."):
                        df = pd.read_csv(uploaded_file)
                    
                    if df.empty:
                        st.error("❌ CSV file is empty")
                    else:
                        st.success(f"✅ CSV loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
                        
                        # Show quick summary stats
                        with st.expander("CSV Summary Statistics"):
                            # Get the memory usage of each column
                            mem_usage = df.memory_usage(deep=True)
                            col_mem_usage = []
                            for col in df.columns:
                                col_mem_usage.append(mem_usage[col] / 1024)  # Convert to KB
                            
                            # Add memory usage to dtype info
                            dtype_info = pd.DataFrame({
                                'Column': df.columns,
                                'Type': df.dtypes.values,
                                'Memory (KB)': col_mem_usage,
                                'Non-Null Count': df.count().values,
                                'Null %': [(1 - df.count().values[i]/len(df)) * 100 for i in range(len(df.columns))]
                            })
                            
                            st.dataframe(dtype_info, use_container_width=True)
                    
                except pd.errors.EmptyDataError:
                    st.error("❌ Uploaded CSV file is empty")
                    df = None
                except pd.errors.ParserError:
                    st.error("❌ Error parsing CSV file. Please check the format")
                    df = None
                except Exception as e:
                    st.error(f"❌ Error loading CSV: {str(e)}")
                    df = None
        else:
            st.info("✍️ Enter your text query for analysis. The model will analyze this as a single item.")
            query_text = st.text_area(
                "Enter your query:", 
                "Analyze this social media post for sentiment and themes.",
                height=150,
                help="Type the text you want to analyze"
            )
            
            # Create a dataframe from the query text if provided
            if query_text:
                df = pd.DataFrame([{"text": query_text}])
    
        # Image configuration section with better UI
        st.markdown("### Image Configuration")
        st.markdown("Add image data to enhance your analysis with multimodal models.")
        
        image_mode = st.radio(
            "Image Mode", 
            ["No Images", "Single Image for All Rows", "Image Paths in CSV"], 
            horizontal=True,
            help="Choose how to include images in your analysis"
        )
        
        # More informative help text
        if image_mode != "No Images":
            st.info("""
            🖼️ **Multimodal Analysis**:
            - Images provide additional context for the AI to analyze
            - Make sure you've selected a multimodal-capable model in the sidebar
            - Image analysis works best with high-quality, relevant images
            """)
        
        image_data = None
        image_path_columns = []
        
        if image_mode == "Single Image for All Rows":
            st.markdown("Upload a single image to use for all rows in your dataset")
            uploaded_image = st.file_uploader(
                "Upload image", 
                type=['jpg', 'jpeg', 'png'],
                help="This image will be used for all rows in your analysis"
            )
            
            if uploaded_image is not None:
                try:
                    # Display the uploaded image
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", width=300)
                    
                    # Process image for use with multimodal LLMs
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    image_data = {
                        "bytes": buffered.getvalue(),
                        "type": uploaded_image.type,
                        "name": uploaded_image.name,
                        "base64": base64.b64encode(buffered.getvalue()).decode()
                    }
                    
                    st.success("✅ Image loaded successfully")
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
        elif image_mode == "Image Paths in CSV" and df is not None and not df.empty:
            st.markdown("Select columns that contain paths to image files")
            st.info("The system will load images from these paths for each row in your dataset")
            
            # Allow users to select columns that contain image paths
            available_columns = df.columns.tolist()
            image_path_columns = st.multiselect(
                "Select columns containing image paths",
                options=available_columns,
                help="Select one or more columns that contain file paths to images"
            )
            
            if image_path_columns:
                # Show example path from first row
                with st.expander("Example Image Paths"):
                    for col in image_path_columns:
                        if not df[col].empty:
                            example_path = df[col].iloc[0]
                            st.write(f"**{col}**: `{example_path}`")
                
                # Verify if paths in the selected columns exist
                st.markdown("#### Path Verification")
                verify_button = st.button("Verify Image Paths", type="secondary")
                
                if verify_button:
                    valid_paths = 0
                    invalid_paths = 0
                    
                    # Create a progress bar for verification
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_paths = sum(len(df[col].dropna()) for col in image_path_columns)
                    checked_paths = 0
                    
                    for col in image_path_columns:
                        for path in df[col].dropna():
                            if os.path.exists(path):
                                valid_paths += 1
                            else:
                                invalid_paths += 1
                            
                            checked_paths += 1
                            progress = checked_paths / total_paths
                            progress_bar.progress(progress)
                            status_text.text(f"Checking paths: {checked_paths}/{total_paths}")
                    
                    if invalid_paths > 0:
                        st.warning(f"⚠️ Found {invalid_paths} invalid image paths")
                        # Show tips for fixing paths
                        st.info("""
                        **Tips for fixing invalid paths:**
                        - Make sure paths are either absolute or relative to the current working directory
                        - Check for typos in filenames or directory names
                        - Verify file permissions
                        """)
                    if valid_paths > 0:
                        st.success(f"✅ Found {valid_paths} valid image paths")
    
    with input_col2:
        if df is not None and not df.empty:
            st.subheader("Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display column info in an expander with better formatting
            with st.expander("Column Information"):
                col_list = []
                for col in df.columns:
                    # Get basic stats for the column
                    null_count = df[col].isna().sum()
                    null_percent = (null_count / len(df)) * 100
                    unique_values = df[col].nunique()
                    
                    col_list.append({
                        "Column": col,
                        "Type": str(df[col].dtype),
                        "Unique Values": unique_values,
                        "Null Count": null_count,
                        "Null %": f"{null_percent:.1f}%"
                    })
                
                # Display as a dataframe
                col_df = pd.DataFrame(col_list)
                st.dataframe(col_df, use_container_width=True)
    
    return uploaded_file, df, image_data, image_path_columns, input_mode, query_text 