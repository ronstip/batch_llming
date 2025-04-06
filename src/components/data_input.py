import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import os

def render_data_input():
    """Render the data input section of the app."""
    st.header("Input Data")
    
    input_col1, input_col2 = st.columns([3, 2])
    
    with input_col1:
        input_mode = st.radio("Input Mode", ["Upload CSV", "Write Query"], horizontal=True)
        
        uploaded_file = None
        query_text = None
        df = None
        
        if input_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ CSV loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
                    
                except Exception as e:
                    st.error(f"❌ Error loading CSV: {str(e)}")
        else:
            query_text = st.text_area("Enter your query:", "Analyze 10 social media posts for sentiment and themes.")
    
        # Image configuration section
        st.markdown("### Image Configuration")
        image_mode = st.radio("Image Mode", ["No Images", "Single Image for All Rows", "Image Paths in CSV"], horizontal=True)
        st.info("""
                🖼️ **Multimodal Support**:
- If you specified image path columns, the system will automatically load images from those paths for each row.     
- If you uploaded a single image, it will be used for all rows in your dataset.    
      
The prompt below will be sent along with any images to multimodal-capable models.    
""")
        image_data = None
        image_path_columns = []
        
        if image_mode == "Single Image for All Rows":
            st.markdown("Upload a single image to use for all rows in your dataset")
            uploaded_image = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
            
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
                verify_button = st.button("Verify Image Paths")
                
                if verify_button:
                    valid_paths = 0
                    invalid_paths = 0
                    
                    for col in image_path_columns:
                        for path in df[col].dropna():
                            if os.path.exists(path):
                                valid_paths += 1
                            else:
                                invalid_paths += 1
                    
                    if invalid_paths > 0:
                        st.warning(f"⚠️ Found {invalid_paths} invalid image paths")
                    if valid_paths > 0:
                        st.success(f"✅ Found {valid_paths} valid image paths")
    
    with input_col2:
        if df is not None and not df.empty:
            st.subheader("Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Display column info in an expander
            with st.expander("Column Information"):
                for col in df.columns:
                    st.write(f"- **{col}**: {df[col].dtype}")
    
    return uploaded_file, df, image_data, image_path_columns, input_mode, query_text 