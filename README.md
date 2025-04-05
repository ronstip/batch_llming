# Social Media Post Analyzer

A Streamlit application for analyzing social media posts using OpenAI's language models. The app extracts insights such as sentiment, themes, target audience, and engagement potential from social media posts.

## Features

- Upload CSV files containing social media posts
- Configure analysis parameters using a dynamic form
- Customize the output schema to fit your specific needs
- Analyze posts using OpenAI's language models
- Visualize results with interactive charts
- Export analysis results to CSV

## Project Structure

```
├── app.py                 # Main application entry point
├── src/                   # Source code directory
│   ├── components/        # UI Components
│   │   ├── data_input.py  # Data input component
│   │   └── configuration.py # Configuration components
│   ├── models/            # Data models
│   │   └── pydantic_models.py # Pydantic model definitions
│   ├── utils/             # Utility functions
│   │   ├── processing.py  # Data processing utilities
│   │   └── styles.py      # UI styling utilities
│   └── visualizations/    # Visualization components
│       └── display.py     # Results display functions
```

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install streamlit pandas matplotlib seaborn langchain langchain-openai
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter your OpenAI API key in the sidebar
2. Upload a CSV file containing social media posts
3. Configure the prompt template and output schema
4. Click "Process Data" to analyze the posts
5. View the results in the Results tab

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- LangChain
- OpenAI API key

## License

This project is licensed under the MIT License - see the LICENSE file for details.