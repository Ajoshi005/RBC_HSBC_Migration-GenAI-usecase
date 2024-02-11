import streamlit as st
from embeddings import query_llm  # Import your function from function.py
import os

# ----------Setting Langsmith params for tracing----------------------------#

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "RBC-HSBC-GENAI-USECASE"

# ----------BUILDING the APP.py----------------------------#

# Set Streamlit app title with emojis related to customer relationship management
st.title("🧑‍💼 Customer Relationship Manager AI Assistant 🤝")

# Define the professional color palette for Streamlit
rbc_canada_palette = {
    "primary": "#ff4b4b",
    "secondary": "#262730",  # A warm orange color used for accentuating elements like borders and icons.
    "background": "#0e1117",  # A dark blue color used as the background color of the interface.
    "text": "#fafafa",  # A soft peach color used for text to ensure better readability.
    "accent1": "#E9C46A",  # A mustard yellow color used as an accent for highlighting specific elements.
    "accent2": "#F4A261",  # A shade of orange used for additional accents or to add visual variety.
    "accent3": "#2A9D8F"  # A slightly lighter shade of teal-green used for highlighting specific elements or actions.
}

# Set page background color and text color using RBC Bank Canada color palette
st.markdown(
    f"""
    <style>
    body {{
        background-color: {rbc_canada_palette["background"]};
        color: {rbc_canada_palette["text"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Get user input from text area
user_input = st.text_area("Please enter your query on transition of your HSBC accounts to RBC:")

# Check if the user has entered any input
if user_input:
    # Process user input and get the model's output and sources
    model_output, sources = query_llm(user_input)

    # Display the model output in a formatted text format
    st.markdown(
        f"""
        <div style='background-color: {rbc_canada_palette["secondary"]}; padding: 10px;'>
            <p style='color: {rbc_canada_palette["accent1"]}; font-size: 16px;'>
                {model_output}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Display the sources in a separate block
    for source in sources:
        st.write(source.metadata["page"])
        st.write(source.metadata["source"])
        st.write("")
        st.write("")
        # st.write(source['page_content'])  # Print the source
        # st.write(source["metadata"]["page"])
        # st.write(source["metadata"]["source"])

        # Add a heading to the sidebar
st.sidebar.header("About the App")

# Add a description to the sidebar
st.sidebar.write(
    "The App responds to your questions on HSBC RBC Transition based on the RBC Product Migration Guide"
)
RBC_doc_url = "https://www.rbc.com/hsbc-canada/product-service-guide.html?#personal"
st.sidebar.markdown(
    f'<a href="{RBC_doc_url}">RBC Product Migration Guide</a>', unsafe_allow_html=True)

sidebar_text = """
### Tech stack:
1) LangChain    - Orchestration
2) Open AI      - embedding and LLM Model
3) Pinecone     - Vector DB
4) Streamlit    - App UI and hosting
"""

st.sidebar.markdown(sidebar_text)

# Define the URLs
linkedin_url = "https://www.linkedin.com/in/akashjoshi/"
github_url = "https://github.com/Ajoshi005"
medium_url = "https://medium.com/p/cdb58657c5c3"

# Add links to LinkedIn, GitHub, and Medium in the sidebar
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <a href="{linkedin_url}" target="_blank">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width=30 height=30>
        </a>
        <a href="{github_url}" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width=30 height=30>
        </a>
        <a href="{medium_url}" target="_blank" style="margin-left: 12px;">Medium Blog</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ###### Disclaimer:
    <div style = 'font-size: xx-small;'>
 - The information provided by this application is for educational purposes only and should not be considered  
    as legal advice.<br> - All answers to the queries are generated by an AI model and may not be entirely accurate or   
    up-to-date. <br> 
    - Please speak to a relationship manager for any specific details relating to your account before making any changes.  
    </div>
    """,
    unsafe_allow_html=True
)
