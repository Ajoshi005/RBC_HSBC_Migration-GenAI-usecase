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
st.set_page_config(page_title="AI CRM Manager",
                   page_icon="🧑‍💼",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )
st.title("Customer Relationship Manager AI Assistant 🤝")

# Define the professional color palette for Streamlit
rbc_canada_palette = {
    "primary": "#F5F5F5",  # light green background
    "secondary": "#E0F2E9",  # light green
    "background": "#0e1117",  # A dark blue color used as the background color of the interface.
    "text": "#fafafa",  # A soft peach color used for text to ensure better readability.
    "accent1": "#555555",  # dark gray
    "accent2": "#000000",  # black
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
# Section to get product type of the query
# Define the options for the dropdown list
product_options = [
    "Personal Chequing and Savings Accounts",
    "Mortgage, Creditor Insurance for Mortgage, Creditor Insurance for Line of Credit",
    "Personal Loans, Lines of Credit and Personal Overdraft ",
    "Investments",
    "Gauranteed Investment Cerificates and Term Deposits",
    "Credit Card",
    "Loyalty"
]

# Create the dropdown list
selected_product = st.selectbox("Select your Product", product_options)

# Get user input from text area based on product selected (prompt selection)
if selected_product:
    user_input = st.text_area(f"Please enter your query on transition of your HSBC personal Banking "
                              f"product {selected_product} to RBC:")
    user_input = ("Regarding "
                  f" {selected_product} : {user_input}")
else:
    user_input = st.text_area(f"Please enter your query on transition of your HSBC Banking accounts to RBC:")

# Add a submit button
submitted = st.button("Submit")


# Check if the user has entered any input
if user_input and submitted:
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
# Define CSS for consistent font styling
st.markdown(
    """
    <style>
    .custom-font {
        font-family: 'Arial', sans-serif;
        font-size: 14px;
    }
    .custom-source {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Response based on below sources from RBC Personal Product Migration Guide:-")
# Display the sources in a separate block
with st.expander("Sources", expanded=False):
    for source in sources:
        st.write("Page:", source.metadata["page"])
        st.markdown(
            f"""
            <div class='custom-source'>
                <p class='custom-font'>
                   {source.page_content}
                </p>
            </div>
            """, unsafe_allow_html=True)
        st.write("")

        # Add a heading to the sidebar
st.sidebar.header("About the App")

# Add a description to the sidebar
st.sidebar.write(
    "Your one stop hub for RBC Transition from HSBC based on the RBC Product Migration Guide"
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
5) Langsmith    - Tracing and testing
"""

st.sidebar.markdown(sidebar_text)

# Define the URLs
linkedin_url = "https://www.linkedin.com/in/akashjoshi/"
github_url = "https://github.com/Ajoshi005"
medium_url = "https://medium.com/@joshiakash89/upgrading-your-rag-game-using-lcel-and-langsmith-89e38781f2cd"

# Add links to LinkedIn, GitHub, and Medium in the sidebar
st.sidebar.markdown(
    f"""
    <a href="{medium_url}" target="_blank" style="margin-bottom: 12px;">Read more about it on my Medium Blog</a>
    <div style="display: flex; align-items: center;">
        <a href="{linkedin_url}" target="_blank">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width=30 height=30>
        </a>
        <a href="{github_url}" target="_blank" style="margin-left: 12px;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width=30 height=30>
        </a>
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
