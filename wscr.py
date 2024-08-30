import ollama
import streamlit as st
import requests
from bs4 import BeautifulSoup
import io
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image

# Streamlit page configuration
st.set_page_config(layout="wide")

# Apply custom CSS for a dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e0e0e;
        color: #ffffff;
    }
    .stTextInput, .stTextArea, .stSelectbox {
        background-color: #262626;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to scrape the website
def scrape_website(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the body content and clean it
        body_content = soup.body.get_text(separator="\n", strip=True)
        return body_content
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to scrape the website: {e}")
        return ""

def model_res_generator(messages):
    try:
        stream = ollama.chat(
            model=st.session_state["model"],
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]
    except (httpx.ConnectError, ollama.ResponseError) as e:
        st.error(f"An error occurred: {e}")
        return

# Streamlit UI setup
st.image("p1.png", width=420)
st.title("DataScrape ai")
st.text("Scraping web content and analyzing it using a Large Language Model")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "model" not in st.session_state:
    st.session_state["model"] = ""
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = ""
if "new_message" not in st.session_state:
    st.session_state["new_message"] = False
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "scraped_content" not in st.session_state:
    st.session_state["scraped_content"] = ""

st.sidebar.write("DataScrape ai Settings")

# Sidebar menu
with st.sidebar:
    try:
        models = [model["name"] for model in ollama.list()["models"]]
        st.session_state["model"] = st.selectbox("Choose your model", models)
    except requests.exceptions.RequestException:
        st.error("Unable to connect to model API")
    
    prompt_option = st.radio("Select Prompt Option:", ("Basic Prompt", "Advanced Prompt"))
    
    if prompt_option == "Basic Prompt":
        default_prompt = "Analyze and summarize the content."
    else:
        default_prompt = (
            "Role: Act as an expert content analyst specializing in web data.\n"
            "Context: You have scraped content from a webpage that includes various types of data (e.g., articles, blog posts, product descriptions).\n"
            "Task: Analyze the scraped content, focusing on key themes, sentiment, and insights. Summarize the findings in a concise manner, highlighting the most relevant points.\n"
            "Constraints: Keep the analysis focused on actionable insights that can inform business decisions. Use clear, non-technical language suitable for a general audience.\n"
            "Additional Guidance: If the content includes multiple topics, prioritize those most relevant to [specific goal or industry]. Provide the summary in bullet points, and if applicable, suggest potential applications of the findings."
        )
    
    st.session_state["system_prompt"] = st.text_area("System Prompt", value=default_prompt, height=200)
    
    if st.button("Reset"):
        st.session_state["messages"] = []
        st.session_state["new_message"] = False
        st.session_state["user_query"] = ""
        st.session_state["scraped_content"] = ""
        st.rerun()

    url = st.text_input("Enter the website URL to scrape")
    if st.button("Scrape Website"):
        st.session_state["scraped_content"] = scrape_website(url)
        st.write("Scraped content:")
        st.text_area("Scraped Content", st.session_state["scraped_content"], height=300)

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add some space before the query box
st.write("")
st.write("")

if st.session_state["new_message"]:
    st.session_state["user_query"] = ""
    st.session_state["new_message"] = False
    st.rerun()

if prompt := st.text_input("What is your query?", key="user_query"):
    # Include scraped content in the query
    augmented_prompt = prompt + "\n\n" + st.session_state["scraped_content"]

    # Prepare messages
    messages = [{"role": "user", "content": augmented_prompt}]
    if st.session_state["system_prompt"]:
        messages.insert(0, {"role": "system", "content": st.session_state["system_prompt"]})
    
    # Add the latest message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing the content..."):
            # Generate response based on the augmented prompt
            try:
                message = "".join(model_res_generator(messages))
                st.session_state["messages"].append({"role": "assistant", "content": message})
            except Exception as e:
                st.error(f"Failed to generate response: {e}")

    # Set flag for new message
    st.session_state["new_message"] = True
    st.rerun()

st.sidebar.info("Built by DW 8-30-24")