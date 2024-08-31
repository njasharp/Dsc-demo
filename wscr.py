import streamlit as st
import requests
from bs4 import BeautifulSoup
import io
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image
import os
from groq import Groq

# Streamlit page configuration
st.set_page_config(layout="wide")

# Apply custom CSS for a dark theme, including the sidebar
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e0e0e;
        color: #ffffff;
    }
    .css-18e3th9 {
        background-color: #0e0e0e;  /* Sidebar background color */
    }
    .css-1d391kg {
        background-color: #0e0e0e;  /* Sidebar selectbox background color */
    }
    .css-1v0mbdj {
        background-color: #262626;  /* Dropdown background color */
        color: #ffffff;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stCheckbox {
        background-color: #262626;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stTextInput > div > input {
        background-color: #262626;
        color: #ffffff;
    }
    .stTextArea > div > textarea {
        background-color: #262626;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to fetch available models from Groq
def fetch_groq_models():
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        models_info = client.models.list()  # Fetching models info from Groq API

        models = {}
        for model in models_info.data:  # Iterate over the data attribute
            model_id = model.id
            description = f"Owned by {model.owned_by}, Context Window: {model.context_window}"
            models[model_id] = description
            
        return models
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return {}

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
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=st.session_state["model"],
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return ""

# Function to read the uploaded file content
def read_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string()
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
    return ""

# Streamlit UI setup
st.image("p1.png", width=420)
st.title("SmartSuggest Prompt")
st.text("SmartSuggest Prompt content and analyzing it using a Large Language Model")

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
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None
if "uploaded_file_content" not in st.session_state:
    st.session_state["uploaded_file_content"] = ""
if "show_steps" not in st.session_state:
    st.session_state["show_steps"] = True
if "use_both_prompts" not in st.session_state:
    st.session_state["use_both_prompts"] = False

# Sidebar setup
with st.sidebar:
    st.write("SmartSuggest Settings")

    # Fetch and select model
    models = fetch_groq_models()
    if models:
        st.session_state["model"] = st.selectbox(
            "Choose your model",
            options=list(models.keys()),
            format_func=lambda x: f"{x} - {models[x]}"
        )
    else:
        st.error("No models available. Check your API key and connection.")

    # Select Prompt Option
    prompt_option = st.radio("Select Prompt Option:", ("SmartSuggest Prompt", "Meta Prompt", "Advanced Prompt", "Custom Prompt"))

    # Option to use both original and improved prompts
    st.session_state["use_both_prompts"] = st.checkbox("Use both original and improved prompts for final output", value=False)

    if prompt_option == "SmartSuggest Prompt":
        default_prompt = "Provide your prompt for improvement."
    elif prompt_option == "Meta Prompt":
        default_prompt = "Create a meta-prompt that sets up a structured approach for analyzing and summarizing the content."
    elif prompt_option == "Advanced Prompt":
        default_prompt = (
            "Role: Act as an expert content analyst specializing in web data.\n"
            "Context: You have scraped content from a webpage that includes various types of data (e.g., articles, blog posts, product descriptions).\n"
            "Task: Analyze the scraped content, focusing on key themes, sentiment, and insights. Summarize the findings in a concise manner, highlighting the most relevant points.\n"
            "Constraints: Keep the analysis focused on actionable insights that can inform business decisions. Use clear, non-technical language suitable for a general audience.\n"
            "Additional Guidance: If the content includes multiple topics, prioritize those most relevant to [specific goal or industry]. Provide the summary in bullet points, and if applicable, suggest potential applications of the findings."
        )
    else:
        default_prompt = "Enter your custom prompt here."

    st.session_state["system_prompt"] = st.text_area("System Prompt", value=default_prompt, height=200)

    # Upload an image
    st.session_state["uploaded_image"] = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])

    # Toggle to show/hide steps
    st.session_state["show_steps"] = st.checkbox("Show steps and results for each phase", value=True)

    if st.button("Reset"):
        st.session_state["messages"] = []
        st.session_state["new_message"] = False
        st.session_state["user_query"] = ""
        st.session_state["scraped_content"] = ""
        st.session_state["uploaded_image"] = None
        st.session_state["uploaded_file_content"] = ""
        st.rerun()

    url = st.text_input("Enter the website URL to scrape")
    if st.button("Scrape Website"):
        st.session_state["scraped_content"] = scrape_website(url)
        st.write("Scraped content:")
        st.text_area("Scraped Content", st.session_state["scraped_content"], height=300)

    # Upload a PDF, CSV, or TXT file
    uploaded_file = st.file_uploader("Upload a PDF, CSV, or TXT file (optional):", type=["pdf", "csv", "txt"])
    if uploaded_file is not None:
        st.session_state["uploaded_file_content"] = read_uploaded_file(uploaded_file)
        st.write("Uploaded file content:")
        st.text_area("File Content", st.session_state["uploaded_file_content"], height=300)

# Main section for displaying the query input and messages
st.subheader("Enter your query and view results")

st.session_state["user_query"] = st.text_input("What is your query?", key="query_input")
if st.session_state["user_query"]:
    original_prompt = st.session_state["user_query"]
    if prompt_option == "SmartSuggest Prompt":
        st.markdown("Improving the prompt...")
        improvement_prompt = f"Improve the following prompt for better analysis and insights:\n\n{original_prompt}"
        improvement_messages = [{"role": "user", "content": improvement_prompt}]
        improved_prompt = model_res_generator(improvement_messages)
        st.session_state["messages"].append({"role": "assistant", "content": f"**Improved Prompt:** {improved_prompt}"})
        st.markdown(improved_prompt)

if st.session_state["user_query"]:
    st.write("Analyzing the content...")

    # Combine original and improved prompts if the option is selected
    if st.session_state["use_both_prompts"]:
        final_prompt = f"{original_prompt}\n\n{improved_prompt}"
    else:
        final_prompt = improved_prompt

    # Include scraped content, uploaded image, and uploaded file content in the query
    augmented_prompt = final_prompt + "\n\n" + st.session_state["scraped_content"] + "\n\n" + st.session_state["uploaded_file_content"]
    if st.session_state["uploaded_image"]:
        image = Image.open(st.session_state["uploaded_image"])
        augmented_prompt += f"\n\n[Attached Image: {st.session_state['uploaded_image'].name}]"

    # Prepare messages
    messages = [{"role": "user", "content": augmented_prompt}]
    if st.session_state["system_prompt"]:
        messages.insert(0, {"role": "system", "content": st.session_state["system_prompt"]})

    # Step 1: Generate initial response
    st.write("Generating the initial response...")
    try:
        response = model_res_generator(messages)
        st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        response = ""

    # Step 2: Evaluate the response
    if st.session_state["show_steps"]:
        st.subheader("Evaluating the Response")
        evaluation_prompt = f"Evaluate the following response and check if it is good enough:\n\n{response}"
        evaluation_messages = [{"role": "user", "content": evaluation_prompt}]
        try:
            evaluation_response = model_res_generator(evaluation_messages)
            st.markdown(evaluation_response)
            st.session_state["messages"].append({"content": evaluation_response})
        except Exception as e:
            st.error(f"Failed to evaluate response: {e}")

    # Step 3: Grade and provide feedback
    if st.session_state["show_steps"]:
        st.subheader("Grading the Response and Providing Feedback")
        feedback_prompt = f"Grade the quality of this response and provide feedback:\n\n{response}\n\nEvaluation: {evaluation_response}"
        feedback_messages = [{"role": "user", "content": feedback_prompt}]
        try:
            feedback_response = model_res_generator(feedback_messages)
            st.markdown(feedback_response)
            st.session_state["messages"].append({"content": feedback_response})
        except Exception as e:
            st.error(f"Failed to generate feedback: {e}")

    # Final Step: Apply final prompt to the LLM including any additional content
    st.subheader("Final Query and Analysis")
    final_prompt_analysis = f"Final analysis and query based on the improved and evaluated response:\n\n{final_prompt}\n\n"
    final_prompt_analysis += st.session_state["scraped_content"] + "\n\n" + st.session_state["uploaded_file_content"]
    if st.session_state["uploaded_image"]:
        final_prompt_analysis += f"\n\n[Attached Image: {st.session_state['uploaded_image'].name}]"

    final_messages = [{"role": "user", "content": final_prompt_analysis}]
    try:
        final_response = model_res_generator(final_messages)
        st.markdown(final_response)
        st.session_state["messages"].append({"content": final_response})
    except Exception as e:
        st.error(f"Failed to generate final response: {e}")

    if st.session_state["show_steps"]:
        st.write("**Final Query and Analysis:**")
        st.markdown(final_response)

# Sidebar info or footer
st.sidebar.info("Built by DW 8-30-24")
