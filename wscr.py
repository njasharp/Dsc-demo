import streamlit as st
import requests
from bs4 import BeautifulSoup
import io
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image
from groq import Groq
from gtts import gTTS
import os

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

# Initialize the Groq client with the API key from environment variable
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Define supported models
SUPPORTED_MODELS = {
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it"
}

# Retry mechanism for Groq API
def query_groq_with_retry(messages, model, retries=3):
    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                continue  # Retry if possible
            else:
                st.error(f"An error occurred after {retries} attempts: {e}")
                return None

# Function to stream responses from Groq
def groq_stream_responses(messages):
    for i in range(5):  # Simulate streaming (replace with actual API call when possible)
        yield f"Generated response part {i + 1}"

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

# CoT (Chain of Thought) prompt generator
def cot_prompt_generator(prompt):
    cot_template = f"""
    1. Understand the Problem: Carefully read and understand the user's question or request.
    2. Break Down the Reasoning Process: Outline the steps required to solve the problem or respond to the request logically and sequentially. Think aloud and describe each step in detail.
    3. Explain Each Step: Provide reasoning or calculations for each step, explaining how you arrive at each part of your answer.
    4. Arrive at the Final Answer: Only after completing all steps, provide the final answer or solution.
    5. Review the Thought Process: Double-check the reasoning for errors or gaps before finalizing your response.
    The user request is: {prompt}
    """
    return cot_template

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

# Set default models
default_model = list(SUPPORTED_MODELS.keys())[0]  # First model as default

if "model_1" not in st.session_state:
    st.session_state["model_1"] = default_model
if "model_2" not in st.session_state:
    st.session_state["model_2"] = default_model
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

# Sidebar spinner placeholder
sidebar_placeholder = st.sidebar.empty()

st.sidebar.write("SmartSuggest Settings")

# Sidebar menu
with st.sidebar:
    st.session_state["model_1"] = st.selectbox(
        "Select LLM model for the First Step (Model 1)",
        list(SUPPORTED_MODELS.keys()),
        index=list(SUPPORTED_MODELS.keys()).index(st.session_state["model_1"]),
        key="model_1_select"
    )

    st.session_state["model_2"] = st.selectbox(
        "Select LLM model for Final Step Feedback (Model 2)",
        list(SUPPORTED_MODELS.keys()),
        index=list(SUPPORTED_MODELS.keys()).index(st.session_state["model_2"]),
        key="model_2_select"
    )

    # **Added "CoT Prompt" to the prompt options**
    prompt_option = st.radio("Select Prompt Option:", (
        "SmartSuggest Prompt", 
        "Meta Prompt", 
        "Advanced Prompt", 
        "Reflection Prompt",
        "CoT Prompt",  # New Option Added
        "Custom Prompt"
    ))

    # Option to use both original and improved prompts
    st.session_state["use_both_prompts"] = st.checkbox(
        "Use both original and improved prompts for final output", 
        value=False
    )

    # **Define the CoT Prompt**
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
    elif prompt_option == "CoT Prompt":
        default_prompt = cot_prompt_generator("Enter your problem or request here.")
    else:
        default_prompt = "Enter your custom prompt here."

    st.session_state["system_prompt"] = st.text_area(
        "System Prompt", 
        value=default_prompt, 
        height=200
    )

    st.session_state["uploaded_image"] = st.file_uploader(
        "Upload an image (optional):", 
        type=["png", "jpg", "jpeg"]
    )

    st.session_state["show_steps"] = st.checkbox(
        "Show steps and results for each phase", 
        value=True
    )

    url = st.text_input("Enter the website URL to scrape")
    if st.button("Scrape Website"):
        st.session_state["scraped_content"] = scrape_website(url)
        st.write("Scraped content:")
        st.text_area("Scraped Content", st.session_state["scraped_content"], height=300)

    uploaded_file = st.file_uploader(
        "Upload a PDF, CSV, or TXT file (optional):", 
        type=["pdf", "csv", "txt"]
    )
    if uploaded_file is not None:
        st.session_state["uploaded_file_content"] = read_uploaded_file(uploaded_file)
        st.write("Uploaded file content:")
        st.text_area("File Content", st.session_state["uploaded_file_content"], height=300)

# Display all previous and current messages
for message in st.session_state["messages"]:
    st.markdown(message["content"])

# Add space before the query box
st.write("")

if st.session_state["new_message"]:
    st.session_state["user_query"] = ""
    st.session_state["new_message"] = False
    st.rerun()

if prompt := st.text_input("What is your query?", key="user_query"):
    original_prompt = prompt
    if prompt_option == "SmartSuggest Prompt":
        if st.session_state["show_steps"]:
            st.subheader("Improved Prompt")
        improvement_prompt = f"Improve the following prompt for better analysis and insights:\n\n{prompt}"
        improved_prompt = query_groq_with_retry([{"role": "user", "content": improvement_prompt}], SUPPORTED_MODELS[st.session_state["model_1"]])

        if st.session_state["show_steps"]:
            st.markdown(improved_prompt)
            with sidebar_placeholder:
                with st.spinner("Processing..."):
                    pass

        st.session_state["messages"].append({"role": "assistant", "content": f"**Improved Prompt:** {improved_prompt}"})
        prompt = improved_prompt

    if st.session_state["use_both_prompts"]:
        final_prompt = f"{original_prompt}\n\n{prompt}"
    else:
        final_prompt = prompt

    augmented_prompt = final_prompt + "\n\n" + st.session_state["scraped_content"] + "\n\n" + st.session_state["uploaded_file_content"]
    if st.session_state["uploaded_image"]:
        image = Image.open(st.session_state["uploaded_image"])
        augmented_prompt += f"\n\n[Attached Image: {st.session_state['uploaded_image'].name}]"

    st.session_state["messages"].append({"role": "user", "content": final_prompt})

    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        if st.session_state["show_steps"]:
            st.subheader("Generating the Initial Response")
            try:
                response = query_groq_with_retry([{"role": "user", "content": augmented_prompt}], SUPPORTED_MODELS[st.session_state["model_1"]])
                st.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
                response = ""

        if st.session_state["show_steps"]:
            st.subheader("Final Query and Analysis")
            final_prompt_analysis = f"Final analysis and query based on the improved and evaluated response:\n\n{final_prompt}\n\n"
            final_prompt_analysis += st.session_state["scraped_content"] + "\n\n" + st.session_state["uploaded_file_content"]
            if st.session_state["uploaded_image"]:
                final_prompt_analysis += f"\n\n[Attached Image: {st.session_state['uploaded_image'].name}]"

            try:
                final_response = query_groq_with_retry([{"role": "user", "content": final_prompt_analysis}], SUPPORTED_MODELS[st.session_state["model_2"]])
                st.markdown(final_response)
                st.session_state["messages"].append({"content": final_response})
            except Exception as e:
                st.error(f"Failed to generate final response: {e}")

            # Generate audio from the final response and make it playable
            tts = gTTS(text=final_response, lang='en')
            audio_path = "response.mp3"
            tts.save(audio_path)
            st.audio(audio_path)

        st.session_state["new_message"] = True
        st.rerun()

# Sidebar reset functionality
if st.sidebar.button("Reset"):
    st.session_state["messages"] = []
    st.session_state["new_message"] = False
    st.session_state["user_query"] = ""
    st.session_state["scraped_content"] = ""
    st.session_state["uploaded_image"] = None
    st.session_state["uploaded_file_content"] = ""
    st.sidebar.info("Screen reset. All progress has been cleared.")
    st.rerun()

# Display footer or final message
st.sidebar.info("Built by DW 9-16-24")