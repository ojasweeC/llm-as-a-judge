import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile

def title(app_name):
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1>{app_name}</h1>
        </div>
        """, unsafe_allow_html=True)

def display_llm_header(model_name, is_judge=False):
    role = "Evaluator" if is_judge else "Respondent"
    display_name = model_name if model_name else "Choose your own LLM"
    st.markdown(f"""
    <div style="text-align: center; font-size: 20px; font-weight: 600; margin-top: 10px">{display_name}</div>
    <div style="text-align: center; font-size: 13px; color: #777;">{role}</div>
    """, unsafe_allow_html=True)
    
def display_response_box(response_text):
    st.markdown("")
    response_text = response_text.replace("\n", "<br>")
    styled_html = f"""
    <div style="
        border: 1px solid #5b8db8;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.80rem;
        color: #666;
        line-height: 1.6;
    ">
        <pre style="white-space: pre-wrap; margin: 0;">{response_text}</pre>
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)
    
def display_model_card(model_names, title=None):
    for m in model_names:
        styled_html = f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
        ">
            <strong>{m}</strong><br>
        </div>
        """
        st.markdown(styled_html, unsafe_allow_html=True)
        
def read_document(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.flush()
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load()
            return "\n".join([p.page_content for p in pages])

    return "Unsupported file type."
    