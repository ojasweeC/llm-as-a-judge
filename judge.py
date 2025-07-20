import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from utils import read_document

llm1_model = "GPT-4"
llm2_model = "Llama-3"

col1, spacer, col2 = st.columns([1, 0.25, 1])
col1.write(f"## {llm1_model}")
col1.write(f"{llm1_model} will take a document and 10 standard questions, then return a response.")
col2.write(f"## {llm2_model}")
col2.write(f"{llm2_model} will evaluate the quality of {llm1_model}'s response based on standard criteria.")

load_dotenv(find_dotenv())

llm1 = ChatOpenAI(model_name="gpt-4")
llm2 = ChatOllama(model="llama3:instruct")

# initialize session state
if "llama3_response" not in st.session_state:
    st.session_state.llama3_response = None
if "gpt4_response" not in st.session_state:
    st.session_state.gpt4_response = None

# test that models are working
col1.text_input("Test:", key="llama3_question_input")
if col1.button(f"Ask {llm1_model}"):
    output = llm1.invoke(st.session_state.get("llama3_question_input", "How are you?")).content
    st.session_state.llama3_response = output
if st.session_state.llama3_response:
    col1.write(f"👾 {st.session_state.llama3_response}")

col2.text_input("Test:", key="gpt4_question_input")
if col2.button(f"Ask {llm2_model}"):
    output = llm2.invoke(st.session_state.get("gpt4_question_input", "How are you?")).content
    st.session_state.gpt4_response = output
if st.session_state.gpt4_response:
    col2.write(f"🦙 {st.session_state.gpt4_response}")
    
# main functionality

# LLM1 document Q&A

questions = [
        "What kind of document is this?", # document classification
        "What is the structure of the document?",
        "Who is the author and how old might they be?",   # author identification and inference
        "What is the intended audience?",    # audience identification
        "What is the purpose of the document?",  # rhetorical intent
        "What is the tone of the document?", # subjective interpretation
        "What is the main topic and some key points?",   # central idea identification
        "How important is this document?",  # ambiguous answer
    ]

# prompt: give a response format, tell model not to answer questions it doesn't know
llm1_system = f"""You are an intelligent assistant that carefully reads documents and answers questions about them.
Only provide answers based on the information found in the document — do not guess or hallucinate.
If you do not know the answer to a question, do not fabricate an answer. Just skip it.

Format your response using the following structure for each question, **with line breaks between each**:

Question: [Question text]  
Answer: [Answer text]

Example format:

Question: What is the main topic of the document?  
Answer: The document discusses the impact of climate change on coastal cities.

Question: Who is the author of the document?  
Answer: [No answer]  ← Only include if the document has no clear author info (optional: skip entirely)

Repeat this structure for each question.
"""

llm1_human = """Please read the following document and answer the questions below.
    <Document>
    {document}
    </Document>

    Questions:
    {questions}
    """
    
llm1_prompt = ChatPromptTemplate.from_messages([
    ("system", llm1_system),
    ("human", llm1_human),
])

col1.file_uploader("Upload a document:", type=["txt", "pdf"], key="document_upload")
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = None

if st.session_state.document_upload:
    st.session_state.file_uploaded = True
    uploaded_file = st.session_state.document_upload

    # read file content once and cache in session state
    if "document_text" not in st.session_state:
        st.session_state.document_text = read_document(uploaded_file)

    col1.markdown(f"**{llm1_model} will answer the following questions based on your document:**")
    col1.markdown("\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]))

    if col1.button(f"Run {llm1_model}"):
        with st.spinner("Thinking..."):
            formatted_prompt = llm1_prompt.format_messages(
                document=st.session_state.document_text,
                questions="\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            )

            llm1_response = llm1.invoke(formatted_prompt)
            st.session_state.llm1_qna_output = llm1_response.content
    
# LLM2 evaluation
criteria = [
    "Completeness: Does the response answer all questions?",
    "Relevance: Are the answers relevant to the questions?",
    "Clarity: Are the answers clear and easy to understand?",
    "Accuracy: Are the answers factually correct based on the document?"
]

llm2_system = f"""You are an expert evaluator that assesses the quality of responses from {llm1_model}.
You will be given:
- The original document
- A list of standard questions
- The model's responses to those questions (formatted in XML)

Your task is to evaluate the quality of the responses based on the following five criteria:
{criteria}

For each criterion, output exactly this format (with line breaks between lines):

[Criterion Name]
Explanation: [Your explanation here]
Score: [Numeric score from 1 to 5]

Make sure there is a line break after each of the three lines.
Example:

Correctness  
Explanation: The answers match the facts in the document.  
Score: 5  

Only base your evaluation on the document and questions — do not guess or hallucinate.
"""
    
llm2_human = """Please evaluate the following AI-generated response.

<Document>
{document}
</Document>

<Questions>
{questions}
</Questions>

<AIResponses>
{responses}
</AIResponses>
"""

llm2_prompt = ChatPromptTemplate.from_messages([
    ("system", llm2_system),
    ("human", llm2_human),
])

if st.session_state.get("llm1_qna_output"):
    col1.write(f"👾 {st.session_state.llm1_qna_output}")
    
    col2.markdown(f"**Now, {llm2_model} will evaluate the quality of {llm1_model}'s response based on the following standard criteria.**")
    col2.markdown("\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)]))
    
    if col2.button(f"Run {llm2_model}"):
        with st.spinner("Evaluating..."):
            formatted_evaluation_prompt = llm2_prompt.format_messages(
                document=st.session_state.document_text,
                questions="\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]),
                responses=st.session_state.llm1_qna_output
            )

            llm2_response = llm2.invoke(formatted_evaluation_prompt)
            st.session_state.llm2_evaluation_output = llm2_response.content

if st.session_state.get("llm2_evaluation_output"):
    col2.write(f"🦙 {st.session_state.llm2_evaluation_output}")