import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from utils import read_document, display_llm_header, display_response_box

st.set_page_config(layout="wide")

st.markdown("""
<div style='text-align: center;'>
    <h1>Can an LLM judge another LLM?</h1>
</div>
""", unsafe_allow_html=True)

# api keys
load_dotenv(find_dotenv())

# model instantiation
llm1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=1000)
llm2 = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0, max_tokens=1000)
llm3 = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.0, max_tokens=1000)
llm4 = ChatTogether(model_name="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.0, max_tokens=1000)
llm5 = ChatTogether(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", temperature=0.0, max_tokens=1000)
llm6 = ChatTogether(model_name="google/gemma-2-27b-it", temperature=0.0, max_tokens=1000)

llm_respondents = {
    "GPT-3.5 Turbo": llm1, # https://platform.openai.com/docs/models/gpt-3.5-turbo
    "Claude Sonnet 3.5" : llm2, # https://docs.anthropic.com/en/docs/about-claude/models/overview
    "Meta Llama 3.3" : llm3 # https://docs.together.ai/docs/serverless-models#chat-models
}

llm_judges = {
    "Mistral (7B) Instruct" : llm4, # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    "DeepSeek R1 Distill Qwen" : llm5, # https://lambda.ai/inference-models/deepseek-r1?matchtype=p&adgroup=183210364921&feeditemid=&loc_interest_ms=&loc_physical_ms=9032168&network=g&device=c&devicemodel=&adposition=&utm_source=google&utm_campaign=Google_Search_Generic_Inference-Deepseek&utm_medium=search&utm_term=deepseek%20r1&utm_content=752426254270&hsa_acc=1731978716&hsa_cam=22020190693&hsa_grp=183210364921&hsa_ad=752426254270&hsa_src=g&hsa_tgt=kwd-2393956131046&hsa_kw=deepseek%20r1&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=22020190693&gbraid=0AAAAADrJiRw95O8G9CaS090rnEo7lpC9m&gclid=Cj0KCQjwkILEBhDeARIsAL--pjwwjitFny-_4wVEXYogFQ0dUmGuLRAGR4FIMwn2D-WFYv8FYBRbytUaAqC3EALw_wcB
    "Gemma 2 27B" : llm6 # https://huggingface.co/google/gemma-2-27b
}

# high-level layout of 3 sections: respondents, spacer, judge
info_col, col2, col3, col4, col5 = st.columns([2.25, 2, 2, 2, 2])
 
with col2:
    display_llm_header("GPT-3.5 Turbo")
with col3:
    display_llm_header("Claude Sonnet 3.5")
with col4:
    display_llm_header("Meta Llama 3.3")


info_col.markdown(f"## LLM's")

with info_col:
    info_col.markdown("These models are the **respondents**. They will answer questions based on a document you provide.")
    for model in llm_respondents:
        info_col.markdown(f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    ">
        <strong>{model}</strong><br>
    </div>
    """, unsafe_allow_html=True)

info_col.markdown("")

with info_col:
    info_col.markdown("This model is the **judge**. It will evaluate the quality of the responses.")
    llm_judge = st.session_state.get("selected_judge_model", None)
    card_title = llm_judge if llm_judge else "Select a model below"

    info_col.markdown(f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
    ">
        <strong>{card_title}</strong><br>
    </div>
    """, unsafe_allow_html=True)

    st.selectbox(
        label=" ",
        options=list(llm_judges.keys()),
        key="selected_judge_model"
    )

with col5:
    display_llm_header(llm_judge if llm_judge else "", is_judge=True)

# add spacer
info_col.markdown("")
info_col.markdown(f"#### Try it yourself")

# check that respondent models are working
if "resp_response" not in st.session_state:
    st.session_state.resp_response = None
    
mod, dropdown = info_col.columns([1, 2])  # adjust ratio for better layout
all_models = {**llm_respondents, **llm_judges}
selected_model_name = mod.selectbox("**Model:**", list(all_models.keys()), key="selected_respondent_model")
query = dropdown.text_input("**Ask a sample question:**", placeholder="e.g. What is 1+1?", key="respondent_question_input")
if info_col.button(f"Ask {selected_model_name}"):
    output = all_models[selected_model_name].invoke(query).content
    st.session_state.resp_response = output
if st.session_state.get("resp_response"):
    info_col.markdown(f"👾 {st.session_state.resp_response}")
    
# horizontal line
info_col.markdown("---")

# col2.text_input("Test:", key="gpt4_question_input")
# if col2.button(f"Ask {llm_judge}"):
#     output = llm2.invoke(st.session_state.get("gpt4_question_input", "How are you?")).content
#     st.session_state.gpt4_response = output
# if st.session_state.gpt4_response:
#     col2.write(f"🦙 {st.session_state.gpt4_response}")
    
    
    
    
### MAIN FUNCTIONALITY ###

# Respondents

info_col.markdown(f"## Document Q&A")

info_col.file_uploader("**Upload a document:**", type=["txt", "pdf"], key="document_upload")
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = None

# sample_questions = [
#         "What kind of document is this?", # document classification
#         "What is the structure of the document?",
#         "Who is the author and how old might they be?",   # author identification and inference
#         "What is the intended audience?",    # audience identification
#         "What is the purpose of the document?",  # rhetorical intent
#         "What is the tone of the document?", # subjective interpretation
#         "What is the main topic and some key points?",   # central idea identification
#         "How important is this document?",  # ambiguous answer
#     ]

standard_questions = [
    "What kind of document is this?",            # document classification
    "What is the structure of the document?",    # structural analysis
    "Who is the author and how old might they be?"  # author inference
]
info_col.markdown("#### Questions")
for i, q in enumerate(standard_questions, 1):
    info_col.markdown(f"{i}. {q}")
custom_questions = []
for i in range(1, 4):
    user_q = info_col.text_input(f"Write question {i + len(standard_questions)}:", key=f"custom_q_{i}")
    if user_q.strip():
        custom_questions.append(user_q.strip())

all_questions = standard_questions + custom_questions

# system prompt: model should not hallucinate - only answer based on document, format response in a specific way
respondent_system = f"""You are an intelligent assistant that carefully reads documents and answers questions about them.
Only provide answers that are directly supported by the content of the document — do not guess, speculate, or hallucinate.

You must follow this **strict format** for each question:

Question [number]
Answer: [Insert answer text]

Important:
- If a question cannot be answered based on the document, still include it with:  
  Answer: [No answer]
- Do NOT restate the question.
- Do NOT include any extra commentary, headers, or closing remarks.

Example (follow line breaks, spacing, and punctuation exactly):

Question 1  
Answer: The document discusses the impact of climate change on coastal cities.

Question 2  
Answer: [No answer]

Repeat this structure for **all** questions in numerical order.
"""

# human prompt: doc q&a
respondent_human = """Please read the following document and answer the questions below.
    <Document>
    {document}
    </Document>

    Questions:
    {questions}
    """
    
respondent_prompt = ChatPromptTemplate.from_messages([
    ("system", respondent_system),
    ("human", respondent_human),
])

for model_name in llm_respondents:
    key = f"{model_name}_qna_output"
    if key not in st.session_state:
        st.session_state[key] = None

# streamlit UI for document upload
if st.session_state.document_upload:
    st.session_state.file_uploaded = True
    uploaded_file = st.session_state.document_upload

    # read file content once using read_document from utils, cache in session state
    if "document_text" not in st.session_state:
        st.session_state.document_text = read_document(uploaded_file)

    info_col.markdown("")
    info_col.markdown(f"_The respondents will answer these questions based on your document._")

    if info_col.button(f"Get answers"):
        with st.spinner("Thinking..."):
            formatted_prompt = respondent_prompt.format_messages(
                document=st.session_state.document_text,
                questions="\n".join([f"{i+1}. {q}" for i, q in enumerate(all_questions)])
            )

            for model_name, model in llm_respondents.items():
                response = model.invoke(formatted_prompt).content
                st.session_state[f"{model_name}_qna_output"] = response
    
if st.session_state.get("GPT-3.5 Turbo_qna_output"):
    with col2:
        display_response_box(st.session_state["GPT-3.5 Turbo_qna_output"])

# Claude Sonnet 3.5
if st.session_state.get("Claude Sonnet 3.5_qna_output"):
    with col3:
        display_response_box(st.session_state["Claude Sonnet 3.5_qna_output"])

# Meta Llama 3.3
if st.session_state.get("Meta Llama 3.3_qna_output"):
    with col4:
        display_response_box(st.session_state["Meta Llama 3.3_qna_output"])



# Judge

criteria = {
    "Completeness": "Does the response answer all questions?",
    "Relevance": "Are the answers relevant to the questions?",
    "Clarity": "Are the answers clear and easy to understand?",
    "Accuracy": "Are the answers factually correct based on the document?"
}
criteria_string = "\n".join([f"{name}: {definition}" for name, definition in criteria.items()])

# system prompt: llm2 should evaluate llm1's response based on standard criteria, output in a specific format
judge_system = f"""You are an expert evaluator that assesses the quality of multiple LLM's responses to document-based questions.
You will be given:
- The original document
- A list of questions
- The responses from three LLMs (labeled Model 1, Model 2, and Model 3) to each question

Your task is to evaluate the quality of each model's response to **each question** using the following five criteria:
{criteria_string}

For every question, follow this exact output format:

Question [number]
- Model 1: [Aggregate Score out of 5], [Sub-scores for each criterion], [Brief Explanation]
- Model 2: [Aggregate Score out of 5], [Sub-scores for each criterion], [Brief Explanation]
- Model 3: [Aggregate Score out of 5], [Sub-scores for each criterion], [Brief Explanation]

The **sub-scores** must be written in the order of the criteria and enclosed in square brackets, like: [5, 4, 3, 4]

Example output:

Question 1  
- Model 1: 4.5, [5, 4, 4, 5], Clear, complete, and well-supported answers  
- Model 2: 3.2, [4, 3, 3, 3], Relevant but vague answers with minor omissions  
- Model 3: 2.8, [3, 3, 2, 3], Lacks detail and omits key points  

You must follow this structure exactly.  
Do **not** skip any questions or models.  
Do **not** include any commentary outside of this format.  
Do **not** hallucinate information — only use what is in the document and responses.

---

After evaluating all questions, select the best-performing model overall.

Use this exact format:

Best Model: Model [number]  
[One brief sentence explaining your choice]

The chosen model should be the one with the highest and most consistent performance across all questions and criteria.
"""
    
# human prompt: evaluate with doc, questions, and responses as context
judge_human = """Please evaluate the following AI-generated response.

<Document>
{document}
</Document>

<Questions>
{questions}
</Questions>

<Model1Responses>
{model1_responses}
</Model1Responses>

<Model2Responses>
{model2_responses}
</Model2Responses>

<Model3Responses>
{model3_responses}
</Model3Responses>
"""

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", judge_system),
    ("human", judge_human),
])

if st.session_state.get("GPT-3.5 Turbo_qna_output"):    
    info_col.markdown("")
    info_col.markdown("#### Evaluation")
    info_col.markdown(f"_Now, {llm_judge} will evaluate the quality of responses based on the following standard criteria on a scale of 0-5._")
    info_col.markdown("\n".join([f"{i+1}. **{name}**: {definition}" for i, (name, definition) in enumerate(criteria.items())]))

    
    if info_col.button(f"Get evaluation"):
        with st.spinner("Evaluating..."):
            formatted_evaluation_prompt = judge_prompt.format_messages(
                document=st.session_state.document_text,
                questions="\n".join([f"{i+1}. {q}" for i, q in enumerate(all_questions)]),
                model1_responses=st.session_state.get("GPT-3.5 Turbo_qna_output"),
                model2_responses=st.session_state.get("Claude Sonnet 3.5_qna_output"),
                model3_responses=st.session_state.get("Meta Llama 3.3_qna_output")
            )

            judge_response = llm_judges[llm_judge].invoke(formatted_evaluation_prompt)
            st.session_state.judge_eval_output = judge_response.content

if st.session_state.get("judge_eval_output"):
    with col5:
        display_response_box(st.session_state.judge_eval_output)