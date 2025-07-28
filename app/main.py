# main.py
import streamlit as st
import requests
from frontend.utils import title, display_llm_header, display_response_box, display_model_card, read_document
from backend.llms import generate_respondent_prompt, generate_judge_prompt

# set backend url for all api requests
backend_url = "http://localhost:8000"

# API: on startup, backend will instantiate all llm's and load document and question data

# main page setup
st.set_page_config(layout="wide")
title("Can an LLM judge another LLM?")
info_col, col2, col3, col4, col5 = st.columns([2.25, 2, 2, 2, 2])
with col2:
    display_llm_header("GPT-3.5 Turbo")
with col3:
    display_llm_header("Claude Sonnet 3.5")
with col4:
    display_llm_header("Meta Llama 3.3")
    
# API: load configs for respondent llms
if "llm_respondents" not in st.session_state:
    try:
        res = requests.get(f"{backend_url}/models/respondents")
        if res.ok:
            st.session_state.llm_respondents = res.json()
    except Exception as e:
        st.error(f"Error fetching respondent models: {e}")

# API: load configs for judge llms
if "llm_judges" not in st.session_state:
    try:
        res = requests.get(f"{backend_url}/models/judges")
        if res.ok:
            st.session_state.llm_judges = res.json()
    except Exception as e:
        st.error(f"Error fetching judge models: {e}")
    
with info_col:
    st.markdown(f"## LLM's")
    
    # display names of respondent llms
    st.markdown("These models are the **respondents**. They will answer questions based on a document you provide.")
    display_model_card([
        model["model_display_name"] for model in st.session_state.llm_respondents
    ])
    
    st.markdown("")
    
    # display name of judge llm (based on user choice)
    st.markdown("This model is the **judge**. It will evaluate the quality of the responses.")
    llm_judge_name = st.session_state.get("selected_judge_model", None)
    card_title = {llm_judge_name} if llm_judge_name else {"Select a model below"}
    display_model_card(card_title)
    st.selectbox(
        label=" ",
        options= [model["model_display_name"] for model in st.session_state.llm_judges],
        key="selected_judge_model"
    )
    
# update judge column header
with col5:
    display_llm_header(llm_judge_name if llm_judge_name else "", is_judge=True)
        
# test out any of the models
with info_col:
    st.markdown("")
    st.markdown(f"#### Try it yourself")
    
    mod, dropdown = st.columns([1, 2])
    all_models = {
        m["model_display_name"]: m["model_id"]
        for m in st.session_state.llm_respondents + st.session_state.llm_judges
    }
    selected_model_name = mod.selectbox("**Model:**", list(all_models.keys()), key="selected_respondent_model")
    query = dropdown.text_input("**Ask a sample question:**", placeholder="e.g. What is 1+1?", key="respondent_question_input")
    
    # API: invoke the chosen model
    if "sample_response" not in st.session_state:
        st.session_state.sample_response = None
    if st.button(f"Ask {selected_model_name}"):
        model_id = all_models[selected_model_name]
        response = requests.post(
            f"{backend_url}/respond",
            json={"model_id": model_id, "prompt": query}
        )
        if response.ok:
            st.session_state.sample_response = response.json()["output"]
        else:
            st.error("Model failed to respond.")
    if st.session_state.get("sample_response"):
        info_col.markdown(f"👾 {st.session_state.sample_response}")
        
info_col.markdown("---")

with info_col:
    st.markdown(f"## Document Q&A")
    
    # API: get all documents data from api in this format
    # class Document(BaseModel):
    #     doc_id: int
    #     doc_name: str
    #     doc_type: str
    #     doc_path: str
    #     doc_txt: str = ""
    if st.button("Load documents from API"):
        try:
            response = requests.get(f"{backend_url}/documents")
            if response.status_code == 200:
                st.session_state.documents = response.json()
        except Exception as e:
            st.error(f"Error: {e}")
    if "documents" in st.session_state:
        doc_options = {
            f"{doc['doc_name']}": doc["doc_id"]
            for doc in st.session_state.documents
        }
        
        # user selects a document from api
        selected_name = st.selectbox("Select a document:", list(doc_options.keys()))
        st.session_state.file_uploaded_api = doc_options[selected_name]  # store selection
        st.session_state.file_uploaded_st = None                         # reset ST upload

        # load text of user chosen doc into document_text
        response = requests.get(f"{backend_url}/document-text/{st.session_state.file_uploaded_api}")
        if response.ok:
            doc_data = response.json()
            st.session_state.document_text = doc_data["doc_txt"]
        else:
            st.error(response.json().get("detail", "Unknown error occurred"))
        
    # or the user can upload a document regularly
    uploaded_file = st.file_uploader("**OR:**", type=["txt", "pdf"], key="document_upload")
    if uploaded_file:
        st.session_state.file_uploaded_st = uploaded_file  # store upload
        st.session_state.file_uploaded_api = None          # reset API selection
        st.session_state.document_text = read_document(uploaded_file)

    # some standard questions
    st.markdown("#### Questions")
    
    standard_questions = [
        "What kind of document is this?",
        "What is the structure of the document?",
        "Who is the author and how old might they be?"
    ]
    for i, q in enumerate(standard_questions, 1):
        st.markdown(f"{i}. {q}")
        
    # user may choose to load suggested questions from api that are doc_type specific
    if "loaded_api_questions" not in st.session_state:
        st.session_state.loaded_api_questions = []
    if st.session_state.get("file_uploaded_api") is not None:
        if st.button("Load Questions from API"):
            try:
                response = requests.get(f"{backend_url}/question-set/{st.session_state.file_uploaded_api}")
                if response.ok:
                    question_data = response.json()
                    st.session_state.loaded_api_questions = question_data["questions"]
                else:
                    st.error("Could not load questions from API.")
            except Exception as e:
                st.error(f"Error: {e}")
    if st.session_state.loaded_api_questions:
        for i, q in enumerate(st.session_state.loaded_api_questions, len(standard_questions) + 1):
            st.markdown(f"{i}. {q}")
            
    # user can also add their own questions
    custom_questions = []
    for i in range(1, 4):
        user_q = st.text_input(f"Write a custom question:", key=f"custom_q_{i}")
        if user_q.strip():
            custom_questions.append(user_q.strip())
            
    all_questions = (standard_questions + st.session_state.loaded_api_questions + custom_questions)

    ###
    if "document_text" in st.session_state and st.session_state.llm_respondents:
        st.markdown("")
        st.markdown(f"_The respondents will answer these questions based on your document._")
        
        for model in st.session_state.llm_respondents:
            key = f"{model['model_id']}_qna_output"
            if key not in st.session_state:
                st.session_state[key] = None
                
        if info_col.button(f"Get answers"):
            with st.spinner("Thinking..."):
                respondent_prompt = generate_respondent_prompt()
                formatted_prompt = respondent_prompt.format(
                    document=st.session_state.document_text,
                    questions="\n".join([f"{i+1}. {q}" for i, q in enumerate(all_questions)])
                )
                for model in st.session_state.llm_respondents:
                    model_id = model["model_id"]
                    try:
                        response = requests.post(
                            f"{backend_url}/respond",
                            json={"model_id": model_id, "prompt": formatted_prompt}
                        )
                        if response.ok:
                            st.session_state[f"{model_id}_qna_output"] = response.json()["output"]
                        else:
                            st.session_state[f"{model_id}_qna_output"] = f"❌ Error: {response.text}"
                    except Exception as e:
                        st.session_state[f"{model_id}_qna_output"] = f"❌ Exception: {e}"

# print respondent's answers
for i, model in enumerate(st.session_state.llm_respondents):
    output = st.session_state.get(f"{model["model_id"]}_qna_output")
    if output:
        with [col2, col3, col4][i]:
            display_response_box(output)
        
# standard criteria
criteria = {
    "Completeness": "Does the response answer all questions?",
    "Relevance": "Are the answers relevant to the questions?",
    "Clarity": "Are the answers clear and easy to understand?",
    "Accuracy": "Are the answers factually correct based on the document?"
}
        
criteria_string = "\n".join([f"{name}: {definition}" for name, definition in criteria.items()])
judge_prompt = generate_judge_prompt(criteria_string)

with info_col:
    if (
        st.session_state.get("0_qna_output") and
        st.session_state.get("1_qna_output") and
        st.session_state.get("2_qna_output")
    ):
        
        if "judge_eval_output" not in st.session_state:
                st.session_state.judge_eval_output = None
        
        st.markdown("")
        st.markdown("#### Evaluation")
        st.markdown(f"_Now, {llm_judge_name} will evaluate the quality of responses based on the following standard criteria on a scale of 0-5._")
        st.markdown("\n".join([f"{i+1}. **{name}**: {definition}" for i, (name, definition) in enumerate(criteria.items())]))

        
        if st.button(f"Get evaluation"):
            with st.spinner("Evaluating..."):
                formatted_evaluation_prompt = judge_prompt.format(
                    document=st.session_state.document_text,
                    questions="\n".join([f"{i+1}. {q}" for i, q in enumerate(all_questions)]),
                    model1_responses=st.session_state.get("GPT-3.5 Turbo_qna_output"),
                    model2_responses=st.session_state.get("Claude Sonnet 3.5_qna_output"),
                    model3_responses=st.session_state.get("Meta Llama 3.3_qna_output")
                )
                llm_judge_id = all_models[llm_judge_name]
                try:
                    response = requests.post(
                        f"{backend_url}/respond",
                        json={"model_id": llm_judge_id, "prompt": formatted_evaluation_prompt}
                    )
                    if response.ok:
                        st.session_state.judge_eval_output = response.json()["output"]
                    else:
                        st.session_state.judge_eval_output = f"❌ Error: {response.text}"
                except Exception as e:
                    st.session_state.judge_eval_output = f"❌ Exception: {e}"
                
if st.session_state.get("judge_eval_output"):
    with col5:
        display_response_box(st.session_state.judge_eval_output)