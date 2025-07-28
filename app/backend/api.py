import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import Document, QuestionSet, LLMConfig, ModelCall
from .llms import instantiate_llms, generate_respondent_prompt, generate_judge_prompt
from langchain.document_loaders import PyPDFLoader

# this is a stub for the API backend
app = FastAPI()

DOCUMENTS = []
QUESTION_SETS = []
LLM_CONFIGS = []
LLM_INSTANCES = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def go():
    return {"message": "API is running"}

@app.on_event("startup")
def load_models():
    global LLM_CONFIGS, LLM_INSTANCES
    LLM_CONFIGS, LLM_INSTANCES = instantiate_llms()

@app.on_event("startup")
def load_data():
    global DOCUMENTS, QUESTION_SETS

    # Load documents
    with open("/Users/ojaswee-chaudhary/Documents/stealth/llm_as_a_judge/app/backend/data/documents.json", "r") as file:
        data = json.load(file)
    DOCUMENTS = [
        Document(
            doc_id=i,
            doc_name=item["doc_name"],
            doc_type=item["doc_type"],
            doc_path=item["doc_path"],
            doc_txt=""
        )
        for i, item in enumerate(data)
    ]
    print(f"✅ Loaded {len(DOCUMENTS)} documents from ./data/documents.json")

    with open("/Users/ojaswee-chaudhary/Documents/stealth/llm_as_a_judge/app/backend/data/questions.json", "r") as file:
        qs_data = json.load(file)
    QUESTION_SETS = [
        QuestionSet(
            doc_type=item["doc_type"],
            questions=item["questions"]
        )
        for item in qs_data
    ]

@app.get("/models", response_model=list[LLMConfig])
def get_models():
    return LLM_CONFIGS

@app.get("/models/respondents", response_model=list[LLMConfig])
def get_respondent_models():
    return [llm for llm in LLM_CONFIGS if not llm.is_judge]

@app.get("/models/judges", response_model=list[LLMConfig])
def get_judge_models():
    return [llm for llm in LLM_CONFIGS if llm.is_judge]

@app.get("/documents", response_model=list[Document])
def get_documents():
    return DOCUMENTS
        
@app.get("/document-text/{doc_id}", response_model=Document)
def get_document_text(doc_id: int):
    doc = next((d for d in DOCUMENTS if d.doc_id == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not os.path.exists(doc.doc_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    if not doc.doc_txt:
        loader = PyPDFLoader(doc.doc_path)
        pages = loader.load()
        doc.doc_txt = "\n".join([p.page_content for p in pages])

    return doc

@app.get("/question-set/{doc_id}", response_model=QuestionSet)
def get_questions(doc_id: int):
    # extract doc_type using id
    # return questions for that type
    doc = next((d for d in DOCUMENTS if d.doc_id == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Find the question set that matches the document's type
    question_set = next((qs for qs in QUESTION_SETS if qs.doc_type == doc.doc_type), None)
    if not question_set:
        raise HTTPException(status_code=404, detail=f"No questions found for type '{doc.doc_type}'")

    return question_set

@app.post("/respond")
def respond(payload: ModelCall):
    model = LLM_INSTANCES[payload.model_id]
    response = model.invoke(payload.prompt)
    return {"output": response.content}