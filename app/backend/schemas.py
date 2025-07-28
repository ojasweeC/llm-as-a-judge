from pydantic import BaseModel

class Document(BaseModel):
    doc_id: int
    doc_name: str
    doc_type: str
    doc_path: str # to be converted to a URL
    doc_txt: str = ""
    
class QuestionSet(BaseModel):
    doc_type: str
    questions: list[str]
    
class LLMConfig(BaseModel):
    model_id: int
    model_name: str
    model_display_name: str
    is_judge: bool = False
    
class ModelCall(BaseModel):
    model_id: int
    prompt: str