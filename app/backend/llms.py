from dotenv import load_dotenv, find_dotenv
from .schemas import LLMConfig
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

def instantiate_llms():
    load_dotenv(find_dotenv())

    llm_instances = {
        0: ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=1000),
        1: ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0, max_tokens=1000),
        2: ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.0, max_tokens=1000),
        3: ChatTogether(model_name="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.0, max_tokens=1000),
        4: ChatTogether(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", temperature=0.0, max_tokens=1000),
        5: ChatTogether(model_name="google/gemma-2-27b-it", temperature=0.0)
    }

    llm_configs = [
        LLMConfig(model_id=0, model_name="gpt-3.5-turbo", model_display_name="GPT-3.5 Turbo", is_judge=False),
        LLMConfig(model_id=1, model_name="claude-3-5-sonnet-latest", model_display_name="Claude Sonnet 3.5", is_judge=False),
        LLMConfig(model_id=2, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo", model_display_name="Meta Llama 3.3", is_judge=False),
        LLMConfig(model_id=3, model_name="mistralai/Mistral-7B-Instruct-v0.3", model_display_name="Mistral (7B) Instruct", is_judge=True),
        LLMConfig(model_id=4, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", model_display_name="DeepSeek R1 Distill Qwen", is_judge=True),
        LLMConfig(model_id=5, model_name="google/gemma-2-27b-it", model_display_name="Gemma 2 27B", is_judge=True)
    ]

    return llm_configs, llm_instances

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
    
def generate_respondent_prompt():
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
    
    return respondent_prompt
    
def generate_judge_prompt(criteria):
    judge_system = f"""You are a very strict evaluator that assesses the quality of multiple LLM's responses to document-based questions.
    You will be given:
    - The original document
    - A list of questions
    - The responses from three LLMs (labeled Model 1, Model 2, and Model 3) to each question

    Your task is to evaluate the quality of each model's response to **each question** using the following five criteria:
    {criteria}

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

    After evaluating each question, you will provide an overall accuracy percentage out of 100% for each model.

    ---

    After evaluating all questions, select the best-performing model overall.

    Use this exact format:

    Best Model: Model [number]  
    [One brief sentence explaining your choice]

    The chosen model should be the one with the highest and most consistent performance across all questions and criteria.
    """
    
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
    
    return judge_prompt