from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import XMLOutputParser

import tempfile

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
