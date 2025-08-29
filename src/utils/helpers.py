from typing import Dict, List, Optional
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from googletrans import Translator

def load_medical_knowledge(pdf_path: str) -> Chroma:
    """Load and index medical knowledge from PDF"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=OpenAIEmbeddings(),
        persist_directory="db"
    )
    return vectorstore

def translate_text(text: str, target_lang: str = 'bn') -> str:
    """Translate text to target language"""
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text

def validate_patient_info(info: Dict) -> Dict[str, List[str]]:
    """Validate patient information and return any errors"""
    errors = {}
    required_fields = ['name', 'age', 'gender', 'symptoms']
    
    for field in required_fields:
        if field not in info:
            errors[field] = [f"{field} is required"]
        elif not info[field]:
            errors[field] = [f"{field} cannot be empty"]
            
    return errors

def format_medical_response(response: str, translate: bool = False) -> str:
    """Format medical response and optionally translate to Bangla"""
    formatted = response.strip()
    
    if translate:
        bangla = translate_text(formatted)
        return f"""English:
{formatted}

বাংলা:
{bangla}"""
    
    return formatted
